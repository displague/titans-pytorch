param(
    [string]$NanochatDir = "external/nanochat",
    [int]$Depth = 12,
    [int]$MaxSeqLen = 1024,
    [int]$DeviceBatchSize = 1,
    [int]$TotalBatchSize = 65536,
    [int]$NumIterations = 6000,
    [string[]]$Seeds = @("1337", "2026"),
    [string]$WindowPattern = "L",
    [double]$CandidateGateMix = 0.05,
    [bool]$CandidateOddLayersOnly = $true,
    [int]$CandidateGateStartIter = 16,
    [int]$CandidateGateRampIters = 32,
    [double]$CandidateWeightDecay = 0.2,
    [double]$CandidateMatrixLr = 0.02,
    [string]$RunLabel = "odd_sched16r32_mix005_n6000",
    [string]$OutputJson = "",
    [string]$OutputCsv = "",
    [bool]$RunQuickTests = $true,
    [string[]]$QuickPytestTargets = @("tests/test_attention_fallback.py"),
    [bool]$RunQuickEval = $true,
    [string]$EvalModes = "bpb,sample",
    [int]$EvalDeviceBatchSize = 4,
    [int]$EvalSplitTokens = 65536,
    [string]$EvalSeed = "",
    [string]$PostcheckJson = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}

function Invoke-NativeLogged {
    param(
        [string]$Executable,
        [string[]]$Arguments,
        [string]$LogPath
    )

    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Executable @Arguments *> $LogPath
        return $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $prev
    }
}

if ([string]::IsNullOrWhiteSpace($OutputJson)) {
    $OutputJson = "experiments/nanochat_transfer/results/nanochat_protocol_{0}_latest.json" -f $RunLabel
}
if ([string]::IsNullOrWhiteSpace($OutputCsv)) {
    $OutputCsv = "experiments/nanochat_transfer/results/nanochat_protocol_{0}_history.csv" -f $RunLabel
}
if ([string]::IsNullOrWhiteSpace($PostcheckJson)) {
    $PostcheckJson = "experiments/nanochat_transfer/results/nanochat_postcheck_latest.json"
}

if (-not [System.IO.Path]::IsPathRooted($OutputJson)) {
    $OutputJson = Join-Path $RepoRoot $OutputJson
}
if (-not [System.IO.Path]::IsPathRooted($OutputCsv)) {
    $OutputCsv = Join-Path $RepoRoot $OutputCsv
}
if (-not [System.IO.Path]::IsPathRooted($PostcheckJson)) {
    $PostcheckJson = Join-Path $RepoRoot $PostcheckJson
}

$resultsDir = Split-Path -Path $PostcheckJson -Parent
if ($resultsDir -and !(Test-Path $resultsDir)) {
    New-Item -ItemType Directory -Path $resultsDir -Force | Out-Null
}

$protocolScript = Join-Path $PSScriptRoot "run_nanochat_24h_protocol.ps1"
if (!(Test-Path $protocolScript)) {
    throw "Protocol script not found: $protocolScript"
}

$protocolArgs = @(
    "-ExecutionPolicy", "Bypass",
    "-File", $protocolScript,
    "-NanochatDir", $NanochatDir,
    "-ApplyCandidatePatch",
    "-Depth", "$Depth",
    "-MaxSeqLen", "$MaxSeqLen",
    "-DeviceBatchSize", "$DeviceBatchSize",
    "-TotalBatchSize", "$TotalBatchSize",
    "-NumIterations", "$NumIterations",
    "-Seeds", ($Seeds -join ","),
    "-WindowPattern", $WindowPattern,
    "-CandidateGateMix", "$CandidateGateMix",
    "-CandidateGateStartIter", "$CandidateGateStartIter",
    "-CandidateGateRampIters", "$CandidateGateRampIters",
    "-CandidateWeightDecay", "$CandidateWeightDecay",
    "-CandidateMatrixLr", "$CandidateMatrixLr",
    "-RunLabel", $RunLabel,
    "-OutputJson", $OutputJson,
    "-OutputCsv", $OutputCsv
)
if ($CandidateOddLayersOnly) {
    $protocolArgs += "-CandidateOddLayersOnly"
}

Write-Output "Running protocol: $protocolScript"
& powershell @protocolArgs
if ($LASTEXITCODE -ne 0) {
    throw "Protocol execution failed with exit code $LASTEXITCODE."
}

if (!(Test-Path $OutputJson)) {
    throw "Protocol summary was not produced: $OutputJson"
}
$protocolSummary = Get-Content $OutputJson -Raw | ConvertFrom-Json
if ($null -eq $protocolSummary -or $null -eq $protocolSummary.runs -or $protocolSummary.runs.Count -eq 0) {
    throw "Protocol summary did not contain run records."
}

$seedForEval = $EvalSeed
if ([string]::IsNullOrWhiteSpace($seedForEval)) {
    $seedForEval = [string]$protocolSummary.runs[0].seed
}

$controlRun = $protocolSummary.runs | Where-Object { $_.recipe -eq "control" -and [string]$_.seed -eq [string]$seedForEval } | Select-Object -First 1
$candidateRun = $protocolSummary.runs | Where-Object { $_.recipe -eq "candidate_slot" -and [string]$_.seed -eq [string]$seedForEval } | Select-Object -First 1
if ($null -eq $controlRun -or $null -eq $candidateRun) {
    throw "Could not resolve control/candidate run tags for seed '$seedForEval'."
}

$postcheck = [ordered]@{
    timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    run_label = $RunLabel
    num_iterations = $NumIterations
    seed_for_eval = [string]$seedForEval
    protocol_output_json = $OutputJson
    protocol_output_csv = $OutputCsv
    quick_tests = $null
    eval_runs = @()
    candidate_minus_control_val_bpb = $null
}

$nanochatAbs = (Resolve-Path $NanochatDir).Path
Push-Location $nanochatAbs
try {
    $python = Join-Path ".venv" "Scripts/python.exe"
    if (!(Test-Path $python)) {
        $python = "python"
    }
    $env:PYTHONUTF8 = "1"
    $env:PYTHONIOENCODING = "utf-8"

    if ($RunQuickTests -and $QuickPytestTargets.Count -gt 0) {
        $testLog = Join-Path $resultsDir ("nanochat_quick_tests_{0}.log" -f $RunLabel)
        $testArgs = @("-m", "pytest", "-q") + $QuickPytestTargets
        $testExit = Invoke-NativeLogged -Executable $python -Arguments $testArgs -LogPath $testLog
        $postcheck.quick_tests = [ordered]@{
            command = "python -m pytest -q {0}" -f ($QuickPytestTargets -join " ")
            exit_code = $testExit
            log_path = $testLog
        }
        if ($testExit -ne 0) {
            throw "Quick tests failed. See $testLog"
        }
    }

    if ($RunQuickEval) {
        foreach ($evalTarget in @(
            @{ recipe = "control"; run_tag = [string]$controlRun.run_tag },
            @{ recipe = "candidate_slot"; run_tag = [string]$candidateRun.run_tag }
        )) {
            $evalLog = Join-Path $resultsDir ("nanochat_base_eval_{0}_{1}.log" -f $RunLabel, $evalTarget.recipe)
            $evalArgs = @(
                "-m", "scripts.base_eval",
                "--model-tag", $evalTarget.run_tag,
                "--eval", $EvalModes,
                "--device-batch-size", "$EvalDeviceBatchSize",
                "--split-tokens", "$EvalSplitTokens"
            )
            $evalExit = Invoke-NativeLogged -Executable $python -Arguments $evalArgs -LogPath $evalLog
            $evalText = if (Test-Path $evalLog) { Get-Content $evalLog -Raw } else { "" }
            $trainMatch = [regex]::Match($evalText, "train bpb:\s*([0-9]+\.[0-9]+)")
            $valMatch = [regex]::Match($evalText, "val bpb:\s*([0-9]+\.[0-9]+)")
            $trainBpb = $null
            $valBpb = $null
            if ($trainMatch.Success) {
                $trainBpb = [double]$trainMatch.Groups[1].Value
            }
            if ($valMatch.Success) {
                $valBpb = [double]$valMatch.Groups[1].Value
            }

            $postcheck.eval_runs += [ordered]@{
                recipe = $evalTarget.recipe
                run_tag = $evalTarget.run_tag
                exit_code = $evalExit
                eval_modes = $EvalModes
                train_bpb = $trainBpb
                val_bpb = $valBpb
                log_path = $evalLog
            }

            if ($evalExit -ne 0) {
                throw "base_eval failed for run '$($evalTarget.run_tag)'. See $evalLog"
            }
        }
    }
}
finally {
    Pop-Location
}

$controlEval = $postcheck.eval_runs | Where-Object { $_.recipe -eq "control" } | Select-Object -First 1
$candidateEval = $postcheck.eval_runs | Where-Object { $_.recipe -eq "candidate_slot" } | Select-Object -First 1
if ($null -ne $controlEval -and $null -ne $candidateEval -and $null -ne $controlEval.val_bpb -and $null -ne $candidateEval.val_bpb) {
    $postcheck.candidate_minus_control_val_bpb = [double]$candidateEval.val_bpb - [double]$controlEval.val_bpb
}

$postcheck | ConvertTo-Json -Depth 8 | Set-Content -Path $PostcheckJson -Encoding UTF8
Write-Output "Saved postcheck summary: $PostcheckJson"
