param(
    [string]$NanochatDir = "external/nanochat",
    [switch]$PrepareData,
    [switch]$ApplyCandidatePatch,
    [int]$Depth = 12,
    [int]$MaxSeqLen = 1024,
    [int]$DeviceBatchSize = 1,
    [int]$TotalBatchSize = 65536,
    [int]$NumIterations = 30000,
    [string[]]$Seeds = @("1337", "2026"),
    [string]$WindowPattern = "L",
    [int]$TokenizerMaxChars = 2000000000,
    [bool]$AutoPrepareIfMissing = $true,
    [bool]$DisableTorchCompile = $true,
    [double]$CandidateGateMix = 0.15,
    [switch]$CandidateOddLayersOnly,
    [int]$CandidateGateStartIter = 0,
    [int]$CandidateGateRampIters = 0,
    [double]$CandidateWeightDecay = 0.18,
    [double]$CandidateMatrixLr = 0.018,
    [int]$SaveEvery = 2000,
    [switch]$ContinueFromLatest,
    [string]$RunLabel = "",
    [string]$OutputJson = "experiments/nanochat_transfer/results/nanochat_protocol_latest.json",
    [string]$OutputCsv = "experiments/nanochat_transfer/results/nanochat_protocol_history.csv"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

if (-not [System.IO.Path]::IsPathRooted($OutputJson)) {
    $OutputJson = Join-Path $RepoRoot $OutputJson
}
if (-not [System.IO.Path]::IsPathRooted($OutputCsv)) {
    $OutputCsv = Join-Path $RepoRoot $OutputCsv
}

function Invoke-ExternalCommand {
    param(
        [string]$Description,
        [string]$Executable,
        [string[]]$Arguments
    )

    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE."
    }
}

function Read-LatestCheckpointMeta {
    param([string]$CheckpointDir)

    if (!(Test-Path $CheckpointDir)) {
        return $null
    }

    $metaFile = Get-ChildItem -Path $CheckpointDir -Filter "meta_*.json" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime, Name |
        Select-Object -Last 1
    if ($null -eq $metaFile) {
        return $null
    }

    try {
        return Get-Content -Path $metaFile.FullName -Raw | ConvertFrom-Json
    }
    catch {
        return $null
    }
}

if (!(Test-Path $NanochatDir)) {
    throw "nanochat not found at '$NanochatDir'. Run setup_nanochat.ps1 first."
}
$NanochatDir = (Resolve-Path $NanochatDir).Path
$normalizedSeeds = @()
foreach ($seedEntry in $Seeds) {
    foreach ($seedPart in ($seedEntry -split ",")) {
        $trimmedSeed = $seedPart.Trim()
        if (![string]::IsNullOrWhiteSpace($trimmedSeed)) {
            $normalizedSeeds += $trimmedSeed
        }
    }
}
if ($normalizedSeeds.Count -eq 0) {
    throw "At least one seed is required."
}
$Seeds = $normalizedSeeds
if ($CandidateGateMix -lt 0 -or $CandidateGateMix -gt 1) {
    throw "CandidateGateMix must be in [0, 1]."
}
if ($CandidateWeightDecay -le 0) {
    throw "CandidateWeightDecay must be > 0."
}
if ($CandidateMatrixLr -le 0) {
    throw "CandidateMatrixLr must be > 0."
}
if ($CandidateGateStartIter -lt 0) {
    throw "CandidateGateStartIter must be >= 0."
}
if ($CandidateGateRampIters -lt 0) {
    throw "CandidateGateRampIters must be >= 0."
}
if ($SaveEvery -le 0) {
    throw "SaveEvery must be > 0."
}

Push-Location $NanochatDir
try {
    if (!(Get-Command uv -ErrorAction SilentlyContinue)) {
        throw "uv was not found in PATH."
    }

    if (!(Test-Path ".venv")) {
        Invoke-ExternalCommand -Description "uv venv" -Executable "uv" -Arguments @("venv")
    }
    Invoke-ExternalCommand -Description "uv sync --extra gpu" -Executable "uv" -Arguments @("sync", "--extra", "gpu")

    $python = Join-Path ".venv" "Scripts/python.exe"
    if (!(Test-Path $python)) {
        $python = "python"
    }
    $baseDirOutput = & $python -c "from nanochat.common import get_base_dir; print(get_base_dir())"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to resolve nanochat base directory."
    }
    $baseDirLine = ($baseDirOutput | Select-Object -Last 1)
    $nanochatBaseDir = $baseDirLine.Trim()
    if ([string]::IsNullOrWhiteSpace($nanochatBaseDir)) {
        throw "Resolved empty nanochat base directory."
    }

    $tokenizerPath = Join-Path $env:USERPROFILE ".cache\nanochat\tokenizer\tokenizer.pkl"
    $shouldPrepareData = [bool]$PrepareData
    if (!$shouldPrepareData -and $AutoPrepareIfMissing -and !(Test-Path $tokenizerPath)) {
        Write-Output "Tokenizer missing at '$tokenizerPath'. Running prepare steps."
        $shouldPrepareData = $true
    }
    if ($shouldPrepareData) {
        Invoke-ExternalCommand -Description "nanochat dataset prep" -Executable $python -Arguments @("-m", "nanochat.dataset", "-n", "8")
        Invoke-ExternalCommand -Description "nanochat tokenizer train" -Executable $python -Arguments @("-m", "scripts.tok_train", "--max-chars=$TokenizerMaxChars")
        Invoke-ExternalCommand -Description "nanochat tokenizer eval" -Executable $python -Arguments @("-m", "scripts.tok_eval")
    }
    elseif (!(Test-Path $tokenizerPath)) {
        throw "Tokenizer not found at '$tokenizerPath'. Run with -PrepareData."
    }

    if ($ApplyCandidatePatch) {
        & powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "apply_candidate_patch.ps1") -NanochatDir $NanochatDir
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to apply candidate patch."
        }
    }

    $baseTrainPath = Join-Path $NanochatDir "scripts/base_train.py"
    $hasCandidateFlags = Select-String -Path $baseTrainPath -Pattern "--symplectic-gate-enabled" -Quiet
    if (!$hasCandidateFlags) {
        throw "Candidate gate CLI flags are not available in nanochat. Run with -ApplyCandidatePatch."
    }
    if ($CandidateOddLayersOnly) {
        $hasOddLayerFlag = Select-String -Path $baseTrainPath -Pattern "--symplectic-gate-odd-layers-only" -Quiet
        if (!$hasOddLayerFlag) {
            throw "Odd-layer candidate flag is not available in nanochat. Refresh patch with apply_candidate_patch.ps1."
        }
    }
    if ($CandidateGateStartIter -gt 0 -or $CandidateGateRampIters -gt 0) {
        $hasStartFlag = Select-String -Path $baseTrainPath -Pattern "--symplectic-gate-start-iter" -Quiet
        $hasRampFlag = Select-String -Path $baseTrainPath -Pattern "--symplectic-gate-ramp-iters" -Quiet
        if (!$hasStartFlag -or !$hasRampFlag) {
            throw "Gate schedule flags are not available in nanochat. Refresh patch with apply_candidate_patch.ps1."
        }
    }

    $env:OMP_NUM_THREADS = "1"
    $env:PYTHONUTF8 = "1"
    $env:PYTHONIOENCODING = "utf-8"
    $env:WANDB_CONSOLE = "off"
    if ($DisableTorchCompile) {
        $env:TORCHDYNAMO_DISABLE = "1"
        $env:TORCHINDUCTOR_DISABLE = "1"
    }

    $candidateExtra = @(
        "--window-pattern=$WindowPattern",
        "--weight-decay=$CandidateWeightDecay",
        "--matrix-lr=$CandidateMatrixLr",
        "--symplectic-gate-enabled",
        "--symplectic-gate-mix=$CandidateGateMix"
    )
    if ($CandidateOddLayersOnly) {
        $candidateExtra += "--symplectic-gate-odd-layers-only"
    }
    if ($CandidateGateStartIter -gt 0) {
        $candidateExtra += "--symplectic-gate-start-iter=$CandidateGateStartIter"
    }
    if ($CandidateGateRampIters -gt 0) {
        $candidateExtra += "--symplectic-gate-ramp-iters=$CandidateGateRampIters"
    }

    $recipes = @(
        @{
            Name = "control"
            Extra = @("--window-pattern=$WindowPattern")
        },
        @{
            # Candidate slot for Titans-inspired transfer changes.
            # Requires apply_candidate_patch.ps1.
            Name = "candidate_slot"
            Extra = $candidateExtra
        }
    )

    $runResults = @()

    foreach ($seed in $Seeds) {
        foreach ($recipe in $recipes) {
            $recipeName = $recipe["Name"]
            if ([string]::IsNullOrWhiteSpace($RunLabel)) {
                $runTag = "nc_d${Depth}_${recipeName}_s${seed}"
            }
            else {
                $runTag = "nc_${RunLabel}_d${Depth}_${recipeName}_s${seed}"
            }
            $checkpointDir = Join-Path (Join-Path $nanochatBaseDir "base_checkpoints") $runTag
            $resumeFromStep = -1
            $resumeMeta = $null
            $isSkippedCompleted = $false
            if ($ContinueFromLatest) {
                $resumeMeta = Read-LatestCheckpointMeta -CheckpointDir $checkpointDir
                if ($null -ne $resumeMeta -and ($resumeMeta.PSObject.Properties.Name -contains "step")) {
                    $savedStep = [int]$resumeMeta.step
                    if ($savedStep -ge $NumIterations) {
                        $isSkippedCompleted = $true
                    }
                    elseif ($savedStep -gt 0) {
                        $resumeFromStep = $savedStep
                    }
                }
            }

            if ($isSkippedCompleted) {
                $valBpb = $null
                $minValBpb = $null
                $durationSec = $null
                if ($resumeMeta.PSObject.Properties.Name -contains "val_bpb") {
                    $valBpb = [double]$resumeMeta.val_bpb
                }
                if ($resumeMeta.PSObject.Properties.Name -contains "loop_state") {
                    $loopState = $resumeMeta.loop_state
                    if ($null -ne $loopState -and $loopState.PSObject.Properties.Name -contains "min_val_bpb") {
                        $minValBpb = [double]$loopState.min_val_bpb
                    }
                    if ($null -ne $loopState -and $loopState.PSObject.Properties.Name -contains "total_training_time") {
                        $durationSec = [double]$loopState.total_training_time
                    }
                }
                $trainedTokens = [double]($NumIterations * $TotalBatchSize)
                $avgTokPerSec = $null
                if ($null -ne $durationSec -and $durationSec -gt 0) {
                    $avgTokPerSec = $trainedTokens / $durationSec
                }
                Write-Output "Skipping completed $runTag (step >= $NumIterations)."
                $runResults += [PSCustomObject]@{
                    timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
                    run_tag = $runTag
                    recipe = $recipeName
                    seed = [string]$seed
                    duration_sec = $durationSec
                    trained_tokens = $trainedTokens
                    avg_tok_per_sec = $avgTokPerSec
                    val_bpb = $valBpb
                    min_val_bpb = $minValBpb
                    num_iterations = $NumIterations
                    window_pattern = $WindowPattern
                    resumed_from_step = [int]$NumIterations
                    run_status = "skipped_completed"
                }
                continue
            }

            if ($resumeFromStep -gt 0) {
                Write-Output "Starting $runTag (resume from step $resumeFromStep)"
            }
            else {
                Write-Output "Starting $runTag"
            }

            $baseArgs = @(
                "--depth=$Depth",
                "--run=dummy",
                "--model-tag=$runTag",
                "--max-seq-len=$MaxSeqLen",
                "--device-batch-size=$DeviceBatchSize",
                "--total-batch-size=$TotalBatchSize",
                "--eval-every=200",
                "--eval-tokens=262144",
                "--core-metric-every=-1",
                "--sample-every=-1",
                "--save-every=$SaveEvery",
                "--num-iterations=$NumIterations"
            )
            if ($resumeFromStep -gt 0) {
                $baseArgs += "--resume-from-step=$resumeFromStep"
            }

            $cmdArgs = @("-m", "scripts.base_train") + $baseArgs + $recipe["Extra"]
            $timer = [System.Diagnostics.Stopwatch]::StartNew()
            & $python @cmdArgs
            $exitCode = $LASTEXITCODE
            $timer.Stop()
            if ($exitCode -ne 0) {
                throw "Run '$runTag' failed with exit code $exitCode."
            }

            $meta = Read-LatestCheckpointMeta -CheckpointDir $checkpointDir
            $valBpb = $null
            $minValBpb = $null
            if ($null -ne $meta) {
                if ($meta.PSObject.Properties.Name -contains "val_bpb") {
                    $valBpb = [double]$meta.val_bpb
                }
                if ($meta.PSObject.Properties.Name -contains "loop_state") {
                    $loopState = $meta.loop_state
                    if ($null -ne $loopState -and $loopState.PSObject.Properties.Name -contains "min_val_bpb") {
                        $minValBpb = [double]$loopState.min_val_bpb
                    }
                }
            }

            $durationSec = [Math]::Round($timer.Elapsed.TotalSeconds, 3)
            $trainedTokens = [double](($NumIterations - [Math]::Max(0, $resumeFromStep)) * $TotalBatchSize)
            $avgTokPerSec = $null
            if ($durationSec -gt 0) {
                $avgTokPerSec = $trainedTokens / $durationSec
            }
            Write-Output "Completed $runTag | duration_sec=$durationSec | val_bpb=$valBpb | min_val_bpb=$minValBpb"
            $runResults += [PSCustomObject]@{
                timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
                run_tag = $runTag
                recipe = $recipeName
                seed = [string]$seed
                duration_sec = $durationSec
                trained_tokens = $trainedTokens
                avg_tok_per_sec = $avgTokPerSec
                val_bpb = $valBpb
                min_val_bpb = $minValBpb
                num_iterations = $NumIterations
                window_pattern = $WindowPattern
                resumed_from_step = [Math]::Max(0, $resumeFromStep)
                run_status = "completed"
            }
        }
    }

    $deltaRows = @()
    foreach ($seed in $Seeds) {
        $control = $runResults | Where-Object { $_.seed -eq [string]$seed -and $_.recipe -eq "control" } | Select-Object -First 1
        $candidate = $runResults | Where-Object { $_.seed -eq [string]$seed -and $_.recipe -eq "candidate_slot" } | Select-Object -First 1
        if ($null -ne $control -and $null -ne $candidate -and $null -ne $control.val_bpb -and $null -ne $candidate.val_bpb) {
            $speedRatio = $null
            $durationDelta = $null
            if ($null -ne $control.avg_tok_per_sec -and $control.avg_tok_per_sec -gt 0 -and $null -ne $candidate.avg_tok_per_sec) {
                $speedRatio = [double]$candidate.avg_tok_per_sec / [double]$control.avg_tok_per_sec
            }
            if ($null -ne $control.duration_sec -and $null -ne $candidate.duration_sec) {
                $durationDelta = [double]$candidate.duration_sec - [double]$control.duration_sec
            }
            $deltaRows += [PSCustomObject]@{
                seed = [string]$seed
                candidate_minus_control_bpb = [double]$candidate.val_bpb - [double]$control.val_bpb
                candidate_minus_control_duration_sec = $durationDelta
                candidate_speed_ratio = $speedRatio
            }
        }
    }

    $meanDelta = $null
    $meanSpeedRatio = $null
    if ($deltaRows.Count -gt 0) {
        $meanDelta = ($deltaRows | Measure-Object -Property candidate_minus_control_bpb -Average).Average
        $meanSpeedRatio = ($deltaRows | Where-Object { $null -ne $_.candidate_speed_ratio } | Measure-Object -Property candidate_speed_ratio -Average).Average
    }
    Write-Output "Mean candidate minus control bpb: $meanDelta"
    Write-Output "Mean candidate speed ratio: $meanSpeedRatio"

    $summary = [PSCustomObject]@{
        timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        settings = [PSCustomObject]@{
            depth = $Depth
            max_seq_len = $MaxSeqLen
            device_batch_size = $DeviceBatchSize
            total_batch_size = $TotalBatchSize
            num_iterations = $NumIterations
            seeds = $Seeds
            window_pattern = $WindowPattern
            candidate_patch_applied = [bool]$ApplyCandidatePatch
            candidate_gate_mix = $CandidateGateMix
            candidate_odd_layers_only = [bool]$CandidateOddLayersOnly
            candidate_gate_start_iter = $CandidateGateStartIter
            candidate_gate_ramp_iters = $CandidateGateRampIters
            candidate_weight_decay = $CandidateWeightDecay
            candidate_matrix_lr = $CandidateMatrixLr
            save_every = $SaveEvery
            continue_from_latest = [bool]$ContinueFromLatest
            run_label = $RunLabel
        }
        runs = $runResults
        deltas = $deltaRows
        mean_candidate_minus_control_bpb = $meanDelta
        mean_candidate_speed_ratio = $meanSpeedRatio
    }

    $jsonDir = Split-Path -Path $OutputJson -Parent
    if ($jsonDir -and !(Test-Path $jsonDir)) {
        New-Item -ItemType Directory -Path $jsonDir -Force | Out-Null
    }
    $csvDir = Split-Path -Path $OutputCsv -Parent
    if ($csvDir -and !(Test-Path $csvDir)) {
        New-Item -ItemType Directory -Path $csvDir -Force | Out-Null
    }

    $summary | ConvertTo-Json -Depth 8 | Set-Content -Path $OutputJson -Encoding UTF8
    Write-Output "Saved protocol summary: $OutputJson"

    if (Test-Path $OutputCsv) {
        $runResults | Export-Csv -Path $OutputCsv -NoTypeInformation -Append
    }
    else {
        $runResults | Export-Csv -Path $OutputCsv -NoTypeInformation
    }
    Write-Output "Appended protocol history: $OutputCsv"
}
finally {
    Pop-Location
}
