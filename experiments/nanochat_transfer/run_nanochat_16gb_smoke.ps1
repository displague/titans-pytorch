param(
    [string]$NanochatDir = "external/nanochat",
    [switch]$PrepareData,
    [switch]$ApplyCandidatePatch,
    [switch]$EnableCandidateGate,
    [double]$CandidateMix = 0.15,
    [int]$Depth = 12,
    [int]$MaxSeqLen = 512,
    [int]$DeviceBatchSize = 1,
    [int]$TotalBatchSize = 32768,
    [int]$NumIterations = 200,
    [string]$RunTag = "nanochat_16gb_smoke",
    [string]$WindowPattern = "L",
    [int]$TokenizerMaxChars = 2000000000,
    [bool]$AutoPrepareIfMissing = $true,
    [bool]$DisableTorchCompile = $true
)

$ErrorActionPreference = "Stop"

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

if (!(Test-Path $NanochatDir)) {
    throw "nanochat not found at '$NanochatDir'. Run setup_nanochat.ps1 first."
}
$NanochatDir = (Resolve-Path $NanochatDir).Path

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

    if ($EnableCandidateGate) {
        $baseTrainPath = Join-Path $NanochatDir "scripts/base_train.py"
        $hasCandidateFlags = Select-String -Path $baseTrainPath -Pattern "--symplectic-gate-enabled" -Quiet
        if (!$hasCandidateFlags) {
            throw "Candidate gate CLI flags are not available in nanochat. Run with -ApplyCandidatePatch."
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

    $trainArgs = @(
        "--depth=$Depth",
        "--run=dummy",
        "--model-tag=$RunTag",
        "--max-seq-len=$MaxSeqLen",
        "--device-batch-size=$DeviceBatchSize",
        "--total-batch-size=$TotalBatchSize",
        "--eval-every=100",
        "--eval-tokens=262144",
        "--core-metric-every=-1",
        "--sample-every=-1",
        "--save-every=-1",
        "--num-iterations=$NumIterations",
        "--window-pattern=$WindowPattern"
    )

    if ($EnableCandidateGate) {
        $trainArgs += @(
            "--symplectic-gate-enabled",
            "--symplectic-gate-mix=$CandidateMix"
        )
    }

    $cmdArgs = @("-m", "scripts.base_train") + $trainArgs
    $timer = [System.Diagnostics.Stopwatch]::StartNew()
    & $python @cmdArgs
    $exitCode = $LASTEXITCODE
    $timer.Stop()
    if ($exitCode -ne 0) {
        throw "Smoke run failed with exit code $exitCode."
    }
    Write-Output "Completed smoke run '$RunTag' in $([Math]::Round($timer.Elapsed.TotalSeconds, 3)) sec."
}
finally {
    Pop-Location
}
