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
    [string]$RunTag = "nanochat_16gb_smoke"
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $NanochatDir)) {
    throw "nanochat not found at '$NanochatDir'. Run setup_nanochat.ps1 first."
}

Push-Location $NanochatDir
try {
    if (!(Get-Command uv -ErrorAction SilentlyContinue)) {
        throw "uv was not found in PATH."
    }

    if (!(Test-Path ".venv")) {
        uv venv
    }
    uv sync --extra gpu

    $python = Join-Path ".venv" "Scripts/python.exe"
    if (!(Test-Path $python)) {
        $python = "python"
    }

    if ($PrepareData) {
        & $python -m nanochat.dataset -n 8
        & $python -m scripts.tok_train --max-chars=2000000000
        & $python -m scripts.tok_eval
    }

    if ($ApplyCandidatePatch) {
        & powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "apply_candidate_patch.ps1") -NanochatDir $NanochatDir
    }

    $env:OMP_NUM_THREADS = "1"

    $trainArgs = @(
        "--depth=$Depth",
        "--run=$RunTag",
        "--model-tag=$RunTag",
        "--max-seq-len=$MaxSeqLen",
        "--device-batch-size=$DeviceBatchSize",
        "--total-batch-size=$TotalBatchSize",
        "--eval-every=100",
        "--eval-tokens=262144",
        "--core-metric-every=-1",
        "--sample-every=-1",
        "--save-every=-1",
        "--num-iterations=$NumIterations"
    )

    if ($EnableCandidateGate) {
        $trainArgs += @(
            "--symplectic-gate-enabled",
            "--symplectic-gate-mix=$CandidateMix"
        )
    }

    $cmdArgs = @("-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1", "-m", "scripts.base_train", "--") + $trainArgs
    & $python @cmdArgs
}
finally {
    Pop-Location
}
