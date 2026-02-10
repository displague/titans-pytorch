param(
    [string]$NanochatDir = "external/nanochat",
    [switch]$PrepareData,
    [int]$Depth = 12,
    [int]$MaxSeqLen = 1024,
    [int]$DeviceBatchSize = 1,
    [int]$TotalBatchSize = 65536,
    [int]$NumIterations = 30000,
    [string[]]$Seeds = @("1337", "2026")
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

    $env:OMP_NUM_THREADS = "1"

    $recipes = @(
        @{
            Name = "control"
            Extra = @("--window-pattern=SSSL")
        },
        @{
            # Candidate slot for Titans-inspired transfer changes.
            # Keep command valid even before any source patching.
            Name = "candidate_slot"
            Extra = @("--window-pattern=SSSL", "--weight-decay=0.18", "--matrix-lr=0.018")
        }
    )

    foreach ($seed in $Seeds) {
        foreach ($recipe in $recipes) {
            $runTag = "nc_d${Depth}_${($recipe.Name)}_s${seed}"
            Write-Output "Starting $runTag"

            $baseArgs = @(
                "--depth=$Depth",
                "--run=$runTag",
                "--model-tag=$runTag",
                "--max-seq-len=$MaxSeqLen",
                "--device-batch-size=$DeviceBatchSize",
                "--total-batch-size=$TotalBatchSize",
                "--eval-every=200",
                "--eval-tokens=262144",
                "--core-metric-every=-1",
                "--sample-every=-1",
                "--save-every=2000",
                "--num-iterations=$NumIterations",
                "--seed=$seed"
            )

            $cmdArgs = @("--standalone", "--nproc_per_node=1", "-m", "scripts.base_train", "--") + $baseArgs + $recipe.Extra
            & torchrun @cmdArgs
        }
    }
}
finally {
    Pop-Location
}
