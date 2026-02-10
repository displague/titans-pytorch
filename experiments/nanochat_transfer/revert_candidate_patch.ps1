param(
    [string]$NanochatDir = "external/nanochat",
    [string]$PatchPath = ""
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $NanochatDir)) {
    throw "nanochat not found at '$NanochatDir'."
}

if ([string]::IsNullOrWhiteSpace($PatchPath)) {
    $PatchPath = Join-Path $PSScriptRoot "patches/nanochat_symplectic_candidate.patch"
}

if (!(Test-Path $PatchPath)) {
    throw "Patch file not found at '$PatchPath'."
}

# Content-based check first.
$gptPath = Join-Path $NanochatDir "nanochat/gpt.py"
$trainPath = Join-Path $NanochatDir "scripts/base_train.py"
$hasGateConfig = (Test-Path $gptPath) -and (Select-String -Path $gptPath -Pattern "symplectic_gate_enabled" -Quiet)
$hasTrainFlag = (Test-Path $trainPath) -and (Select-String -Path $trainPath -Pattern "--symplectic-gate-enabled" -Quiet)

if (!($hasGateConfig -and $hasTrainFlag)) {
    Write-Output "Patch does not appear to be applied."
    exit 0
}

& git -C $NanochatDir restore --source=HEAD --worktree -- "nanochat/gpt.py" "scripts/base_train.py"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to revert candidate patch."
}

Write-Output "Reverted candidate patch from $NanochatDir"
