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

# Fast content-based check for idempotence.
$gptPath = Join-Path $NanochatDir "nanochat/gpt.py"
$trainPath = Join-Path $NanochatDir "scripts/base_train.py"
$alreadyPatched = $false
if ((Test-Path $gptPath) -and (Test-Path $trainPath)) {
    $hasGateConfig = Select-String -Path $gptPath -Pattern "symplectic_gate_enabled" -Quiet
    $hasTrainFlag = Select-String -Path $trainPath -Pattern "--symplectic-gate-enabled" -Quiet
    $alreadyPatched = $hasGateConfig -and $hasTrainFlag
}

if ($alreadyPatched) {
    Write-Output "Patch already applied."
    exit 0
}

& git -C $NanochatDir apply --ignore-space-change --ignore-whitespace $PatchPath
if ($LASTEXITCODE -ne 0) {
    throw "Failed to apply candidate patch."
}

Write-Output "Applied candidate patch to $NanochatDir"
