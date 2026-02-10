param(
    [string]$NanochatDir = "external/nanochat",
    [string]$PatchPath = "",
    [switch]$ForceReapply
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $NanochatDir)) {
    throw "nanochat not found at '$NanochatDir'."
}
$NanochatDir = (Resolve-Path $NanochatDir).Path

if ([string]::IsNullOrWhiteSpace($PatchPath)) {
    $PatchPath = Join-Path $PSScriptRoot "patches/nanochat_symplectic_candidate.patch"
}

if (!(Test-Path $PatchPath)) {
    throw "Patch file not found at '$PatchPath'."
}
$PatchPath = (Resolve-Path $PatchPath).Path

# Content checks for idempotence and patch version upgrades.
$gptPath = Join-Path $NanochatDir "nanochat/gpt.py"
$trainPath = Join-Path $NanochatDir "scripts/base_train.py"
if ((Test-Path $gptPath) -and (Test-Path $trainPath)) {
    $hasGateConfig = Select-String -Path $gptPath -Pattern "symplectic_gate_enabled" -Quiet
    $hasTokenGateFn = Select-String -Path $gptPath -Pattern "def symplectic_token_gate" -Quiet
    $hasOddLayerConfig = Select-String -Path $gptPath -Pattern "symplectic_gate_odd_layers_only" -Quiet
    $hasTrainFlag = Select-String -Path $trainPath -Pattern "--symplectic-gate-enabled" -Quiet
    $hasOddLayerFlag = Select-String -Path $trainPath -Pattern "--symplectic-gate-odd-layers-only" -Quiet

    $hasCurrentPatch = $hasGateConfig -and $hasTokenGateFn -and $hasOddLayerConfig -and $hasTrainFlag -and $hasOddLayerFlag
    $hasLegacyPatch = $hasGateConfig -and $hasTrainFlag -and !$hasCurrentPatch

    if ($hasCurrentPatch -and -not $ForceReapply) {
        Write-Output "Patch already applied."
        exit 0
    }

    if ($ForceReapply -or $hasLegacyPatch) {
        if ($hasLegacyPatch) {
            Write-Output "Legacy candidate patch detected, refreshing to latest patch revision."
        }
        & git -C $NanochatDir restore --source=HEAD --worktree -- "nanochat/gpt.py" "scripts/base_train.py"
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to reset target files before patch apply."
        }
    }
}

& git -C $NanochatDir apply --check --ignore-space-change --ignore-whitespace $PatchPath
if ($LASTEXITCODE -ne 0) {
    throw "Patch cannot be applied cleanly. Ensure external/nanochat has no conflicting edits in nanochat/gpt.py or scripts/base_train.py."
}
& git -C $NanochatDir apply --ignore-space-change --ignore-whitespace $PatchPath
if ($LASTEXITCODE -ne 0) {
    throw "Failed to apply candidate patch."
}

Write-Output "Applied candidate patch to $NanochatDir"
