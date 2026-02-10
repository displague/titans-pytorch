param(
    [string]$NanochatDir = "external/nanochat",
    [string]$RepoUrl = "https://github.com/karpathy/nanochat"
)

$ErrorActionPreference = "Stop"

if (!(Test-Path "external")) {
    New-Item -ItemType Directory -Path "external" | Out-Null
}

if (!(Test-Path $NanochatDir)) {
    git clone $RepoUrl $NanochatDir
} else {
    git -C $NanochatDir fetch origin
    git -C $NanochatDir pull --ff-only
}

$commit = git -C $NanochatDir rev-parse --short HEAD
$branch = git -C $NanochatDir rev-parse --abbrev-ref HEAD
Write-Output "nanochat ready: branch=$branch commit=$commit path=$NanochatDir"
