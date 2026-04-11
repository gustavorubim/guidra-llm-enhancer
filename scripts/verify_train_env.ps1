$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

Push-Location $repoRoot
try {
    $python = Resolve-Path .\.venv\Scripts\python.exe
    $env:PYTHONPATH = (Resolve-Path .\src).Path

    & $python.Path -m decomp_clarifier.cli doctor --training $args
} finally {
    Pop-Location
}
