param(
    [switch]$Training
)

$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$torchVersion = "2.10.0"
$torchvisionVersion = "0.25.0"
$pytorchCudaIndex = "https://download.pytorch.org/whl/cu128"

Push-Location $repoRoot
try {
    $extras = ".[dev,test,eval]"
    if ($Training) {
        $extras = ".[dev,test,eval,train-windows-cuda]"
    }

    uv venv .venv --python 3.13
    & .\.venv\Scripts\Activate.ps1
    uv pip install -e $extras

    if ($Training) {
        # Installing from the default index can still yield a CPU-only torch wheel.
        # Force the pinned Windows CUDA build used by the training stack.
        uv pip install --python .\.venv\Scripts\python.exe --reinstall `
            "torch==$torchVersion" `
            "torchvision==$torchvisionVersion" `
            --index-url $pytorchCudaIndex
    }

    if ($Training) {
        Write-Host "Run commands with `$env:PYTHONPATH = (Resolve-Path .\src).Path; python -m decomp_clarifier.cli ..."
        Write-Host "Verify the Windows CUDA training stack with: .\\scripts\\verify_train_env.ps1"
    } else {
        Write-Host "Run commands with `$env:PYTHONPATH = (Resolve-Path .\src).Path; python -m decomp_clarifier.cli ..."
        Write-Host "Verify the core environment with: python -m decomp_clarifier.cli doctor"
    }
} finally {
    Pop-Location
}
