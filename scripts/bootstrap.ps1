$ErrorActionPreference = "Stop"

uv venv .venv --python 3.13
& .\.venv\Scripts\Activate.ps1
uv pip install -e ".[dev,test,eval]"
Write-Host "Run commands with `$env:PYTHONPATH = (Resolve-Path .\src).Path; python -m decomp_clarifier.cli ..."
