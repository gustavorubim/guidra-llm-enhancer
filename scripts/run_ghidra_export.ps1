$ErrorActionPreference = "Stop"

& .\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m decomp_clarifier.cli export-ghidra $args
