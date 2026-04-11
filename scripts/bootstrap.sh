#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

uv venv .venv --python 3.13
source .venv/bin/activate
uv pip install -e ".[dev,test,eval]"
echo "Run commands with: PYTHONPATH=src python -m decomp_clarifier.cli ..."
echo "Verify the core environment with: python -m decomp_clarifier.cli doctor"
echo "For Windows CUDA training, use scripts/bootstrap.ps1 -Training on a native Windows host."
