#!/usr/bin/env bash
set -euo pipefail

uv venv .venv --python 3.13
source .venv/bin/activate
uv pip install -e ".[dev,test,eval]"
echo "Run commands with: PYTHONPATH=src python -m decomp_clarifier.cli ..."
