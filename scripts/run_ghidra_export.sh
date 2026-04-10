#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
python -m decomp_clarifier.cli export-ghidra "$@"
