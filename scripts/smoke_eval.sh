#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
pytest tests/smoke -q
