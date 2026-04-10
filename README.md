# decomp-clarifier

`decomp-clarifier` is a local research prototype for binary-grounded decompiler clarification. It turns synthetic C projects into binaries, exports Ghidra artifacts, assembles function-level datasets, runs baseline cleanup flows, and provides a guarded Windows CUDA training path for Qwen3.5 + Unsloth SFT/GRPO experiments.

## Status

This repository implements the scaffold and cross-platform core pipeline described in [SPEC.md](SPEC.md):

- config-driven Typer CLI
- OpenRouter-backed synthetic project generation with caching
- host-native Clang compilation and test execution
- Ghidra headless orchestration and export parsing
- function-level dataset assembly, packing, baselines, evaluation, and report generation
- Windows-only guarded training entry points and reward/verifier utilities

## Quick Start

### macOS / Linux

```bash
./scripts/bootstrap.sh
source .venv/bin/activate
PYTHONPATH=src python -m decomp_clarifier.cli --help
```

### Windows / PowerShell

```powershell
./scripts/bootstrap.ps1
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m decomp_clarifier.cli --help
```

## Common Commands

```bash
PYTHONPATH=src python -m decomp_clarifier.cli generate-projects --count 5
PYTHONPATH=src python -m decomp_clarifier.cli compile-projects
PYTHONPATH=src python -m decomp_clarifier.cli export-ghidra
PYTHONPATH=src python -m decomp_clarifier.cli build-dataset
PYTHONPATH=src python -m decomp_clarifier.cli run-baselines
PYTHONPATH=src python -m decomp_clarifier.cli eval
PYTHONPATH=src python -m decomp_clarifier.cli report
```

Training commands are intentionally guarded and will fail fast on non-Windows or non-CUDA environments:

```bash
PYTHONPATH=src python -m decomp_clarifier.cli train-sft
PYTHONPATH=src python -m decomp_clarifier.cli train-grpo
```

## Ghidra

The included [run_headless_analysis.command](run_headless_analysis.command) reflects a local macOS setup and was used to shape the default headless adapter. Override the install path with either:

- `DECOMP_CLARIFIER_GHIDRA_DIR`
- `DECOMP_CLARIFIER_GHIDRA_ANALYZE_HEADLESS`
- `configs/ghidra/default.yaml`

## Testing

```bash
pytest --cov=src/decomp_clarifier --cov-config=coverage.toml --cov-report=term-missing --cov-fail-under=90
```
