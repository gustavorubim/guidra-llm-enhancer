## Manual Pipeline Run

This runbook is for a full manual pass through the current repository pipeline on Windows.

It covers:

- environment setup
- each CLI step in order
- what to verify after each step
- how much data to generate for a meaningful pilot
- current limitations you need to account for manually

## Scope

What the repo can validate end to end today:

- project generation
- compile/test validation
- Ghidra export
- function-level dataset build
- baseline prediction generation
- baseline evaluation/report generation
- SFT training
- SFT checkpoint evaluation
- GRPO training
- GRPO checkpoint evaluation
- training telemetry artifacts

What is not fully wired today:

- hardware overlay configs under `configs/training/windows_cuda_*.yaml` are not auto-merged by the CLI
- there is no single command that directly compares `SFT` vs `GRPO` in one combined report
- checkpoint evaluation requires a finished checkpoint directory with saved adapter/model files

Because of that, the practical meaning of "full pipeline" right now is:

1. build the dataset and baseline report
2. run SFT
3. evaluate the SFT checkpoint
4. run GRPO
5. evaluate the GRPO checkpoint
6. compare the checkpoint reports and manual inspection samples

## Recommended Run Sizes

The generator enforces at least `3` C functions per project, and the current `sft` dataset profile creates one record per function per task key:

- `full_clarify`
- `cleanup`
- `rename`

That means the current dataset builder effectively produces about:

`aligned_functions * 3` SFT records

The split is project-level, not function-level, with:

- `80%` train
- `10%` val
- `10%` test

So project count matters more than raw record count.

### Recommended counts

`10 projects`

- good for smoke-testing the mechanics
- not enough for a meaningful quality signal
- only about `1` val project and `1` test project

`30 projects`

- bare minimum pilot
- enough to catch obvious regressions
- about `3` val projects and `3` test projects
- reasonable if you want the cheapest first full pass

`50 projects`

- recommended first "real" pilot
- about `5` val projects and `5` test projects
- usually enough to get a more stable read on whether the pipeline is behaving

`75-100 projects`

- better for a stronger signal
- recommended if you want to judge whether SFT/GRPO are worth iterating further

### Practical rule

Use:

- `10` for smoke
- `50` for a first meaningful run
- `100` if you want a more credible pilot dataset

## Environment Setup

Training is supported only on native Windows with NVIDIA CUDA.

### 1. Create the environment

```powershell
.\scripts\bootstrap.ps1 -Training
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = (Resolve-Path .\src).Path
```

If the training venv ends up with `torch ... +cpu` instead of a CUDA build, repair it
before continuing:

```powershell
uv pip install --python .\.venv\Scripts\python.exe --reinstall torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
```

### 2. Create `.env`

Start from `.env.example`:

```powershell
Copy-Item .env.example .env
```

Set:

- `OPENROUTER_API_KEY`
- `DECOMP_CLARIFIER_COMPILER_EXECUTABLE` if `clang` is not on `PATH`
- `DECOMP_CLARIFIER_GHIDRA_DIR` or `DECOMP_CLARIFIER_GHIDRA_ANALYZE_HEADLESS` if Ghidra is not discoverable

### 3. Quick preflight

```powershell
python -m decomp_clarifier.cli doctor
.\scripts\verify_train_env.ps1
python -m decomp_clarifier.cli --help
clang --version
```

Optional Ghidra preflight:

```powershell
Test-Path $env:DECOMP_CLARIFIER_GHIDRA_ANALYZE_HEADLESS
```

If you use `DECOMP_CLARIFIER_GHIDRA_DIR`, confirm:

```powershell
Test-Path (Join-Path $env:DECOMP_CLARIFIER_GHIDRA_DIR 'support\analyzeHeadless.bat')
```

## Fresh Start

If you want a fresh run, clear runtime outputs before starting:

```powershell
Get-ChildItem data\cache,data\raw,data\interim,data\processed,artifacts\logs,artifacts\reports,artifacts\runs,artifacts\models,artifacts\checkpoints,ghidra\project -Force |
  Remove-Item -Recurse -Force
```

Then recreate the directories:

```powershell
New-Item -ItemType Directory -Force data\cache,data\raw,data\interim,data\processed,artifacts\logs,artifacts\reports,artifacts\runs,artifacts\models,artifacts\checkpoints,ghidra\project | Out-Null
```

## Manual Run Order

For a first real pilot, use `50` projects.

### 1. Generate Projects

```powershell
python -m decomp_clarifier.cli generate-projects --count 50
```

What this does:

- calls OpenRouter
- validates each generated project structurally
- compiles/tests it immediately
- attempts one structured repair pass on compile/test failures
- quarantines invalid projects
- retries until it reaches the requested valid count or exhausts attempts

Verify:

```powershell
$latest = Get-ChildItem artifacts\runs\generate-* | Sort-Object LastWriteTime | Select-Object -Last 1
Get-Content (Join-Path $latest.FullName 'metrics.json')
(Get-ChildItem data\raw\generated_projects -Directory).Count
(Get-ChildItem data\raw\manifests -File).Count
```

Success criteria:

- `generated_count` equals your requested count
- `repaired_count` is visible when the repair pass salvages a project
- project directory count matches the requested count
- each project has a `project_manifest.json`

Watch for:

- high `quarantined_count`
- extremely high `repaired_count`, which usually means the base generation prompt/model is drifting
- generation failing to reach the target count within `count * 5` attempts

### 2. Recompile All Saved Projects

```powershell
python -m decomp_clarifier.cli compile-projects
```

Verify:

```powershell
$latest = Get-ChildItem artifacts\runs\compile-* | Sort-Object LastWriteTime | Select-Object -Last 1
Get-Content (Join-Path $latest.FullName 'metrics.json')
(Get-ChildItem data\raw\binaries -Directory).Count
```

Spot-check a few compile manifests:

```powershell
Get-ChildItem data\raw\binaries -Recurse -Filter compile_manifest.json | Select-Object -First 3 | ForEach-Object {
  Get-Content $_.FullName | Select-Object -First 40
}
```

Success criteria:

- `compiled` equals the number of projects
- each binary directory contains `compile_manifest.json`
- `binaries` is non-empty in each compile manifest
- test results, if present, are passing

### 3. Export Ghidra Artifacts

```powershell
python -m decomp_clarifier.cli export-ghidra
```

Verify:

```powershell
$latest = Get-ChildItem artifacts\runs\ghidra-* | Sort-Object LastWriteTime | Select-Object -Last 1
Get-Content (Join-Path $latest.FullName 'metrics.json')
(Get-ChildItem data\raw\ghidra_exports -Directory).Count
```

Required files per export directory:

- `project_manifest.json`
- `functions.jsonl`
- `ghidra_headless.log`
- `source_compile_manifest.json`

Quick check:

```powershell
Get-ChildItem data\raw\ghidra_exports -Directory | Select-Object -First 3 | ForEach-Object {
  $_.FullName
  Get-ChildItem $_.FullName
}
```

Success criteria:

- exported directory count matches compiled project count
- every export dir contains `project_manifest.json` and `functions.jsonl`
- `functions.jsonl` is non-empty for most projects

### 4. Build Dataset

```powershell
python -m decomp_clarifier.cli build-dataset
```

Outputs:

- `data/processed/sft/function_dataset.jsonl`
- `data/processed/sft/dataset_manifest.json`
- `data/processed/sft/sft_records.jsonl`
- `data/processed/rl/rl_records.jsonl`

Verify:

```powershell
Get-Content data\processed\sft\dataset_manifest.json
(Get-Content data\processed\sft\function_dataset.jsonl).Count
(Get-Content data\processed\sft\sft_records.jsonl).Count
(Get-Content data\processed\rl\rl_records.jsonl).Count
```

Success criteria:

- all files above exist
- `function_dataset.jsonl` line count is greater than `0`
- `sft_records.jsonl` and `rl_records.jsonl` line counts are greater than `0`
- `sft_records.jsonl` count should roughly match:
  `aligned_functions * 3`

Important note:

The numeric values in `configs/dataset/sft.yaml` task weights are not currently used as sampling weights. The current implementation uses the task keys only, so each function becomes one record for each of:

- `full_clarify`
- `cleanup`
- `rename`

### 5. Run Baselines

```powershell
python -m decomp_clarifier.cli run-baselines --base-model-id Qwen/Qwen3.5-2B --generation-model-id openai/gpt-5.4-mini --strong-model-id openai/gpt-5.4-xhigh --remote-workers 8
```

Verify:

```powershell
$latest = Get-ChildItem artifacts\runs\baseline-* | Sort-Object LastWriteTime | Select-Object -Last 1
$path = Join-Path $latest.FullName 'baseline_predictions.jsonl'
$path
(Get-Content $path).Count
```

Success criteria:

- `baseline_predictions.jsonl` exists
- it contains `active_baseline_system_count * function_dataset_rows` lines

The three baseline systems are:

- `raw_ghidra`
- `naming_only`
- `prompt_only_cleanup`

Optional benchmark systems:

- `generation_model`
- `strong_model`
- `base_qwen_openrouter`
- `base_qwen` when `--base-model-local-id` is set

If you want those extra systems in the baseline run:

```powershell
python -m decomp_clarifier.cli run-baselines `
  --generation-model-id openai/gpt-5.4-mini `
  --strong-model-id openai/gpt-5.4-xhigh `
  --base-model-id Qwen/Qwen3.5-2B
```

To also run the local Windows CUDA base model comparison:

```powershell
python -m decomp_clarifier.cli run-baselines `
  --generation-model-id openai/gpt-5.4-mini `
  --strong-model-id openai/gpt-5.4-xhigh `
  --base-model-id Qwen/Qwen3.5-2B `
  --base-model-local-id Qwen/Qwen3.5-2B
```

Notes:

- `generation_model` and `strong_model` require `OPENROUTER_API_KEY`
- `base_qwen_openrouter` uses the RL prompt over OpenRouter and defaults to the same model id passed to `--base-model-id`; use `--base-model-openrouter-id` to override it
- `base_qwen` is local-only and runs only when `--base-model-local-id` is provided
- baseline prediction rows now include `json_valid` and `raw_text`

### 6. Evaluate Baselines

```powershell
python -m decomp_clarifier.cli eval
python -m decomp_clarifier.cli report
```

Verify:

```powershell
Get-ChildItem artifacts\reports | Sort-Object LastWriteTime | Select-Object -Last 5
```

Success criteria:

- a new `.md`, `.html`, and `.json` report exists under `artifacts/reports`
- the `.md` report lists aggregate metrics and sample entries

Important note:

This `eval` path scores baseline predictions only. Trained checkpoints use
`eval-sft-checkpoint` and `eval-grpo-checkpoint`.

### 7. Run SFT

```powershell
python -m decomp_clarifier.cli train-sft
```

Expected output path:

- `artifacts/runs/train-sft-<timestamp>/model/sft_training_manifest.json`

Verify:

```powershell
$latest = Get-ChildItem artifacts\runs\train-sft-* | Sort-Object LastWriteTime | Select-Object -Last 1
$model = Join-Path $latest.FullName 'model'
Get-ChildItem $model -Recurse
Get-Content (Join-Path $model 'sft_training_manifest.json')
Get-Content (Join-Path $model 'logs\\sft_summary.json')
```

Training telemetry to check:

- `logs/sft_metrics.jsonl`
- `logs/sft_metrics.csv`
- `plots/sft_loss.png`
- `tensorboard/`

TensorBoard:

```powershell
tensorboard --logdir (Join-Path $model 'tensorboard')
```

Success criteria:

- the manifest exists
- the model directory contains saved trainer output
- `sft_metrics.jsonl` and `sft_metrics.csv` exist and are non-empty
- `sft_loss.png` exists
- loss does not immediately diverge to `nan`

### 7.5 Evaluate The SFT Checkpoint

After SFT finishes, evaluate the checkpoint on the held-out split:

```powershell
python -m decomp_clarifier.cli eval-sft-checkpoint --split val
```

PowerShell wrapper:

```powershell
.\scripts\eval_sft_checkpoint.ps1 -Split val
```

If you want to evaluate a specific finished checkpoint instead of the latest one:

```powershell
python -m decomp_clarifier.cli eval-sft-checkpoint `
  --checkpoint-dir artifacts/runs/train-sft-YYYYMMDD-HHMMSS/model `
  --split val
```

Outputs under `artifacts/runs/eval-sft-checkpoint-<timestamp>/`:

- `checkpoint_eval_manifest.json`
- `predictions.jsonl`
- `sample_evaluations.jsonl`
- `reports/`
- `comparison.md`
- `inspection_samples.md`
- `inspection_samples.jsonl`

Manual inspection file:

- `inspection_samples.md` shows:
  - original source
  - Ghidra decompiled code
  - reconstructed checkpoint output
  - verifier metrics for each sample

Recommended manual check:

```powershell
$latest = Get-ChildItem artifacts\runs\eval-sft-checkpoint-* | Sort-Object LastWriteTime | Select-Object -Last 1
Get-Content (Join-Path $latest.FullName 'checkpoint_eval_manifest.json')
Get-Content (Join-Path $latest.FullName 'comparison.md')
Get-Content (Join-Path $latest.FullName 'inspection_samples.md') | Select-Object -First 120
```

Success criteria:

- `predictions.jsonl` and `sample_evaluations.jsonl` exist and are non-empty
- `comparison.md` renders a metric table with the checkpoint column and any available baseline columns
- `inspection_samples.md` contains readable examples with source/decompiled/reconstructed sections
- the checkpoint is at least competitive with `prompt_only_cleanup` on the held-out split

### 8. Confirm GRPO Base Checkpoint

`train-grpo` now defaults to the latest completed SFT checkpoint when the training profile leaves `model.base_model_id` unset.

Before running GRPO, verify what it will use:

```powershell
Get-Content configs\training\grpo_qwen35_2b.yaml
```

If `model.base_model_id` is blank in that profile, the CLI will auto-resolve the latest finished `train-sft-*` run.

If you want to override that behavior, set `model.base_model_id` explicitly in a custom profile:

```yaml
model:
  base_model_id: artifacts/runs/train-sft-YYYYMMDD-HHMMSS/model
  loader_variant: unsloth
```

Recommended approach:

- copy `configs/training/grpo_qwen35_2b.yaml` to a new file
- change only `model.base_model_id`
- run `train-grpo --training-profile <your_new_profile_name>`

Important note:

The CLI loads exactly one training profile file. If you need different memory settings, create a custom profile that bakes those values in.

### 9. Run GRPO

```powershell
python -m decomp_clarifier.cli train-grpo
python -m decomp_clarifier.cli eval-grpo-checkpoint --split val

```

Or, if you made a custom profile:

```powershell
python -m decomp_clarifier.cli train-grpo --training-profile grpo_qwen35_2b_from_sft
```

Expected output path:

- `artifacts/runs/train-grpo-<timestamp>/model/grpo_training_manifest.json`

Verify:

```powershell
$latest = Get-ChildItem artifacts\runs\train-grpo-* | Sort-Object LastWriteTime | Select-Object -Last 1
$model = Join-Path $latest.FullName 'model'
Get-ChildItem $model -Recurse
Get-Content (Join-Path $model 'grpo_training_manifest.json')
Get-Content (Join-Path $model 'logs\\grpo_summary.json')
```

Training telemetry to check:

- `logs/grpo_metrics.jsonl`
- `logs/grpo_metrics.csv`
- `plots/grpo_reward.png`
- `tensorboard/`

TensorBoard:

```powershell
tensorboard --logdir (Join-Path $model 'tensorboard')
```

Success criteria:

- the manifest exists
- reward logs exist
- `grpo_reward.png` exists
- `reward_mean` is being recorded in `grpo_metrics.jsonl`
- rewards are not flat zero for the entire run

Important note:

- `rl_records.jsonl` now uses a compact GRPO-specific prompt rather than the full SFT prompt. This keeps most rollout prompts under the `896` token cap in the 12 GB profile.
- The default GRPO reward stack now prefers execution-backed behavior checks when `tests_ref` resolves to a generated project manifest, falls back to the similarity proxy otherwise, and only unlocks cleanup or readability bonuses after compile and behavior pass.

### 10. Run The Full Model Matrix

If you want to run the canonical Qwen + Gemma matrix end to end, use:

```powershell
.\scripts\run_training_matrix.ps1
```

This currently runs the active focused path only:

- `sft_qwen35_2b`
- `grpo_qwen35_2b`
- the matching SFT and GRPO checkpoint eval commands

It also refreshes:

- `artifacts/reports/model_matrix_summary.md`
- `artifacts/reports/model_matrix_summary.json`
- `artifacts/reports/target_comparison_table.md`
- `artifacts/reports/target_comparison_table.json`

### 9.5 Evaluate The GRPO Checkpoint

After GRPO finishes, evaluate the checkpoint on the held-out split:

```powershell
python -m decomp_clarifier.cli eval-grpo-checkpoint --split val
```

That command now defaults to `384` decode tokens, which matches the GRPO rollout budget in the 12 GB profile.

PowerShell wrapper:

```powershell
.\scripts\eval_grpo_checkpoint.ps1 -Split val
```

If you want to evaluate a specific finished checkpoint instead of the latest one:

```powershell
python -m decomp_clarifier.cli eval-grpo-checkpoint `
  --checkpoint-dir artifacts/runs/train-grpo-YYYYMMDD-HHMMSS/model `
  --split val
```

Outputs under `artifacts/runs/eval-grpo-checkpoint-<timestamp>/` mirror the SFT evaluation artifacts:

- `checkpoint_eval_manifest.json`
- `predictions.jsonl`
- `sample_evaluations.jsonl`
- `reports/`
- `comparison.md`
- `inspection_samples.md`
- `inspection_samples.jsonl`

Recommended manual check:

```powershell
$latest = Get-ChildItem artifacts\runs\eval-grpo-checkpoint-* | Sort-Object LastWriteTime | Select-Object -Last 1
Get-Content (Join-Path $latest.FullName 'checkpoint_eval_manifest.json')
Get-Content (Join-Path $latest.FullName 'comparison.md')
Get-Content (Join-Path $latest.FullName 'inspection_samples.md') | Select-Object -First 120
```

Success criteria:

- the GRPO eval artifacts are present and non-empty
- `comparison.md` shows a table with no obvious collapse versus the baseline reference
- `inspection_samples.md` contains both good and bad examples you can inspect manually
- GRPO is not materially worse than SFT on compile or behavior-oriented verifier fields

## What "Success" Looks Like

### Pipeline health success

You can consider the pipeline healthy if all of the following are true:

- generation reaches the requested valid project count
- compile step succeeds for all generated projects
- Ghidra export exists for all compiled binaries
- dataset files are non-empty
- baseline report is generated
- SFT finishes and emits telemetry
- SFT checkpoint evaluation runs and writes reports plus inspection samples
- GRPO finishes and emits telemetry
- GRPO checkpoint evaluation runs and writes reports plus inspection samples

### First meaningful experiment success

For a first meaningful pilot, target:

- `50` projects
- at least `~5` test projects after split
- at least `~100` held-out SFT records as a rough goal

Why that threshold:

- with only `10` projects, the test split is too small to trust
- with `30` projects, you can see rough trends
- with `50+` projects, the held-out slice is usually large enough to spot obvious wins or failures

### Training success

For the current codebase, training success should be judged primarily by:

- non-empty saved checkpoints
- healthy telemetry files
- stable or improving SFT loss
- non-zero and non-degenerate GRPO reward logs
- no immediate collapse into invalid JSON or empty `cleaned_c`

## What You Still Need To Verify Manually

The repo now gives you checkpoint evaluation commands, but you still need to
manually inspect the examples and compare `SFT` vs `GRPO`.

Recommended manual follow-up after both evaluations:

1. keep the baseline report as your reference point
2. open the latest `comparison.md` from the SFT eval run
3. open the latest `comparison.md` from the GRPO eval run
4. compare:
   - json validity
   - compile success
   - behavior success
   - readability score
   - readability improvement
   - naming score
5. read both `inspection_samples.md` files and compare the reconstructed code quality
6. if the validation split looks good, rerun both checkpoint eval commands on `--split test` once

Important note:

- there is still no one-shot CLI that produces a single combined `baseline vs SFT vs GRPO`
  report in one run
- if you already have the checkpoint eval artifacts, `python scripts/build_target_comparison_table.py`
  will merge the stored baseline metrics plus the SFT and GRPO manifests into one table
- the intended manual workflow is:
  - baseline report from `eval`
  - SFT checkpoint report from `eval-sft-checkpoint`
  - GRPO checkpoint report from `eval-grpo-checkpoint`
- checkpoint and baseline comparison tables now zero out behavior and naming credit for invalid-JSON outputs

## Short Command List

```powershell
.\scripts\bootstrap.ps1 -Training
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = (Resolve-Path .\src).Path

python -m decomp_clarifier.cli doctor
.\scripts\verify_train_env.ps1

python -m decomp_clarifier.cli generate-projects --count 50
python -m decomp_clarifier.cli compile-projects
python -m decomp_clarifier.cli export-ghidra
python -m decomp_clarifier.cli build-dataset
python -m decomp_clarifier.cli run-baselines

python -m decomp_clarifier.cli run-baselines --base-model-id Qwen/Qwen3.5-2B --generation-model-id openai/gpt-5.4-mini --strong-model-id openai/gpt-5.4-xhigh

python -m decomp_clarifier.cli eval
python -m decomp_clarifier.cli report
python -m decomp_clarifier.cli train-sft
python -m decomp_clarifier.cli eval-sft-checkpoint --split val
python -m decomp_clarifier.cli train-grpo --training-profile grpo_qwen35_2b_from_sft
python -m decomp_clarifier.cli eval-grpo-checkpoint --split val
```

## Focused Qwen Command Reference

Use the block below for the current active workflow: `Qwen3.5-2B` only.

`Qwen` GRPO should be launched from an explicit validated SFT checkpoint via `--base-model-id`.

Replace the `<path>` placeholders in the last line with the actual `checkpoint_eval_manifest.json` paths you want to include.

```powershell
python -m decomp_clarifier.cli build-dataset --dataset-profile sft --app-profile default

python -m decomp_clarifier.cli train-sft --training-profile sft_qwen35_2b --app-profile default
python -m decomp_clarifier.cli eval-sft-checkpoint --training-profile sft_qwen35_2b --app-profile default --split val --inspection-sample-count 8 --max-new-tokens 384 --temperature 0.0
python -m decomp_clarifier.cli train-grpo --training-profile grpo_qwen35_2b --base-model-id <path-to-validated-qwen-sft-checkpoint> --app-profile default
python -m decomp_clarifier.cli eval-grpo-checkpoint --checkpoint-dir <path-to-qwen-grpo-model-dir> --training-profile grpo_qwen35_2b --app-profile default --split val --inspection-sample-count 8 --max-new-tokens 384 --temperature 0.0
python .\scripts\build_model_matrix_summary.py --app-profile default --eval-manifest "sft_qwen35_2b=<path-to-sft-qwen35-2b-checkpoint_eval_manifest.json>" --eval-manifest "grpo_qwen35_2b=<path-to-grpo-qwen35-2b-checkpoint_eval_manifest.json>"

```
