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
- GRPO training
- training telemetry artifacts

What is not fully wired today:

- there is no trained-model inference CLI yet
- there is no CLI path that evaluates an SFT or GRPO checkpoint against the baseline report
- `train-grpo` does not automatically consume the latest SFT checkpoint
- hardware overlay configs under `configs/training/windows_cuda_*.yaml` are not auto-merged by the CLI

Because of that, the practical meaning of "full pipeline" right now is:

1. build the dataset and baseline report
2. run SFT
3. run GRPO
4. verify the training artifacts and telemetry

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
- project directory count matches the requested count
- each project has a `project_manifest.json`

Watch for:

- high `quarantined_count`
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
python -m decomp_clarifier.cli run-baselines
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
- it contains `3 * function_dataset_rows` lines

The three baseline systems are:

- `raw_ghidra`
- `naming_only`
- `prompt_only_cleanup`

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

This evaluation path currently scores baseline predictions only. It does not yet score trained checkpoints.

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

### 8. Point GRPO at the SFT Checkpoint

This is a manual step.

`train-grpo` does not automatically use the latest SFT output.

Before GRPO, create a temporary training profile or edit `configs/training/grpo_qwen35_4b.yaml` so:

- `model.base_model_id` points at the SFT model directory

Example value:

```yaml
model:
  base_model_id: artifacts/runs/train-sft-YYYYMMDD-HHMMSS/model
  loader_variant: unsloth
```

Recommended approach:

- copy `configs/training/grpo_qwen35_4b.yaml` to a new file
- change only `model.base_model_id`
- run `train-grpo --training-profile <your_new_profile_name>`

Important note:

The CLI loads exactly one training profile file. The `windows_cuda_16gb.yaml` and related files are not auto-composed on top of `sft_qwen35_4b.yaml` or `grpo_qwen35_4b.yaml`. If you need lower-memory settings, create a custom profile that bakes those values in.

### 9. Run GRPO

```powershell
python -m decomp_clarifier.cli train-grpo
```

Or, if you made a custom profile:

```powershell
python -m decomp_clarifier.cli train-grpo --training-profile grpo_qwen35_4b_from_sft
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

## What "Success" Looks Like

### Pipeline health success

You can consider the pipeline healthy if all of the following are true:

- generation reaches the requested valid project count
- compile step succeeds for all generated projects
- Ghidra export exists for all compiled binaries
- dataset files are non-empty
- baseline report is generated
- SFT finishes and emits telemetry
- GRPO finishes and emits telemetry

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

Because there is no trained-model evaluation CLI yet, this repo does not currently give you an automatic "SFT beat baseline" or "GRPO beat SFT" report.

So after the first full manual run, the next reasonable manual follow-up is:

1. keep the baseline report as your reference point
2. inspect SFT and GRPO telemetry
3. run a small custom inference script or notebook against held-out samples
4. compare:
   - JSON validity
   - placeholder cleanup
   - rename quality
   - compile proxy success
   - readability

If you want that final comparison to be repeatable, the next engineering step should be a dedicated trained-checkpoint inference/eval CLI.

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
python -m decomp_clarifier.cli eval
python -m decomp_clarifier.cli report
python -m decomp_clarifier.cli train-sft
python -m decomp_clarifier.cli train-grpo --training-profile grpo_qwen35_4b_from_sft
```
