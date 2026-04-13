# Plan: Multi-System Benchmark Comparison Table

## Context

The current evaluation pipeline compares checkpoints against three baselines
(raw_ghidra, naming_only, prompt_only_cleanup) but does not show the base
Qwen model before fine-tuning, the model that generated the training data, or
a strong API reference. The report format is also a flat bullet list rather
than a table, making side-by-side comparisons hard to read.

This plan adds five new systems to the benchmark and replaces the flat list
with a proper markdown/HTML table showing all systems as columns.

## Target comparison table

| Metric | raw_ghidra | naming_only | base_qwen | sft | grpo | prompt_only_cleanup | generation_model | strong_model |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| json_valid_rate | … | … | … | … | … | … | … | … |
| readability_score | … | … | … | … | … | … | … | … |
| naming_score | … | … | … | … | … | … | … | … |
| compile_success_rate | … | … | … | … | … | … | … | … |
| behavior_success_rate | … | … | … | … | … | … | … | … |

Column ordering is narrative: floor baseline → heuristic → our model
progression (pre-ft → SFT → GRPO) → API comparators → upper bound.

---

## Files to change

| File | Change |
|------|--------|
| `src/decomp_clarifier/cli.py` | Extend `run-baselines` with three new flags; add Clang check at startup |
| `src/decomp_clarifier/evaluation/report_builder.py` | Add `render_comparison_table()`; update `render_markdown` and `render_html` |
| `src/decomp_clarifier/evaluation/checkpoint_eval.py` | Read `json_valid` from JSONL; gate behavior/naming on json_valid; add readability_improvement; replace flat sections with table |

---

## Change 1 — Extend `run-baselines` CLI command

**File:** `src/decomp_clarifier/cli.py` (lines 384–433)

Add three new `typer.Option` parameters to `run_baselines`:

```python
generation_model_id: str = typer.Option("openai/gpt-5.4-mini")
strong_model_id: str = typer.Option("openai/gpt-5.4-xhigh")
base_model_id: str | None = typer.Option(None)
```

### API model baselines (generation_model, strong_model)

For each of `generation_model_id` and `strong_model_id`, create a
`PromptOnlyCleanupBaseline` instance with that model ID (same pattern as the
existing `prompt_only_cleanup` baseline, reusing `PromptOnlyCleanupBaseline`
from `src/decomp_clarifier/baselines/simple_llm_cleanup.py`).

Run each over all samples and write records with system names
`"generation_model"` and `"strong_model"` respectively.

### Base model baseline (base_qwen)

Only runs when `--base-model-id` is passed (e.g. `Qwen/Qwen3.5-2B`).

Construct a minimal `TrainingConfig` with `model.base_model_id` set to the
provided value and no `sft_checkpoint_dir`. Instantiate `CheckpointPredictor`
from `src/decomp_clarifier/inference/checkpoint_predictor.py` using
`format_rl_prompt` as the prompt formatter (same as `eval-grpo-checkpoint`).
Use `system = "base_qwen"`.

Gate with a `try/import unsloth` block; if unsloth is not available, log a
warning and skip rather than crashing the whole command.

### Serialization — include json_valid and raw_text

The current writer serializes `{sample_id, system, output}`. Extend each
written record to also include `json_valid` and `raw_text` (both already
present on `PredictionRecord` objects returned by `CheckpointPredictor`).
For `PromptOnlyCleanupBaseline` predictions (which return
`ClarifiedFunctionOutput` directly), set `json_valid=True` and
`raw_text=None` when writing.

---

## Change 2 — `report_builder.py`: add comparison table renderer

**File:** `src/decomp_clarifier/evaluation/report_builder.py`

### New function: `render_comparison_table`

```python
_COLUMN_ORDER = [
    "raw_ghidra", "naming_only", "base_qwen",
    "sft_checkpoint", "grpo_checkpoint",
    "prompt_only_cleanup", "generation_model", "strong_model",
]

def render_comparison_table(systems: dict[str, dict[str, float]]) -> str:
    ...
```

- Accepts `{system_name: {metric_name: value}}`.
- Columns follow `_COLUMN_ORDER`; any unrecognised systems are appended
  alphabetically after the known ones.
- Rows are the union of all metric keys, in this fixed order:
  `json_valid_rate`, `field_complete_rate`, `readability_score`,
  `naming_score`, `compile_success_rate`, `behavior_success_rate`.
- Values formatted to three decimal places. Missing values shown as `—`.
- Returns a GitHub-flavoured markdown table string.

### Update `render_markdown`

Replace the current flat `- metric: value` list with a call to
`render_comparison_table({run_id: report.metrics})` when there is only one
system, so the format stays consistent with the multi-system view.

### Update `render_html`

Replace the two-column `<table>` with a proper multi-column HTML table using
the same column/row ordering as `render_comparison_table`.

---

## Change 3 — `checkpoint_eval.py`: use table in comparison report

**File:** `src/decomp_clarifier/evaluation/checkpoint_eval.py`

### `load_baseline_reports` (line 115)

When building each `PredictionRecord`, read `json_valid` from the JSONL
payload with a default of `True` for backward compatibility with old files:

```python
json_valid=payload.get("json_valid", True),
raw_text=payload.get("raw_text"),
```

### `render_comparison_markdown` (line 307)

Replace the current structure (checkpoint metrics as flat list + separate
`### system` sections) with:

1. Keep the header block (checkpoint path, split, sample count).
2. Build a unified dict:
   `all_systems = {stage + "_checkpoint": report_metrics, **baseline_metrics}`
3. Call `render_comparison_table(all_systems)` from `report_builder.py` for
   the main table.
4. Keep the existing "Checkpoint Metrics" flat list as a secondary detail
   section below the table (useful for precise values when the table is wide).

---

## Change 4 — Fix false eval signals in `checkpoint_eval.py`

**File:** `src/decomp_clarifier/evaluation/checkpoint_eval.py`

### 4a — Gate `behavior_success_rate` and `naming_score` on `json_valid`

When aggregating per-sample metrics, only include `behavior_success_rate` and
`naming_score` for samples where `json_valid=True`. For invalid-JSON samples,
treat both as `0.0` (not as missing — they failed).

Current bug: token Jaccard fires on the raw model text even when JSON parsing
fails, inflating `behavior_success_rate`. At SFT eval this produced
`behavior_success_rate: 0.771` alongside `json_valid_rate: 0.248` — the high
behavior score was entirely an artifact of identifier overlap in raw text.

### 4b — Report `readability_improvement` as a metric

After computing per-system `readability_score`, compute and report
`readability_improvement = readability_score - baseline_readability_score`
where the baseline is the `raw_ghidra` system's score. Add this field to the
metrics dict and include it as a row in `render_comparison_table` (raw_ghidra
column will always show `0.000`).

### 4c — Surface Clang unavailability at eval startup

At the start of `eval-grpo-checkpoint` (and `run-baselines`), check whether
`clang` is available on PATH. If not, log a prominent warning:

```
WARNING: clang not found on PATH — compile_success_rate will be 0.0 for all
samples. Install clang or add it to PATH to get real compile metrics.
```

Do not abort — just warn so the user knows a `0.0` compile rate is an
environment issue, not a model failure.

---

## Backward compatibility

- Old baseline JSONL files without `json_valid`/`raw_text` fields load fine
  (defaults applied in `load_baseline_reports`).
- The new `generation_model` and `strong_model` columns only appear if the
  most recent baseline run included them; the table renderer handles missing
  columns gracefully (shows `—`).
- Existing callers of `build_report` and `write_report` are unaffected.

---

## Verification

1. **Unit test — table renderer** (`tests/unit/test_report_builder.py`): call
   `render_comparison_table` with a fixture dict of two systems and assert the
   header row contains the right system names and a data row contains the right
   formatted value.

2. **Unit test — load_baseline_reports backward compat**: write a small
   baseline JSONL without `json_valid`, call `load_baseline_reports`, assert
   the returned evaluations have `json_valid=True`.

3. **Smoke test**: run `decomp-clarifier run-baselines
   --generation-model-id openai/gpt-5.4-mini
   --strong-model-id openai/gpt-5.4-xhigh`
   and confirm the output JSONL contains records with `system` values
   `generation_model` and `strong_model`.

4. **End-to-end**: after running `eval-grpo-checkpoint`, open
   `artifacts/runs/eval-grpo-checkpoint-*/comparison.md` and verify it renders
   a markdown table with all expected column headers.

5. **Unit test — behavior gating**: build a fixture with two samples, one
   `json_valid=True` and one `json_valid=False` (both with high token Jaccard);
   assert `behavior_success_rate` equals `1.0` not `0.5` after aggregation
   (invalid sample contributes `0.0`, not its raw Jaccard score).

6. **Unit test — readability_improvement**: assert the metric equals
   `checkpoint_readability - raw_ghidra_readability` and the `raw_ghidra`
   column shows `0.000`.

7. **Smoke test — Clang warning**: temporarily rename `clang` on PATH (or run
   in an env without it) and confirm the warning is printed without aborting.
