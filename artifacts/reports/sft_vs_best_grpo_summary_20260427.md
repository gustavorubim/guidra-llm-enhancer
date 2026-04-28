# SFT vs Best GRPO Performance Summary

- Generated: `2026-04-27T20:29:27-04:00`
- Split/sample count: `val` / `105`
- SFT eval: `eval-sft-checkpoint-20260425-021627`
- Best GRPO eval: `eval-grpo-checkpoint-20260425-024529`
- Same sample order: `true`

Interpretation: `Raw Ghidra` is the raw decompiler baseline. `Original Qwen` is the base Qwen model through OpenRouter before SFT/GRPO.

## Main Table

| Metric | Raw Ghidra | Original Qwen | Prompt-only cleanup | SFT checkpoint | Best GRPO checkpoint |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Composite score | 0.302 | 0.087 | 0.520 | 0.762 | 0.767 |
| JSON valid | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| Field complete | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| Readability | 0.534 | 0.534 | 0.728 | 0.876 | 0.875 |
| Readability gain | 0.000 | 0.000 | 0.194 | 0.341 | 0.341 |
| Naming | 0.143 | 0.000 | 0.410 | 0.914 | 0.914 |
| Compile success | 0.029 | 0.029 | 0.324 | 0.705 | 0.705 |
| Behavior success | 0.000 | 0.000 | 0.295 | 0.543 | 0.562 |

## Key Deltas

| Comparison | Composite | Behavior | Compile | Readability | Naming | JSON |
|:---|---:|---:|---:|---:|---:|---:|
| SFT - Raw Ghidra | 0.460 | 54.3 pp | 67.6 pp | 0.341 | 0.771 | 0.0 pp |
| Best GRPO - Raw Ghidra | 0.466 | 56.2 pp | 67.6 pp | 0.341 | 0.771 | 0.0 pp |
| Best GRPO - SFT | 0.006 | 1.9 pp | 0.0 pp | -0.000 | 0.000 | 0.0 pp |
| Current pinned champion - SFT | -0.025 | -3.8 pp | -5.7 pp | -0.001 | 0.004 | 0.0 pp |

## Notes

- Best GRPO by the campaign score is `0005-context_plus_constant_strict_json_v1` / `eval-grpo-checkpoint-20260425-024529`.
- The currently pinned champion config remains `0019-invalid_scope_guard_v1` / `eval-grpo-checkpoint-20260425-123742`; it was selected by the full-prompt target campaign, but it is not the top score on the 105-sample context-plus-strict eval.
- The latest 50-sample q200 scouts on April 26 were below both the latest SFT and the best 105-sample GRPO, so they are not promoted in this summary.

## Source Paths

- SFT checkpoint: `artifacts\runs\eval-sft-checkpoint-20260425-021627\checkpoint_eval_manifest.json`
- Best GRPO checkpoint: `artifacts\runs\eval-grpo-checkpoint-20260425-024529\checkpoint_eval_manifest.json`
- Current pinned champion: `artifacts\runs\eval-grpo-checkpoint-20260425-123742\checkpoint_eval_manifest.json`
