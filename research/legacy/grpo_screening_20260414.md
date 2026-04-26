# GRPO Screening - 2026-04-14

## Current observed result

Held-out `val` metrics from the latest finished runs:

| Metric | SFT | GRPO | Delta |
|---|---:|---:|---:|
| json_valid_rate | 0.914 | 0.886 | -0.029 |
| readability_score | 0.832 | 0.851 | +0.020 |
| naming_score | 0.616 | 0.551 | -0.065 |
| compile_success_rate | 0.733 | 0.571 | -0.162 |
| behavior_success_rate | 0.914 | 0.867 | -0.048 |

Interpretation: the current GRPO stage improves surface readability a bit, but it gives back too much on syntax validity, compile success, behavior proxy, and naming.

## Hard findings from the current implementation

These are not speculative training hypotheses. They are concrete implementation issues or confounds.

### F1. SFT and GRPO training are both consuming all splits, not only `train`

- `build-dataset` packs every sample into `sft_records.jsonl` and `rl_records.jsonl` without filtering by `sample.split`.
- `train-sft` and `train-grpo` then train directly on those full packed files.

Evidence:

- `FunctionDatasetSample.split` is assigned in the dataset builder.
- `build-dataset` packs all `samples`: `src/decomp_clarifier/cli.py:643-653`
- `train-sft` reads `sft_records.jsonl`: `src/decomp_clarifier/cli.py:1015-1023`
- `train-grpo` reads `rl_records.jsonl`: `src/decomp_clarifier/cli.py:1045-1054`

Impact:

- current `val` numbers are not clean hold-out numbers
- SFT vs GRPO comparisons are still useful for debugging direction, but not for trustworthy generalization claims

### F2. The RL dataset config is not actually shaping the RL dataset

- `configs/dataset/rl.yaml` declares a curated RL set with `include_task_types` and `prompt_limit`.
- The dataset builder currently uses only `task_mix`; `include_task_types` and `prompt_limit` are unused.

Evidence:

- RL dataset config: `configs/dataset/rl.yaml`
- Builder only uses `task_mix`: `src/decomp_clarifier/dataset/builders.py:27-35`

Impact:

- the GRPO dataset is not smaller or more curated than SFT, contrary to `SPEC.md`
- rename-only samples are still in the RL set
- the run budget is spread across more prompt types than intended

### F3. SFT and GRPO are not being evaluated under the same prompt context

- SFT evaluation uses `format_prompt`
- GRPO evaluation uses `format_rl_prompt`

Evidence:

- stage-specific prompt selection: `src/decomp_clarifier/evaluation/checkpoint_eval.py:420`
- full prompt includes assembly, strings, callers, callees: `src/decomp_clarifier/dataset/prompt_formatter.py:8-29`
- RL prompt removes assembly, strings, callers, and uses a much smaller context: `src/decomp_clarifier/dataset/prompt_formatter.py:33-49`

Impact:

- part of the observed SFT -> GRPO drop may be prompt-context loss rather than policy degradation
- the current comparison is not an apples-to-apples checkpoint comparison

### F4. The current GRPO run was extremely short relative to the dataset

- latest GRPO training finished at `epoch = 0.0351` and `step = 150`
- RL dataset size is `1068`
- reward telemetry contains only `38` reward-function rows, which means only a small fraction of prompt batches were actually rolled out

Evidence:

- run manifest: `artifacts/runs/train-grpo-20260414-154212/model/grpo_training_manifest.json:22-29`
- trainer state: `artifacts/runs/train-grpo-20260414-154212/model/checkpoint-150/trainer_state.json:5-7`
- GRPO profile: `configs/training/grpo_qwen35_2b_12gb.yaml:11-15`

Impact:

- the run is heavily budget-constrained
- any reward or prompt improvement has very little time to move the policy

### F5. The hallucination penalty is mis-specified and penalizes gold outputs

- `hallucination_penalty()` scans the whole `cleaned_c` for `name(` patterns.
- `_allowed_callees()` includes target callees, but not the current function name.
- As a result, the function definition itself is treated like an unsupported call.

Evidence:

- penalty regex: `src/decomp_clarifier/training/grpo/rewards.py:107-115`
- allowed call set construction: `src/decomp_clarifier/dataset/packers.py:16-20`

Observed locally:

- gold target outputs still get a positive hallucination penalty on about `93%` of samples
- when the source function name is added to the allowed set, that false penalty drops to `0`

Impact:

- reward shaping is systematically pushing against correct outputs
- this is likely one of the biggest reasons GRPO struggles to hold onto compile-safe SFT behavior

### F6. The behavior reward is too weak to prevent compile regressions

- `behavior_success` is derived from token-overlap similarity, not execution.
- valid GRPO outputs frequently get `behavior_success = True` while still failing compile.

Evidence:

- behavior proxy implementation: `src/decomp_clarifier/evaluation/behavior_eval.py`
- reward composition applies a soft gate, not a hard reject: `src/decomp_clarifier/training/grpo/rewards.py:168-187`

Observed locally on the latest `val` eval:

- SFT valid outputs with `behavior_success=True` but `compile_success=False`: `19`
- GRPO valid outputs with `behavior_success=True` but `compile_success=False`: `31`

Impact:

- GRPO can still collect useful positive reward for cleaner-looking but uncompilable code
- this matches the observed readability up / compile down pattern

### F7. The `384` token completion cap is causing real JSON truncation on long samples

Observed locally on the GRPO eval predictions:

- `12` invalid JSON outputs total
- `9` of those are truncated JSON at `383-384` tokens
- truncated-invalid samples have much longer targets on average than valid samples

Impact:

- part of the JSON-validity drop is not policy collapse; it is a hard decode-budget issue
- long-function samples are at structural disadvantage during GRPO

### F8. One configured reward term is effectively dead

- `decompiler_type_penalty` is configured, but it is `0.0` throughout the training telemetry and on gold targets in current data.

Evidence:

- implementation: `src/decomp_clarifier/training/grpo/rewards.py:99-104`
- configured weight: `configs/training/grpo_qwen35_2b_12gb.yaml:26-35`

Impact:

- the effective reward surface is smaller than it looks on paper
- this is not the main failure, but it is noise in the config and diagnosis

## Hypotheses after accounting for the hard findings

### H1. The main problem is not that GRPO is "bad"; it is badly budgeted

Why it is plausible:

- only `3.5%` of an epoch was completed
- only `38` reward batches were logged
- learning rate is small for such a short run

Expected signature:

- longer GRPO runs should recover some of the compile/behavior drop even without major reward changes

### H2. Reward shaping is currently biased toward surface cleanup over compilable faithfulness

Why it is plausible:

- compile-false / behavior-true outputs can still receive non-trivial reward
- the behavior proxy is weak
- readability is the only headline metric that improved

Expected signature:

- stricter compile gating or stronger compile weighting should improve compile success and reduce readability-only regressions

### H3. The hallucination penalty bug is materially distorting the policy gradient

Why it is plausible:

- the current reward punishes even gold outputs
- that pressure is unrelated to actual hallucination behavior

Expected signature:

- fixing the penalty should improve compile and naming stability without needing more prompt context

### H4. The compact RL prompt removed information that SFT relied on

Why it is plausible:

- RL prompt drops assembly, strings, and caller context
- full-clarify tasks degrade more than cleanup tasks

Expected signature:

- restoring some of that context for RL should help full-clarify and rename more than cleanup

### H5. The current RL task mix is wrong for the available run budget

Why it is plausible:

- rename-only samples are included even though the RL config says otherwise
- the spec explicitly says RL should be smaller and more curated than SFT

Expected signature:

- training only on `full_clarify + cleanup` should improve signal density and reduce wasted rollout budget

### H6. The decode budget is too short for the longest functions

Why it is plausible:

- most invalid GRPO generations are literal truncation at the cap
- long-target samples are overrepresented among failures

Expected signature:

- a larger completion cap or a length-filtered curriculum should improve JSON validity and compile rate on the long tail

### H7. The learning rate is too conservative for the current step budget

Why it is plausible:

- `2e-6` may be fine for a long run, but not for `38` reward batches

Expected signature:

- a moderate LR increase should improve convergence only if the reward is first made trustworthy

## Study design

## Phase 0 - Fix the measurement confounds first

Do this before interpreting another SFT-vs-GRPO comparison:

1. Train SFT and GRPO on `train` only.
2. Make RL dataset curation real:
   - honor `include_task_types`
   - honor `prompt_limit`
3. Evaluate SFT and GRPO under the same prompt family.
   - recommended: report both `full_prompt` and `rl_prompt`
4. Keep the current metric table, but add a scalar study score:

```text
study_score =
  0.30 * behavior_success_rate +
  0.25 * compile_success_rate +
  0.20 * json_valid_rate +
  0.15 * readability_score +
  0.10 * naming_score
```

Hard guardrails:

- `compile_success_rate` must not fall by more than `0.01`
- `behavior_success_rate` must not fall by more than `0.01`
- `json_valid_rate` must not fall by more than `0.02`

Without Phase 0, the next runs will still be informative for local debugging, but they will not tell you cleanly whether GRPO is actually better.

## Phase 1 - Single-factor structural ablations

Run these one at a time from the same SFT checkpoint and the same fixed train/val split.

### Run A0 - Clean baseline rerun

Purpose:

- establish a trustworthy post-fix baseline

Changes:

- no intentional GRPO algorithm change
- only Phase 0 hygiene fixes

### Run A1 - Fix hallucination penalty only

Purpose:

- isolate the effect of removing a known bad reward signal

Changes:

- exclude the current function definition from hallucination counting
- or include `source_function_name` in the allowed set

### Run A2 - Curated RL dataset only

Purpose:

- test whether smaller, better-targeted RL prompts help more than raw dataset volume

Changes:

- `include_task_types = [full_clarify, cleanup]`
- optional `prompt_limit` around `400-700` prompts for the first scout

### Run A3 - Prompt context restoration only

Purpose:

- test whether GRPO needs more grounding than the current compact prompt provides

Changes:

- add back the smallest useful subset first:
  - assembly
  - strings
  - maybe callers

### Run A4 - Longer completion budget only

Purpose:

- measure how much of the JSON-validity drop is hard truncation

Changes:

- increase completion cap for RL rollout and GRPO eval
- if memory is tight, couple this with the curated RL set instead of changing multiple variables at once

### Run A5 - More training budget only

Purpose:

- test the undertraining hypothesis directly

Changes:

- increase effective GRPO budget substantially
- target at least `0.25-0.50` epoch equivalent before judging the method

## Phase 2 - Reward-shaping ablations

Only start Phase 2 after A0-A5 identifies whether the main bottleneck is data/prompt/budget.

### Run B1 - Stricter compile safety

Changes:

- stronger compile weight and/or harder compile gate
- do not let compile-false / behavior-true outputs receive mildly positive reward

Expected outcome:

- compile and behavior should rise together
- readability may dip slightly

### Run B2 - Lower reliance on the behavior proxy

Changes:

- reduce behavior weight until the proxy is improved
- rely more on compile + signature + formatting during early RL

Expected outcome:

- fewer compile regressions caused by lexical-overlap false positives

### Run B3 - Curriculum by task type

Changes:

- start with `cleanup` and `full_clarify`
- add `rename` only if it still improves the scalar score after the policy is stable

Expected outcome:

- less reward fragmentation early in training

## Phase 3 - Hyperparameter sweep on the winning structure

After one structural variant clearly wins, sweep small hyperparameter changes:

1. learning rate: current, `2x`, maybe `0.5x`
2. generations per prompt: current vs lower
3. warmup ratio: current vs smaller

Do not sweep these until the reward and dataset are trustworthy. Otherwise you will optimize around a broken signal.

## Proposed run order

If you want a practical first block, run them in this order:

1. A0 clean baseline rerun
2. A1 fix hallucination penalty
3. A2 curated RL dataset
4. A5 more training budget
5. A3 prompt context restoration
6. A4 longer completion budget
7. B1 stricter compile safety
8. combined winner from the best two single-factor runs

## Decision rules

### If A1 wins clearly

Next step:

- keep the hallucination fix
- then test A5 and B1

Interpretation:

- reward corruption was a first-order blocker

### If A5 wins clearly

Next step:

- keep the longer budget
- then test A1 and A2 on top

Interpretation:

- the current GRPO run was simply too short to judge

### If A3 wins clearly

Next step:

- keep the richer RL prompt
- then test B1

Interpretation:

- GRPO losses were prompt-context losses, not purely RL losses

### If A4 wins clearly

Next step:

- keep the longer completion budget or add a length curriculum

Interpretation:

- a large part of the regression was decode truncation

### If none of A1-A5 beats A0

Next step:

- freeze prompt and dataset
- revisit the reward stack and behavior proxy before more expensive sweeps

Interpretation:

- the current reward design is probably the dominant problem

## Recommended immediate next steps

1. Fix split hygiene and RL dataset curation before trusting another headline comparison.
2. Fix the hallucination penalty bug.
3. Rerun a clean baseline.
4. Increase effective GRPO budget enough that the method is being judged after meaningful exposure.

## Notes on the existing autoresearch loop

The current `research/program.md` is useful for iterative GRPO work, but this diagnosis found several first-order issues outside the currently editable surface:

- split hygiene in the CLI packing/training path
- evaluation prompt comparability
- dataset curation path

If you want the autonomous loop to produce trustworthy conclusions, its allowed surface should be widened to cover those measurement fixes first.
