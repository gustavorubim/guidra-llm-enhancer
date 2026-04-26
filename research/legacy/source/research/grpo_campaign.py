from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from decomp_clarifier.c_source import extract_called_functions
from decomp_clarifier.dataset.prompt_formatter import (
    format_context_plus_prompt,
    format_context_plus_strict_prompt,
    format_prompt,
    format_rl_prompt,
    prompt_messages,
)
from decomp_clarifier.paths import ProjectPaths
from decomp_clarifier.research.autoresearch import (
    RewardTelemetrySnapshot,
    classify_invalid_prediction_rows,
    score_metrics,
)
from decomp_clarifier.schemas.dataset import FunctionDatasetSample, PackedRLRecord
from decomp_clarifier.settings import load_dotenv

DEFAULT_SEED_PROFILE = "configs/training/grpo_qwen35_2b_gdpo_300.yaml"
DEFAULT_SCOUT_SAMPLE_LIMIT = 50
DEFAULT_KEEP_IMPROVEMENT = 0.005
DEFAULT_CONFIRM_IMPROVEMENT = 0.01
DEFAULT_TARGET_IMPROVEMENT = 0.02
DEFAULT_EVAL_PROMPT_PROFILE = "full"
DEFAULT_EVAL_MAX_NEW_TOKENS = 1024
DEFAULT_SEARCH_SPACE = "default"
_SCORE_KEYS = (
    "json_valid_rate",
    "field_complete_rate",
    "readability_score",
    "compile_success_rate",
    "behavior_success_rate",
    "naming_score",
)


PROMPT_BUILDERS: dict[str, PromptBuilder] = {
    "compact": format_rl_prompt,
    "context_plus": format_context_plus_prompt,
    "context_plus_strict": format_context_plus_strict_prompt,
    "full": format_prompt,
}


class CampaignError(RuntimeError):
    """Raised when the GRPO campaign cannot continue safely."""


@dataclass(frozen=True)
class CampaignExperiment:
    experiment_id: str
    hypothesis: str
    short_description: str
    overrides: dict[str, Any]
    dataset_overrides: dict[str, Any] = field(default_factory=dict)


PromptBuilder = Callable[[FunctionDatasetSample], str]


def _long300_campaign_candidates() -> list[CampaignExperiment]:
    return [
        CampaignExperiment(
            experiment_id="long300_lr_low_v1",
            hypothesis=(
                "The 300-step control may still be over-updating late in training. Lowering the "
                "learning rate and shortening warmup could preserve the good curve more cleanly."
            ),
            short_description="300-step low lr",
            overrides={"training": {"learning_rate": 1.0e-6, "warmup_ratio": 0.05}},
        ),
        CampaignExperiment(
            experiment_id="long300_lr_high_v1",
            hypothesis=(
                "The current long-horizon run may still be too conservative. A modestly higher "
                "learning rate might convert the alive reward path into larger validation gains."
            ),
            short_description="300-step high lr",
            overrides={"training": {"learning_rate": 2.0e-6, "warmup_ratio": 0.05}},
        ),
        CampaignExperiment(
            experiment_id="long300_gradaccum2_v1",
            hypothesis=(
                "Stabler optimizer updates may matter more than more reward tweaks. Increasing "
                "gradient accumulation with a lower LR tests whether update noise is the limiter."
            ),
            short_description="300-step accum2",
            overrides={
                "training": {
                    "grad_accum_steps": 2,
                    "learning_rate": 1.0e-6,
                    "warmup_ratio": 0.05,
                }
            },
        ),
        CampaignExperiment(
            experiment_id="long300_g5_v1",
            hypothesis=(
                "Six rollouts per prompt was too much, but four may still leave too much variance. "
                "Five rollouts could improve group baselines without collapsing quality."
            ),
            short_description="300-step g5",
            overrides={"training": {"generations_per_prompt": 5}},
        ),
        CampaignExperiment(
            experiment_id="long300_g5_lowlr_v1",
            hypothesis=(
                "If extra rollouts help but make updates too sharp, combining five generations "
                "with a lower LR may reduce variance without overshooting."
            ),
            short_description="300-step g5 low lr",
            overrides={
                "training": {
                    "generations_per_prompt": 5,
                    "learning_rate": 1.0e-6,
                    "warmup_ratio": 0.05,
                }
            },
        ),
        CampaignExperiment(
            experiment_id="long300_completion352_v1",
            hypothesis=(
                "The best long run still leaves a small invalid JSON tail. A slight cut to 352 "
                "tokens may reduce clipping without the damage seen at 256."
            ),
            short_description="300-step completion 352",
            overrides={"training": {"max_completion_length": 352}},
        ),
        CampaignExperiment(
            experiment_id="long300_completion320_v1",
            hypothesis=(
                "A 320-token contract plus stronger invalid-output penalties may reduce malformed "
                "answers while staying much safer than the earlier 256-token regime."
            ),
            short_description="300-step completion 320",
            overrides={
                "training": {
                    "max_completion_length": 320,
                    "max_invalid_completion_ratio": 0.85,
                    "reward_weights": {
                        "invalid_json_penalty": 0.35,
                        "invalid_length_penalty": 1.0,
                        "truncation_penalty": 2.0,
                    },
                }
            },
        ),
        CampaignExperiment(
            experiment_id="long300_ratio140_v1",
            hypothesis=(
                "The model may need a tighter completion-ratio contract more than a shorter token "
                "budget. Reducing max_completion_ratio to 1.4 tests that directly."
            ),
            short_description="300-step ratio 1.4",
            overrides={"training": {"max_completion_ratio": 1.4}},
        ),
        CampaignExperiment(
            experiment_id="long300_ratio130_v1",
            hypothesis=(
                "A stronger ratio guard with an explicit invalid-output contract may suppress the "
                "remaining oversized bad-tail completions better than token clipping alone."
            ),
            short_description="300-step ratio 1.3",
            overrides={
                "training": {
                    "max_completion_ratio": 1.3,
                    "max_invalid_completion_ratio": 0.8,
                    "reward_weights": {
                        "invalid_json_penalty": 0.35,
                        "invalid_length_penalty": 1.0,
                        "truncation_penalty": 2.0,
                    },
                }
            },
        ),
        CampaignExperiment(
            experiment_id="long300_signature_up_v1",
            hypothesis=(
                "The long-horizon control already has a good verifier curve. A slightly stronger "
                "format and signature bias may improve the JSON-valid tail without shrinking "
                "budget."
            ),
            short_description="300-step signature up",
            overrides={
                "training": {
                    "reward_weights": {
                        "format": 1.1,
                        "signature": 1.75,
                    }
                }
            },
        ),
        CampaignExperiment(
            experiment_id="long300_verifier_heavy_v1",
            hypothesis=(
                "Now that 300 steps are clearly helping, a heavier compile/behavior mix may move "
                "the remaining gap on semantic metrics faster than style-oriented reward."
            ),
            short_description="300-step verifier heavy",
            overrides={
                "training": {
                    "reward_weights": {
                        "compile": 4.0,
                        "behavior": 4.0,
                        "cleanup": 0.75,
                        "readability": 0.25,
                    }
                }
            },
        ),
        CampaignExperiment(
            experiment_id="long300_verifier_heavy_v2",
            hypothesis=(
                "A more aggressive verifier-heavy mix with slightly less signature emphasis may "
                "improve compile and behavior if structure reward is still too dominant."
            ),
            short_description="300-step verifier heavy v2",
            overrides={
                "training": {
                    "reward_weights": {
                        "compile": 4.25,
                        "behavior": 4.25,
                        "signature": 1.25,
                        "readability": 0.25,
                    }
                }
            },
        ),
        CampaignExperiment(
            experiment_id="long300_readability_up_v1",
            hypothesis=(
                "The current best run already clears verifier gates. Increasing readability and "
                "cleanup could improve human-facing quality without giving back correctness."
            ),
            short_description="300-step readability up",
            overrides={
                "training": {
                    "reward_weights": {
                        "cleanup": 1.1,
                        "readability": 0.5,
                    }
                }
            },
        ),
        CampaignExperiment(
            experiment_id="long300_hallucination_up_v1",
            hypothesis=(
                "If the remaining invalid tail is mostly drift rather than length, stronger "
                "hallucination and decompiler-type penalties may tighten the policy."
            ),
            short_description="300-step hallucination up",
            overrides={
                "training": {
                    "reward_weights": {
                        "hallucination_penalty": 1.5,
                        "decompiler_type_penalty": 1.0,
                    }
                }
            },
        ),
        CampaignExperiment(
            experiment_id="long300_cool_sampling_v1",
            hypothesis=(
                "The best 300-step run may still benefit from slightly cooler rollout sampling. "
                "Reducing temperature and top-p could lower bad-tail variance without changing LR."
            ),
            short_description="300-step cool sampling",
            overrides={
                "training": {
                    "rollout_temperature": 0.9,
                    "rollout_top_p": 0.95,
                    "rollout_top_k": 50,
                }
            },
        ),
        CampaignExperiment(
            experiment_id="long300_cool_sampling_v2",
            hypothesis=(
                "A cooler sampler plus a mild repetition penalty may suppress redundant malformed "
                "structure more effectively than temperature alone."
            ),
            short_description="300-step cool sampling v2",
            overrides={
                "training": {
                    "rollout_temperature": 0.85,
                    "rollout_top_p": 0.9,
                    "rollout_top_k": 50,
                    "rollout_repetition_penalty": 1.02,
                }
            },
        ),
        CampaignExperiment(
            experiment_id="long300_repetition_v1",
            hypothesis=(
                "Malformed long answers may be reinforced by local repetition loops. A small "
                "repetition penalty without cooling the sampler tests that directly."
            ),
            short_description="300-step repetition",
            overrides={"training": {"rollout_repetition_penalty": 1.03}},
        ),
        CampaignExperiment(
            experiment_id="long300_invalid_contract_v1",
            hypothesis=(
                "The current best run still has seven parse failures. A stronger invalid-output "
                "contract may finally reduce that tail if given the full 300-step horizon."
            ),
            short_description="300-step invalid contract",
            overrides={
                "training": {
                    "max_invalid_completion_ratio": 0.8,
                    "reward_weights": {
                        "invalid_json_penalty": 0.35,
                        "invalid_length_penalty": 1.0,
                        "truncation_penalty": 2.0,
                    },
                }
            },
        ),
        CampaignExperiment(
            experiment_id="long300_invalid_contract_v2",
            hypothesis=(
                "Format and signature bias may only help if malformed JSON is clearly dominated. "
                "This combines stronger structure reward with the stricter invalid contract."
            ),
            short_description="300-step invalid contract v2",
            overrides={
                "training": {
                    "max_invalid_completion_ratio": 0.8,
                    "reward_weights": {
                        "format": 1.1,
                        "signature": 1.75,
                        "invalid_json_penalty": 0.35,
                        "invalid_length_penalty": 1.0,
                        "truncation_penalty": 2.0,
                    },
                }
            },
        ),
        CampaignExperiment(
            experiment_id="long300_scope_contract_v1",
            hypothesis=(
                "The scope penalty did not win by itself, but it may still help when combined "
                "with a slightly shorter budget and a stricter invalid-output contract."
            ),
            short_description="300-step scope contract",
            overrides={
                "training": {
                    "max_completion_length": 352,
                    "max_invalid_completion_ratio": 0.8,
                    "reward_weights": {
                        "invalid_scope_penalty": 2.0,
                        "invalid_json_penalty": 0.35,
                        "invalid_length_penalty": 1.0,
                        "truncation_penalty": 2.0,
                    },
                }
            },
        ),
    ]


def tag_for_now(now: datetime) -> str:
    return now.strftime("%Y%m%d-%H%M%S")


def _repo_root(start: Path | None = None) -> Path:
    return ProjectPaths.discover(start=start)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Sequence[PackedRLRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(row.model_dump_json() for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def _load_function_samples(path: Path, *, split: str) -> list[FunctionDatasetSample]:
    samples: list[FunctionDatasetSample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sample = FunctionDatasetSample.model_validate_json(line)
        if sample.split == split:
            samples.append(sample)
    return samples


def _allowed_callees(sample: FunctionDatasetSample) -> list[str]:
    return list(
        dict.fromkeys(
            [
                *sample.callees,
                *extract_called_functions(sample.target_clean_code),
                sample.source_function_name,
            ]
        )
    )


def pack_campaign_rl_records(
    samples: Sequence[FunctionDatasetSample],
    *,
    prompt_mode: str,
    task_types: Sequence[str],
    prompt_limit: int | None = None,
) -> list[PackedRLRecord]:
    if prompt_mode not in PROMPT_BUILDERS:
        raise CampaignError(
            f"unknown campaign prompt_mode={prompt_mode!r}; "
            f"expected one of {sorted(PROMPT_BUILDERS)}"
        )
    allowed_task_types = set(task_types)
    prompt_builder = PROMPT_BUILDERS[prompt_mode]
    selected = [
        sample
        for sample in samples
        if not allowed_task_types or sample.task_type in allowed_task_types
    ]
    if prompt_limit is not None:
        selected = selected[:prompt_limit]
    records: list[PackedRLRecord] = []
    for sample in selected:
        prompt = prompt_builder(sample)
        records.append(
            PackedRLRecord(
                sample_id=sample.sample_id,
                task_type=sample.task_type,
                prompt=prompt,
                prompt_messages=prompt_messages(prompt),
                source_function_name=sample.source_function_name,
                raw_code=sample.ghidra_decompiled_code,
                compile_reference_source=sample.compile_reference_source or sample.source_code,
                target_clean_code=sample.target_clean_code,
                target_renamings=json.dumps(sample.rename_map_target, sort_keys=True),
                allowed_imports=json.dumps(sample.imports),
                allowed_callees=json.dumps(_allowed_callees(sample)),
                compiler_executable=sample.compiler_executable,
                tests_ref=sample.tests_ref,
            )
        )
    return records


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _metrics_from_eval_manifest(manifest: Mapping[str, Any]) -> dict[str, float]:
    metrics = manifest.get("metrics", {})
    return {key: float(metrics.get(key, 0.0)) for key in _SCORE_KEYS}


def _normalized_path_ref(value: object) -> str:
    return str(value).replace("\\", "/").casefold()


def _profile_ref_matches(
    *,
    manifest_ref: object,
    root: Path,
    profile_path: Path,
) -> bool:
    try:
        relative_profile = profile_path.relative_to(root)
    except ValueError:
        relative_profile = profile_path
    normalized_manifest_ref = _normalized_path_ref(manifest_ref)
    return normalized_manifest_ref in {
        _normalized_path_ref(profile_path),
        _normalized_path_ref(relative_profile),
    }


def _hard_keep_gates_pass(metrics: Mapping[str, float], champion: Mapping[str, float]) -> bool:
    return (
        float(metrics.get("compile_success_rate", 0.0))
        >= float(champion.get("compile_success_rate", 0.0)) - 0.01
        and float(metrics.get("behavior_success_rate", 0.0))
        >= float(champion.get("behavior_success_rate", 0.0)) - 0.01
        and float(metrics.get("json_valid_rate", 0.0))
        >= float(champion.get("json_valid_rate", 0.0)) - 0.02
    )


def _sft_target_gates_pass(metrics: Mapping[str, float], sft_metrics: Mapping[str, float]) -> bool:
    return (
        float(metrics.get("compile_success_rate", 0.0))
        >= float(sft_metrics.get("compile_success_rate", 0.0)) - 0.01
        and float(metrics.get("behavior_success_rate", 0.0))
        >= float(sft_metrics.get("behavior_success_rate", 0.0)) - 0.01
        and float(metrics.get("json_valid_rate", 0.0))
        >= float(sft_metrics.get("json_valid_rate", 0.0)) - 0.02
    )


def sft_target_passed(
    metrics: Mapping[str, float],
    *,
    candidate_score: float,
    sft_metrics: Mapping[str, float],
    sft_score: float,
    target_improvement: float,
) -> bool:
    return (
        _sft_target_gates_pass(metrics, sft_metrics)
        and candidate_score >= sft_score + target_improvement
    )


def _run_dirs(root: Path, prefix: str) -> list[Path]:
    return sorted(
        (root / "artifacts" / "runs").glob(f"{prefix}-*"),
        key=lambda path: path.stat().st_mtime,
    )


def _run_logged(root: Path, logger: logging.Logger, args: list[str]) -> None:
    logger.info("running command: %s", " ".join(args))
    creationflags = 0
    if sys.platform == "win32":
        creationflags = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        )
    process = subprocess.Popen(
        args,
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=creationflags,
    )
    assert process.stdout is not None
    for line in process.stdout:
        logger.info(line.rstrip())
    returncode = process.wait()
    if returncode != 0:
        raise CampaignError(
            f"command failed with exit code {returncode}: {' '.join(args)}"
        )


def _new_run_from_command(root: Path, prefix: str, args: list[str], logger: logging.Logger) -> Path:
    before = {path.name for path in _run_dirs(root, prefix)}
    _run_logged(root, logger, args)
    after = [path for path in _run_dirs(root, prefix) if path.name not in before]
    if not after:
        raise CampaignError(f"command did not create a new {prefix} run")
    return after[-1]


def _reward_telemetry_from_train_dir(train_dir: Path) -> RewardTelemetrySnapshot | None:
    metrics_path = train_dir / "model" / "logs" / "grpo_metrics.jsonl"
    if not metrics_path.exists():
        return None
    for row in reversed(_read_jsonl(metrics_path)):
        if row.get("source") != "reward_func":
            continue
        return RewardTelemetrySnapshot(
            reward_mean=float(row.get("reward_mean", 0.0)),
            reward_std=float(row.get("reward_std", 0.0)),
            gate_factor_mean=float(row.get("components/gate_factor_mean", 0.0)),
            compile_mean=float(row.get("components/compile_mean", 0.0)),
            behavior_mean=float(row.get("components/behavior_mean", 0.0)),
            behavior_from_execution_mean=float(
                row.get("components/behavior_from_execution_mean", 0.0)
            ),
            json_valid_mean=float(row.get("components/json_valid_mean", 0.0)),
            signature_mean=float(row.get("components/signature_mean", 0.0)),
        )
    return None


def _invalid_reasons_from_eval_dir(eval_dir: Path | None) -> Counter[str]:
    if eval_dir is None:
        return Counter()
    predictions_path = eval_dir / "predictions.jsonl"
    if not predictions_path.exists():
        return Counter()
    return classify_invalid_prediction_rows(_read_jsonl(predictions_path))


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise CampaignError(f"expected mapping in {path}")
    return data


def apply_training_overrides(
    base_payload: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    def merge(left: Any, right: Any) -> Any:
        if isinstance(left, dict) and isinstance(right, Mapping):
            merged = dict(left)
            for key, value in right.items():
                merged[key] = merge(merged.get(key), value)
            return merged
        return right

    return merge(dict(base_payload), overrides)


def _post_target_campaign_candidates() -> list[CampaignExperiment]:
    strict_context_data = {
        "prompt_mode": "context_plus_strict",
        "task_types": ["full_clarify", "cleanup"],
        "prompt_limit": 1000,
    }
    strict_context_full_only = {
        "prompt_mode": "context_plus_strict",
        "task_types": ["full_clarify"],
        "prompt_limit": 1000,
    }

    def strict_training(extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return apply_training_overrides(
            {"max_seq_length": 1792, "max_prompt_length": 1024},
            dict(extra or {}),
        )

    def quick_overrides(
        *,
        max_steps: int = 25,
        save_steps: int | None = None,
    ) -> dict[str, Any]:
        return {
            "training": {
                "max_steps": max_steps,
                "save_steps": save_steps if save_steps is not None else max_steps,
            }
        }

    def candidate(
        experiment_id: str,
        hypothesis: str,
        training: Mapping[str, Any] | None = None,
        *,
        max_steps: int = 25,
        dataset_overrides: dict[str, Any] | None = None,
    ) -> CampaignExperiment:
        return CampaignExperiment(
            experiment_id=experiment_id,
            hypothesis=hypothesis,
            short_description=experiment_id.removeprefix("post_").replace("_", " "),
            overrides=apply_training_overrides(
                quick_overrides(max_steps=max_steps),
                {"training": dict(training or {})},
            ),
            dataset_overrides=dataset_overrides or {},
        )

    return [
        candidate(
            "post_scope_penalty_175_q25",
            "Test whether the winning scope guard is slightly too strong.",
            {"reward_weights": {"invalid_scope_penalty": 1.75}},
        ),
        candidate(
            "post_scope_penalty_225_q25",
            "Test whether a small scope-penalty increase improves the g19 tail.",
            {"reward_weights": {"invalid_scope_penalty": 2.25}},
        ),
        candidate(
            "post_scope_penalty_250_q25",
            "Probe the upper edge before the failed v2 scope penalty.",
            {"reward_weights": {"invalid_scope_penalty": 2.5}},
        ),
        candidate(
            "post_scope_penalty_275_q25",
            "Check whether scope pressure fails gradually or only near 3.0.",
            {"reward_weights": {"invalid_scope_penalty": 2.75}},
        ),
        candidate(
            "post_invalid_ratio_055_q25",
            "Tighten the invalid-completion allowance below the winning run.",
            {"max_invalid_completion_ratio": 0.55},
        ),
        candidate(
            "post_invalid_ratio_075_q25",
            "Relax the invalid-completion allowance to see if v1 was over-gated.",
            {"max_invalid_completion_ratio": 0.75},
        ),
        candidate(
            "post_ratio_140_q25",
            "Constrain relative output length without reducing token budget.",
            {"max_completion_ratio": 1.4},
        ),
        candidate(
            "post_ratio_130_q25",
            "Apply a stronger output-ratio guard while keeping 384 tokens.",
            {"max_completion_ratio": 1.3},
        ),
        candidate(
            "post_completion_352_q25",
            "Try a modestly shorter completion budget than the g19 champion.",
            {"max_completion_length": 352},
        ),
        candidate(
            "post_completion_416_q25",
            "Check whether a little more completion budget helps behavior wins.",
            {"max_completion_length": 416},
        ),
        candidate(
            "post_completion_320_soft_q25",
            "Retest 320 tokens with the g19 scope guard instead of the 256 contract.",
            {"max_completion_length": 320, "reward_weights": {"truncation_penalty": 2.5}},
        ),
        candidate(
            "post_completion_448_q25",
            "Test whether the full-prompt eval wants more room than 384 tokens.",
            {"max_completion_length": 448},
        ),
        candidate(
            "post_behavior_375_q25",
            "Nudge behavior reward after g19 improved hard metrics.",
            {"reward_weights": {"behavior": 3.75}},
        ),
        candidate(
            "post_behavior_400_q25",
            "Probe a stronger behavior-only nudge from the g19 profile.",
            {"reward_weights": {"behavior": 4.0}},
        ),
        candidate(
            "post_compile_375_q25",
            "Nudge compile reward while preserving the behavior balance.",
            {"reward_weights": {"compile": 3.75}},
        ),
        candidate(
            "post_compile_400_q25",
            "Probe stronger compile pressure from the g19 profile.",
            {"reward_weights": {"compile": 4.0}},
        ),
        candidate(
            "post_verifier_375_q25",
            "Raise compile and behavior together without changing style weights.",
            {"reward_weights": {"compile": 3.75, "behavior": 3.75}},
        ),
        candidate(
            "post_verifier_400_sig125_q25",
            "Trade some signature pressure for stronger verifier rewards.",
            {"reward_weights": {"compile": 4.0, "behavior": 4.0, "signature": 1.25}},
        ),
        candidate(
            "post_signature_175_q25",
            "Check whether signature fidelity is still the limiting safety signal.",
            {"reward_weights": {"signature": 1.75}},
        ),
        candidate(
            "post_format_120_q25",
            "Increase format pressure now that JSON validity is at 1.0.",
            {"reward_weights": {"format": 1.2}},
        ),
        candidate(
            "post_cleanup_readability_q25",
            "See if human-facing quality can rise without giving back verifier wins.",
            {"reward_weights": {"cleanup": 1.1, "readability": 0.5}},
        ),
        candidate(
            "post_hallucination_150_q25",
            "Suppress drift with a stronger hallucination penalty.",
            {"reward_weights": {"hallucination_penalty": 1.5}},
        ),
        candidate(
            "post_constant_300_q25",
            "Increase invented-constant pressure while keeping scope at g19.",
            {"reward_weights": {"unknown_constant_penalty": 3.0}},
        ),
        candidate(
            "post_bool_100_q25",
            "Add a light bool/type guard without the failed heavy type mix.",
            {"reward_weights": {"unsupported_bool_penalty": 1.0}},
        ),
        candidate(
            "post_constant_bool_q25",
            "Combine light constant and bool guards after the g19 win.",
            {
                "reward_weights": {
                    "unknown_constant_penalty": 2.5,
                    "unsupported_bool_penalty": 1.0,
                }
            },
        ),
        candidate(
            "post_cool_sampling_q25",
            "Cool rollout sampling to reduce bad-tail variance.",
            {"rollout_temperature": 0.9, "rollout_top_p": 0.95, "rollout_top_k": 50},
        ),
        candidate(
            "post_cool_sampling_rep_q25",
            "Combine cooler sampling with a tiny repetition penalty.",
            {
                "rollout_temperature": 0.85,
                "rollout_top_p": 0.9,
                "rollout_top_k": 50,
                "rollout_repetition_penalty": 1.02,
            },
        ),
        candidate(
            "post_warm_sampling_q25",
            "Test whether a bit more rollout diversity helps find behavior wins.",
            {"rollout_temperature": 1.05, "rollout_top_p": 1.0},
        ),
        candidate(
            "post_top_p_095_q25",
            "Lower top-p without changing temperature.",
            {"rollout_top_p": 0.95},
        ),
        candidate(
            "post_repetition_102_q25",
            "Add a small repetition penalty without changing sampling temperature.",
            {"rollout_repetition_penalty": 1.02},
        ),
        candidate(
            "post_lr_low_q50",
            "Run a longer low-LR pilot to test stability around g19.",
            {"learning_rate": 1.0e-6, "warmup_ratio": 0.05},
            max_steps=50,
        ),
        candidate(
            "post_lr_high_q50",
            "Run a longer high-LR pilot to test whether g19 is under-updating.",
            {"learning_rate": 2.0e-6, "warmup_ratio": 0.05},
            max_steps=50,
        ),
        candidate(
            "post_gradaccum2_q50",
            "Use a higher effective batch to reduce GRPO update noise.",
            {"grad_accum_steps": 2, "learning_rate": 1.0e-6, "warmup_ratio": 0.05},
            max_steps=50,
        ),
        candidate(
            "post_generations5_q50",
            "Use five rollouts to improve group baselines without the failed g6 jump.",
            {"generations_per_prompt": 5},
            max_steps=50,
        ),
        candidate(
            "post_generations3_q50",
            "Use three rollouts to test whether cheaper updates preserve the signal.",
            {"generations_per_prompt": 3},
            max_steps=50,
        ),
        candidate(
            "post_context_plus_strict_data_q50",
            "Retest context-plus strict data from the g19 reward profile.",
            {"max_seq_length": 1792, "max_prompt_length": 1024},
            max_steps=50,
            dataset_overrides={
                "prompt_mode": "context_plus_strict",
                "task_types": ["full_clarify", "cleanup"],
                "prompt_limit": 1000,
            },
        ),
        candidate(
            "post_context_plus_data_q50",
            "Retest context-plus data without the strict prompt wording.",
            {"max_seq_length": 1792, "max_prompt_length": 1024},
            max_steps=50,
            dataset_overrides={
                "prompt_mode": "context_plus",
                "task_types": ["full_clarify", "cleanup"],
                "prompt_limit": 1000,
            },
        ),
        candidate(
            "post_full_prompt_data_q50",
            "Retest full-prompt data now that the reward contract is safer.",
            {"max_seq_length": 2048, "max_prompt_length": 1152},
            max_steps=50,
            dataset_overrides={
                "prompt_mode": "full",
                "task_types": ["full_clarify", "cleanup"],
                "prompt_limit": 1000,
            },
        ),
        candidate(
            "post_full_only_data_q50",
            "Train only full-clarify rows to remove cleanup task interference.",
            {},
            max_steps=50,
            dataset_overrides={
                "prompt_mode": "context_plus_strict",
                "task_types": ["full_clarify"],
                "prompt_limit": 1000,
            },
        ),
        candidate(
            "post_cleanup_only_data_q50",
            "Train only cleanup rows to test if structure repair is the useful task.",
            {},
            max_steps=50,
            dataset_overrides={
                "prompt_mode": "context_plus_strict",
                "task_types": ["cleanup"],
                "prompt_limit": 1000,
            },
        ),
        candidate(
            "post2_strict_data_q200",
            "Give the strongest post-target data recipe a longer update horizon.",
            strict_training(),
            max_steps=200,
            dataset_overrides=dict(strict_context_data),
        ),
        candidate(
            "post2_strict_data_lr_low_q200",
            "Test whether the strict data lift needs slower long-horizon updates.",
            strict_training({"learning_rate": 1.0e-6, "warmup_ratio": 0.05}),
            max_steps=200,
            dataset_overrides=dict(strict_context_data),
        ),
        candidate(
            "post2_strict_data_lr_high_q200",
            "Test whether the strict data lift is under-updating at the champion LR.",
            strict_training({"learning_rate": 2.0e-6, "warmup_ratio": 0.05}),
            max_steps=200,
            dataset_overrides=dict(strict_context_data),
        ),
        candidate(
            "post2_strict_data_completion448_q200",
            "Allow more completion room for strict-context examples over a longer horizon.",
            strict_training({"max_completion_length": 448}),
            max_steps=200,
            dataset_overrides=dict(strict_context_data),
        ),
        candidate(
            "post2_strict_data_completion320_q200",
            "Check whether shorter strict-context answers retain the hard-metric lift.",
            strict_training({"max_completion_length": 320}),
            max_steps=200,
            dataset_overrides=dict(strict_context_data),
        ),
        candidate(
            "post2_strict_data_ratio140_q200",
            "Apply the best lightweight length-ratio signal to strict-context data.",
            strict_training({"max_completion_ratio": 1.4}),
            max_steps=200,
            dataset_overrides=dict(strict_context_data),
        ),
        candidate(
            "post2_strict_data_verifier375_q200",
            "Nudge compile and behavior rewards on the strict-context data recipe.",
            strict_training({"reward_weights": {"compile": 3.75, "behavior": 3.75}}),
            max_steps=200,
            dataset_overrides=dict(strict_context_data),
        ),
        candidate(
            "post2_strict_data_verifier400_q200",
            "Use stronger verifier pressure with less signature dominance.",
            strict_training(
                {"reward_weights": {"compile": 4.0, "behavior": 4.0, "signature": 1.25}}
            ),
            max_steps=200,
            dataset_overrides=dict(strict_context_data),
        ),
        candidate(
            "post2_strict_data_scope175_q200",
            "Relax scope pressure inside the strict-context data recipe.",
            strict_training({"reward_weights": {"invalid_scope_penalty": 1.75}}),
            max_steps=200,
            dataset_overrides=dict(strict_context_data),
        ),
        candidate(
            "post2_strict_data_scope250_q200",
            "Increase scope pressure inside the strict-context data recipe.",
            strict_training({"reward_weights": {"invalid_scope_penalty": 2.5}}),
            max_steps=200,
            dataset_overrides=dict(strict_context_data),
        ),
        candidate(
            "post2_strict_data_g3_q200",
            "Combine the g3 rollout signal with strict-context data for a longer horizon.",
            strict_training({"generations_per_prompt": 3}),
            max_steps=200,
            dataset_overrides=dict(strict_context_data),
        ),
        candidate(
            "post2_strict_data_g5_q200",
            "Retest five rollouts only where strict-context data improved hard metrics.",
            strict_training({"generations_per_prompt": 5}),
            max_steps=200,
            dataset_overrides=dict(strict_context_data),
        ),
        candidate(
            "post2_strict_data_gradaccum2_q200",
            "Reduce update noise for the strict-context data recipe.",
            strict_training(
                {
                    "grad_accum_steps": 2,
                    "learning_rate": 1.0e-6,
                    "warmup_ratio": 0.05,
                }
            ),
            max_steps=200,
            dataset_overrides=dict(strict_context_data),
        ),
        candidate(
            "post2_strict_full_only_q200",
            "Give full-clarify-only strict-context data a longer horizon.",
            strict_training(),
            max_steps=200,
            dataset_overrides=dict(strict_context_full_only),
        ),
        candidate(
            "post2_strict_full_only_verifier_q200",
            "Pair full-clarify-only strict-context data with stronger verifier rewards.",
            strict_training({"reward_weights": {"compile": 4.0, "behavior": 4.0}}),
            max_steps=200,
            dataset_overrides=dict(strict_context_full_only),
        ),
        candidate(
            "post2_strict_data_limit1500_q200",
            "Broaden the strict-context prompt pool while keeping the same task mix.",
            strict_training(),
            max_steps=200,
            dataset_overrides={
                **strict_context_data,
                "prompt_limit": 1500,
            },
        ),
        candidate(
            "post2_strict_data_readability050_q200",
            "Check whether readability pressure can improve quality after hard metrics lift.",
            strict_training({"reward_weights": {"readability": 0.5, "cleanup": 1.1}}),
            max_steps=200,
            dataset_overrides=dict(strict_context_data),
        ),
        candidate(
            "post2_strict_data_constant300_q200",
            "Retest stronger invented-constant pressure only on strict-context data.",
            strict_training({"reward_weights": {"unknown_constant_penalty": 3.0}}),
            max_steps=200,
            dataset_overrides=dict(strict_context_data),
        ),
        candidate(
            "post2_strict_data_sampling_cool_q200",
            "Cool rollout sampling where strict-context data produced the best new signal.",
            strict_training(
                {
                    "rollout_temperature": 0.9,
                    "rollout_top_p": 0.95,
                    "rollout_top_k": 50,
                }
            ),
            max_steps=200,
            dataset_overrides=dict(strict_context_data),
        ),
        candidate(
            "post2_strict_data_sampling_warm_q200",
            "Add mild rollout diversity to strict-context data over a longer horizon.",
            strict_training({"rollout_temperature": 1.05, "rollout_top_p": 1.0}),
            max_steps=200,
            dataset_overrides=dict(strict_context_data),
        ),
    ]


def choose_campaign_experiment(
    champion_metrics: Mapping[str, float],
    invalid_reasons: Mapping[str, int],
    prior_experiment_ids: set[str],
    *,
    reward_telemetry: RewardTelemetrySnapshot | None = None,
    search_space: str = DEFAULT_SEARCH_SPACE,
) -> CampaignExperiment | None:
    if search_space == "long300":
        for candidate in _long300_campaign_candidates():
            if candidate.experiment_id not in prior_experiment_ids:
                return candidate
        return None
    if search_space == "post_target":
        for candidate in _post_target_campaign_candidates():
            if candidate.experiment_id not in prior_experiment_ids:
                return candidate
        return None

    def quick_overrides(*, max_steps: int = 25, save_steps: int | None = None) -> dict[str, Any]:
        return {
            "training": {
                "max_steps": max_steps,
                "save_steps": save_steps if save_steps is not None else max_steps,
            }
        }

    json_valid_rate = float(champion_metrics.get("json_valid_rate", 0.0))
    compile_rate = float(champion_metrics.get("compile_success_rate", 0.0))
    behavior_rate = float(champion_metrics.get("behavior_success_rate", 0.0))
    parse_errors = int(invalid_reasons.get("json_parse_error", 0))
    low_json = json_valid_rate < 0.94 or parse_errors > 0
    reward_plateau = (
        reward_telemetry is not None
        and reward_telemetry.reward_mean > 0.0
        and reward_telemetry.compile_mean < 0.5
        and reward_telemetry.behavior_mean < 0.35
    )

    candidates = [
        CampaignExperiment(
            experiment_id="prompt_align_full_v1",
            hypothesis=(
                "The best checkpoint evaluation uses the full binary-grounded prompt, while the "
                "current RL dataset trains on a compact prompt. Training GDPO on the same full "
                "prompt contract, including rename rows and a small naming reward, should make "
                "the RL update optimize the distribution we actually score."
            ),
            short_description="full-prompt RL alignment",
            overrides={
                "training": {
                    "max_seq_length": 2048,
                    "max_prompt_length": 1152,
                    "max_completion_length": 384,
                    "reward_weights": {
                        "naming": 0.25,
                    },
                }
            },
            dataset_overrides={
                "prompt_mode": "full",
                "task_types": ["full_clarify", "cleanup", "rename"],
                "prompt_limit": 1000,
            },
        ),
        CampaignExperiment(
            experiment_id="prompt_align_context_plus_v1",
            hypothesis=(
                "Restoring strings and caller/callee context without the full assembly block may "
                "capture most of the useful eval-time grounding while keeping prompts shorter "
                "and less likely to truncate inside GRPO."
            ),
            short_description="context-plus RL alignment",
            overrides={
                "training": {
                    "max_seq_length": 1792,
                    "max_prompt_length": 1024,
                    "max_completion_length": 384,
                    "reward_weights": {
                        "naming": 0.2,
                    },
                }
            },
            dataset_overrides={
                "prompt_mode": "context_plus",
                "task_types": ["full_clarify", "cleanup", "rename"],
                "prompt_limit": 1000,
            },
        ),
        CampaignExperiment(
            experiment_id="prompt_align_context_plus_no_rename_v1",
            hypothesis=(
                "Context-plus training recovered the reward signal and improved JSON, naming, "
                "and readability, but gave back one compile and behavior case on scout. Removing "
                "rename-only rows and leaving naming reward at zero should preserve the extra "
                "metadata while moving pressure back to verifier-safe full_clarify and cleanup."
            ),
            short_description="context-plus no-rename RL",
            overrides={
                "training": {
                    "max_seq_length": 1792,
                    "max_prompt_length": 1024,
                    "max_completion_length": 384,
                }
            },
            dataset_overrides={
                "prompt_mode": "context_plus",
                "task_types": ["full_clarify", "cleanup"],
                "prompt_limit": 1000,
            },
        ),
        CampaignExperiment(
            experiment_id="context_plus_constant_guard_v1",
            hypothesis=(
                "The context-plus no-rename run gained one verifier-safe count_flag sample, but "
                "lost two split_kv samples by inventing MAX_VALUE instead of preserving MAX_VAL. "
                "Adding an opt-in unknown-constant penalty should keep the useful context-plus "
                "behavior while discouraging out-of-scope macro substitutions."
            ),
            short_description="context-plus constant guard",
            overrides={
                "training": {
                    "max_seq_length": 1792,
                    "max_prompt_length": 1024,
                    "max_completion_length": 384,
                    "reward_weights": {
                        "unknown_constant_penalty": 2.0,
                    },
                }
            },
            dataset_overrides={
                "prompt_mode": "context_plus",
                "task_types": ["full_clarify", "cleanup"],
                "prompt_limit": 1000,
            },
        ),
        CampaignExperiment(
            experiment_id="context_plus_constant_strict_json_v1",
            hypothesis=(
                "The constant guard produced a net scout gain in compile and behavior, but it "
                "lost one JSON sample through a long main rewrite with a bloated rename map. "
                "Keeping the constant guard while using stricter renaming instructions and "
                "stronger malformed-output penalties should preserve the hard-metric lift "
                "without sacrificing JSON validity."
            ),
            short_description="constant guard strict JSON",
            overrides={
                "training": {
                    "max_seq_length": 1792,
                    "max_prompt_length": 1024,
                    "max_completion_length": 384,
                    "max_invalid_completion_ratio": 0.65,
                    "reward_weights": {
                        "format": 1.1,
                        "invalid_json_penalty": 0.6,
                        "invalid_length_penalty": 1.5,
                        "truncation_penalty": 3.0,
                        "invalid_scope_penalty": 3.0,
                        "unknown_constant_penalty": 2.0,
                    },
                }
            },
            dataset_overrides={
                "prompt_mode": "context_plus_strict",
                "task_types": ["full_clarify", "cleanup"],
                "prompt_limit": 1000,
            },
        ),
        CampaignExperiment(
            experiment_id="context_plus_strict_type_guard_v1",
            hypothesis=(
                "The strict JSON run is close to the SFT+0.02 target, but paired losses show "
                "two remaining compile failures from inventing MAX_VALUE and two from changing "
                "compile-safe integer flags into bool. Increasing constant/type-safety penalties "
                "and nudging behavior weight should trade less syntax safety for more validator "
                "wins."
            ),
            short_description="strict type and constant guard",
            overrides={
                "training": {
                    "reward_weights": {
                        "behavior": 4.0,
                        "compile": 3.75,
                        "unknown_constant_penalty": 4.0,
                        "unsupported_bool_penalty": 2.0,
                    },
                }
            },
            dataset_overrides={
                "prompt_mode": "context_plus_strict",
                "task_types": ["full_clarify", "cleanup"],
                "prompt_limit": 1000,
            },
        ),
        CampaignExperiment(
            experiment_id="context_plus_strict_behavior_nudge_v1",
            hypothesis=(
                "The type/constant-heavy variant protected JSON but lost compile on scout, so "
                "the safer next move is to keep the strict JSON champion contract unchanged and "
                "only nudge behavior reward upward. If the generation-5 gains are real, this "
                "should look for one or two more behavior wins without over-constraining syntax."
            ),
            short_description="strict behavior nudge",
            overrides={
                "training": {
                    "reward_weights": {
                        "behavior": 4.0,
                    },
                }
            },
            dataset_overrides={
                "prompt_mode": "context_plus_strict",
                "task_types": ["full_clarify", "cleanup"],
                "prompt_limit": 1000,
            },
        ),
        CampaignExperiment(
            experiment_id="context_plus_strict_long500_v1",
            hypothesis=(
                "The reward-weight nudges after the strict JSON champion both regressed, which "
                "points away from reweighting and toward optimization horizon. A 500-step run at "
                "a lower learning rate keeps the generation-5 reward contract intact while "
                "giving the stable signal more time to move behavior."
            ),
            short_description="strict JSON long 500",
            overrides={
                "training": {
                    "max_steps": 500,
                    "save_steps": 100,
                    "learning_rate": 1.0e-6,
                    "warmup_ratio": 0.05,
                }
            },
            dataset_overrides={
                "prompt_mode": "context_plus_strict",
                "task_types": ["full_clarify", "cleanup"],
                "prompt_limit": 1000,
            },
        ),
        CampaignExperiment(
            experiment_id="prompt_align_full_no_rename_v1",
            hypothesis=(
                "If rename rows add noisy pressure, the full binary-grounded prompt may still "
                "help when restricted to full_clarify and cleanup, matching the current curated "
                "RL task subset while fixing the prompt mismatch."
            ),
            short_description="full-prompt no-rename RL",
            overrides={
                "training": {
                    "max_seq_length": 2048,
                    "max_prompt_length": 1152,
                    "max_completion_length": 384,
                }
            },
            dataset_overrides={
                "prompt_mode": "full",
                "task_types": ["full_clarify", "cleanup"],
                "prompt_limit": 1000,
            },
        ),
        CampaignExperiment(
            experiment_id="completion_256_contract_v1",
            hypothesis=(
                "Reducing the GRPO completion budget to 256 tokens and tightening invalid-output "
                "penalties should cut truncated JSON and keep the model focused on a single "
                "function-sized answer."
            ),
            short_description="completion 256 contract",
            overrides={
                "training": {
                    "max_completion_length": 256,
                    "max_prompt_length": 896,
                    "max_invalid_completion_ratio": 0.75,
                    "reward_weights": {
                        "format": 1.15,
                        "signature": 1.25,
                        "cleanup": 1.1,
                        "readability": 0.4,
                        "invalid_json_penalty": 0.3,
                        "invalid_length_penalty": 1.0,
                        "truncation_penalty": 2.0,
                    },
                }
            },
        ),
        CampaignExperiment(
            experiment_id="invalid_scope_guard_v1",
            hypothesis=(
                "The remaining invalid JSON cases are often whole-program rewrites inside a "
                "truncated `cleaned_c` field. Penalizing raw completions that define non-target "
                "functions should push GRPO back toward single-function clarifications."
            ),
            short_description="invalid scope guard",
            overrides={
                "training": {
                    "reward_weights": {
                        "invalid_scope_penalty": 2.0,
                    },
                }
            },
        ),
        CampaignExperiment(
            experiment_id="invalid_scope_guard_v2",
            hypothesis=(
                "The raw scope guard helped a little but barely fired. A stronger version should "
                "penalize off-target whole-program rewrites often enough to matter in short pilots."
            ),
            short_description="invalid scope guard v2",
            overrides=apply_training_overrides(
                quick_overrides(),
                {
                    "training": {
                        "reward_weights": {
                            "invalid_scope_penalty": 3.0,
                        },
                    }
                },
            ),
        ),
        CampaignExperiment(
            experiment_id="invalid_scope_truncation_v1",
            hypothesis=(
                "If invalid outputs are mostly truncated whole-program rewrites, combining raw "
                "scope penalties with stronger truncation penalties should reduce the bad tail."
            ),
            short_description="scope truncation mix",
            overrides=apply_training_overrides(
                quick_overrides(),
                {
                    "training": {
                        "reward_weights": {
                            "invalid_scope_penalty": 2.0,
                            "invalid_json_penalty": 0.4,
                            "invalid_length_penalty": 1.25,
                            "truncation_penalty": 3.0,
                        },
                    }
                },
            ),
        ),
        CampaignExperiment(
            experiment_id="invalid_scope_truncation_v2",
            hypothesis=(
                "A tighter invalid-output contract may work better if both truncation and scope "
                "drift are treated as major failures instead of mild penalties."
            ),
            short_description="scope truncation hard",
            overrides=apply_training_overrides(
                quick_overrides(),
                {
                    "training": {
                        "max_invalid_completion_ratio": 0.75,
                        "reward_weights": {
                            "invalid_scope_penalty": 3.0,
                            "invalid_json_penalty": 0.4,
                            "invalid_length_penalty": 1.25,
                            "truncation_penalty": 3.0,
                        },
                    }
                },
            ),
        ),
        CampaignExperiment(
            experiment_id="completion320_scope_v1",
            hypothesis=(
                "The 256-token contract was too tight, but 384 still clips. A 320-token budget "
                "with raw scope penalties may be a better middle ground for Qwen 2B."
            ),
            short_description="completion 320 scope",
            overrides=apply_training_overrides(
                quick_overrides(),
                {
                    "training": {
                        "max_completion_length": 320,
                        "reward_weights": {
                            "invalid_scope_penalty": 2.0,
                            "truncation_penalty": 2.5,
                        },
                    }
                },
            ),
        ),
        CampaignExperiment(
            experiment_id="completion288_scope_v1",
            hypothesis=(
                "A slightly shorter 288-token completion budget may suppress overlong invalid "
                "JSON without the severe damage from the full 256-token contract."
            ),
            short_description="completion 288 scope",
            overrides=apply_training_overrides(
                quick_overrides(),
                {
                    "training": {
                        "max_completion_length": 288,
                        "reward_weights": {
                            "invalid_scope_penalty": 2.0,
                            "truncation_penalty": 2.5,
                        },
                    }
                },
            ),
        ),
        CampaignExperiment(
            experiment_id="signature_contract_scope_v1",
            hypothesis=(
                "A stronger signature and format bias may help the model finish the intended "
                "single-function JSON object instead of drifting into larger rewrites."
            ),
            short_description="signature contract scope",
            overrides=apply_training_overrides(
                quick_overrides(),
                {
                    "training": {
                        "reward_weights": {
                            "format": 1.15,
                            "signature": 1.75,
                            "invalid_scope_penalty": 2.0,
                            "truncation_penalty": 2.5,
                        },
                    }
                },
            ),
        ),
        CampaignExperiment(
            experiment_id="behavior_heavy_scope_v1",
            hypothesis=(
                "If scope drift is masking semantic improvement, a heavier compile/behavior mix "
                "plus scope penalties may improve verifier-backed gains faster than the champion."
            ),
            short_description="behavior heavy scope",
            overrides=apply_training_overrides(
                quick_overrides(),
                {
                    "training": {
                        "reward_weights": {
                            "compile": 4.0,
                            "behavior": 4.0,
                            "readability": 0.3,
                            "invalid_scope_penalty": 2.0,
                        },
                    }
                },
            ),
        ),
        CampaignExperiment(
            experiment_id="long_horizon_scope_v1",
            hypothesis=(
                "The scope guard may need a slightly longer horizon to influence policy; a short "
                "but not minimal run can test whether the effect compounds past the first pilot."
            ),
            short_description="long horizon scope",
            overrides=apply_training_overrides(
                quick_overrides(max_steps=35),
                {
                    "training": {
                        "learning_rate": 1.0e-6,
                        "warmup_ratio": 0.05,
                        "reward_weights": {
                            "invalid_scope_penalty": 2.0,
                            "truncation_penalty": 2.5,
                        },
                    }
                },
            ),
        ),
        CampaignExperiment(
            experiment_id="sampling_scope_balance_v1",
            hypothesis=(
                "A mildly cooled sampler may help only when combined with the raw scope guard, "
                "because the guard removes the biggest off-target completions."
            ),
            short_description="sampling scope balance",
            overrides=apply_training_overrides(
                quick_overrides(),
                {
                    "training": {
                        "rollout_temperature": 0.9,
                        "rollout_top_p": 0.95,
                        "rollout_top_k": 50,
                        "rollout_repetition_penalty": 1.01,
                        "reward_weights": {
                            "invalid_scope_penalty": 2.0,
                            "truncation_penalty": 2.5,
                        },
                    }
                },
            ),
        ),
        CampaignExperiment(
            experiment_id="sampling_scope_balance_v2",
            hypothesis=(
                "A cooler sampler and a slightly shorter completion budget may reduce truncated "
                "whole-program JSON more reliably than either change alone."
            ),
            short_description="sampling scope balance v2",
            overrides=apply_training_overrides(
                quick_overrides(),
                {
                    "training": {
                        "max_completion_length": 320,
                        "rollout_temperature": 0.85,
                        "rollout_top_p": 0.9,
                        "rollout_top_k": 50,
                        "rollout_repetition_penalty": 1.02,
                        "reward_weights": {
                            "invalid_scope_penalty": 2.0,
                            "truncation_penalty": 2.5,
                        },
                    }
                },
            ),
        ),
        CampaignExperiment(
            experiment_id="format_scope_bias_v1",
            hypothesis=(
                "If the bad tail is mostly malformed structured output, increasing format bias "
                "together with the scope guard may improve JSON validity without a 256-token cap."
            ),
            short_description="format scope bias",
            overrides=apply_training_overrides(
                quick_overrides(),
                {
                    "training": {
                        "reward_weights": {
                            "format": 1.2,
                            "readability": 0.3,
                            "invalid_scope_penalty": 2.0,
                            "invalid_json_penalty": 0.35,
                            "truncation_penalty": 2.5,
                        },
                    }
                },
            ),
        ),
        CampaignExperiment(
            experiment_id="safety_signature_rebalance_v1",
            hypothesis=(
                "Rebalancing toward compile, behavior, and signature fidelity should preserve "
                "the guarded pilot's structural gains better than the current cosmetic-heavy mix."
            ),
            short_description="safety signature rebalance",
            overrides={
                "training": {
                    "reward_weights": {
                        "compile": 3.5,
                        "behavior": 3.5,
                        "signature": 1.5,
                        "cleanup": 1.0,
                        "readability": 0.35,
                        "hallucination_penalty": 1.25,
                        "overshoot_penalty": 1.0,
                        "multi_function_penalty": 3.0,
                    },
                    "max_completion_ratio": 1.5,
                    "max_function_count": 1,
                }
            },
        ),
        CampaignExperiment(
            experiment_id="batch2_completion256_v1",
            hypothesis=(
                "Using a shorter completion budget to fit a larger per-device batch should make "
                "GRPO updates less noisy on Qwen 2B than the current batch-1 setup."
            ),
            short_description="batch2 completion 256",
            overrides={
                "training": {
                    "batch_size": 2,
                    "grad_accum_steps": 1,
                    "max_seq_length": 1408,
                    "max_prompt_length": 768,
                    "max_completion_length": 256,
                    "reward_weights": {
                        "format": 1.1,
                        "signature": 1.25,
                        "invalid_json_penalty": 0.3,
                        "invalid_length_penalty": 1.0,
                        "truncation_penalty": 2.0,
                    },
                }
            },
        ),
        CampaignExperiment(
            experiment_id="long_horizon_stable_v1",
            hypothesis=(
                "If the reward path is alive but plateaued, extending the horizon with a lower "
                "learning rate and higher effective batch should let the guarded reward signal "
                "translate into validation gains."
            ),
            short_description="long horizon stable",
            overrides={
                "training": {
                    "max_steps": 200,
                    "save_steps": 50,
                    "learning_rate": 1.0e-6,
                    "warmup_ratio": 0.05,
                    "grad_accum_steps": 2,
                }
            },
        ),
        CampaignExperiment(
            experiment_id="long_horizon_completion256_v1",
            hypothesis=(
                "Combining the shorter completion contract with a longer, lower-LR run should "
                "improve JSON reliability without giving back compile and behavior."
            ),
            short_description="long horizon completion 256",
            overrides={
                "training": {
                    "max_steps": 200,
                    "save_steps": 50,
                    "learning_rate": 1.0e-6,
                    "warmup_ratio": 0.05,
                    "grad_accum_steps": 2,
                    "max_completion_length": 256,
                    "max_invalid_completion_ratio": 0.75,
                    "reward_weights": {
                        "format": 1.15,
                        "signature": 1.25,
                        "invalid_json_penalty": 0.3,
                        "invalid_length_penalty": 1.0,
                        "truncation_penalty": 2.0,
                    },
                }
            },
        ),
        CampaignExperiment(
            experiment_id="exploration_boost_v1",
            hypothesis=(
                "The current GRPO setup may simply be too conservative. Increasing rollout "
                "diversity and the update size should test whether Qwen 2B is stuck near the SFT "
                "policy because it is not exploring enough."
            ),
            short_description="exploration boost",
            overrides={
                "training": {
                    "max_steps": 150,
                    "save_steps": 50,
                    "learning_rate": 2.5e-6,
                    "warmup_ratio": 0.05,
                    "generations_per_prompt": 6,
                    "max_grad_norm": 0.2,
                }
            },
        ),
        CampaignExperiment(
            experiment_id="exploration_safety_mix_v1",
            hypothesis=(
                "If extra exploration helps but drifts too far, pairing it with stronger "
                "compile and behavior weighting may produce a better curve than the guarded "
                "baseline."
            ),
            short_description="exploration safety mix",
            overrides={
                "training": {
                    "max_steps": 150,
                    "save_steps": 50,
                    "learning_rate": 2.0e-6,
                    "warmup_ratio": 0.05,
                    "generations_per_prompt": 6,
                    "max_grad_norm": 0.2,
                    "reward_weights": {
                        "compile": 3.5,
                        "behavior": 4.0,
                        "signature": 1.25,
                        "cleanup": 1.1,
                        "readability": 0.4,
                    },
                }
            },
        ),
    ]

    priority: list[str] = []
    if json_valid_rate >= 0.97 and behavior_rate < 0.58:
        priority.append("prompt_align_full_v1")
        priority.append("prompt_align_context_plus_v1")
        priority.append("prompt_align_context_plus_no_rename_v1")
        priority.append("context_plus_constant_guard_v1")
        priority.append("context_plus_constant_strict_json_v1")
        priority.append("context_plus_strict_type_guard_v1")
        priority.append("context_plus_strict_behavior_nudge_v1")
        priority.append("context_plus_strict_long500_v1")
        priority.append("prompt_align_full_no_rename_v1")
    if low_json:
        priority.append("completion_256_contract_v1")
        priority.append("invalid_scope_guard_v1")
        priority.append("invalid_scope_guard_v2")
        priority.append("invalid_scope_truncation_v1")
        priority.append("invalid_scope_truncation_v2")
        priority.append("completion320_scope_v1")
        priority.append("completion288_scope_v1")
        priority.append("signature_contract_scope_v1")
        priority.append("format_scope_bias_v1")
        priority.append("sampling_scope_balance_v1")
        priority.append("sampling_scope_balance_v2")
    if compile_rate < 0.66 or behavior_rate < 0.54:
        priority.append("safety_signature_rebalance_v1")
        priority.append("behavior_heavy_scope_v1")
        priority.append("long_horizon_scope_v1")
    if reward_plateau or behavior_rate < 0.56:
        priority.append("long_horizon_stable_v1")
        priority.append("exploration_boost_v1")
        priority.append("exploration_safety_mix_v1")
    if compile_rate < 0.66 or json_valid_rate < 0.94:
        priority.append("batch2_completion256_v1")
    priority.append("long_horizon_completion256_v1")
    priority.extend(
        candidate.experiment_id
        for candidate in candidates
        if candidate.experiment_id not in priority
    )

    for experiment_id in priority:
        if experiment_id in prior_experiment_ids:
            continue
        for candidate in candidates:
            if candidate.experiment_id == experiment_id:
                return candidate
    return None


def build_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(f"decomp_clarifier.grpo_campaign.{log_path.stem}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def _latest_completed_sft_checkpoint(root: Path) -> Path:
    runs_dir = root / "artifacts" / "runs"
    candidates = sorted(
        runs_dir.glob("train-sft-*/model"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        manifest = path / "sft_training_manifest.json"
        if manifest.exists():
            return path
    raise CampaignError("no completed SFT checkpoint was found under artifacts/runs")


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(payload), sort_keys=False), encoding="utf-8")


def _profile_file_name(iteration: int, experiment_id: str) -> str:
    return f"{iteration:04d}-{experiment_id}.yaml"


def _profile_eval_max_new_tokens(payload: Mapping[str, Any]) -> int:
    training = payload.get("training", {})
    if not isinstance(training, Mapping):
        return 384
    value = training.get("max_completion_length")
    if isinstance(value, int) and value > 0:
        return value
    return 384


class GrpoCampaign:
    def __init__(
        self,
        *,
        root: Path,
        tag: str,
        seed_profile: str,
        base_model_id: str | None,
        logger: logging.Logger,
        max_iterations: int,
        scout_sample_limit: int,
        confirm_improvement: float,
        keep_improvement: float,
        target_improvement: float,
        eval_prompt_profile: str,
        eval_max_new_tokens: int,
        sft_profile: str,
        sft_baseline_manifest: Path | None = None,
        stop_on_target: bool = True,
        search_space: str = DEFAULT_SEARCH_SPACE,
    ) -> None:
        self.root = root
        self.tag = tag
        self.seed_profile = seed_profile
        self.logger = logger
        self.max_iterations = max_iterations
        self.scout_sample_limit = scout_sample_limit
        self.confirm_improvement = confirm_improvement
        self.keep_improvement = keep_improvement
        self.target_improvement = target_improvement
        self.eval_prompt_profile = eval_prompt_profile
        self.eval_max_new_tokens = eval_max_new_tokens
        self.sft_profile = sft_profile
        self.sft_baseline_manifest = sft_baseline_manifest
        self.stop_on_target = stop_on_target
        self.search_space = search_space
        self.campaign_dir = root / "research" / "campaigns" / tag
        self.profile_dir = self.campaign_dir / "profiles"
        self.dataset_dir = self.campaign_dir / "datasets"
        self.log_path = self.campaign_dir / "experiment_log.jsonl"
        self.champion_path = self.campaign_dir / "champion.json"
        self.sft_baseline_path = self.campaign_dir / "sft_baseline.json"
        self.manifest_path = self.campaign_dir / "campaign_manifest.json"
        self.base_profile_payload = _load_yaml(self._resolve_profile_path(seed_profile))
        self.base_model_id = (
            base_model_id if base_model_id else str(_latest_completed_sft_checkpoint(root))
        )
        if self.eval_prompt_profile not in {"stage", "compact", "full"}:
            raise CampaignError("eval_prompt_profile must be one of: stage, compact, full")
        if self.eval_max_new_tokens <= 0:
            raise CampaignError("eval_max_new_tokens must be positive")

    def _resolve_profile_path(self, profile: str) -> Path:
        candidate = Path(profile)
        if candidate.is_absolute() or any(separator in profile for separator in ("/", "\\")):
            return candidate if candidate.is_absolute() else self.root / candidate
        return self.root / "configs" / "training" / f"{profile}.yaml"

    def _profile_snapshot(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        training = dict(payload.get("training", {}))
        return {
            "max_steps": training.get("max_steps"),
            "batch_size": training.get("batch_size"),
            "grad_accum_steps": training.get("grad_accum_steps"),
            "max_prompt_length": training.get("max_prompt_length"),
            "max_completion_length": training.get("max_completion_length"),
            "learning_rate": training.get("learning_rate"),
            "warmup_ratio": training.get("warmup_ratio"),
            "rollout_temperature": training.get("rollout_temperature"),
            "rollout_top_p": training.get("rollout_top_p"),
            "rollout_top_k": training.get("rollout_top_k"),
            "rollout_min_p": training.get("rollout_min_p"),
            "rollout_repetition_penalty": training.get("rollout_repetition_penalty"),
            "reward_weights": training.get("reward_weights", {}),
        }

    def _write_campaign_manifest(self) -> None:
        payload = {
            "tag": self.tag,
            "seed_profile": self.seed_profile,
            "base_model_id": self.base_model_id,
            "sft_profile": self.sft_profile,
            "target_improvement": self.target_improvement,
            "eval_prompt_profile": self.eval_prompt_profile,
            "eval_max_new_tokens": self.eval_max_new_tokens,
            "scout_sample_limit": self.scout_sample_limit,
            "confirm_improvement": self.confirm_improvement,
            "keep_improvement": self.keep_improvement,
            "search_space": self.search_space,
            "stop_on_target": self.stop_on_target,
            "paths": {
                "campaign_dir": str(self.campaign_dir),
                "profiles": str(self.profile_dir),
                "datasets": str(self.dataset_dir),
                "experiment_log": str(self.log_path),
                "champion": str(self.champion_path),
                "sft_baseline": str(self.sft_baseline_path),
            },
        }
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _load_or_eval_sft_baseline(self) -> dict[str, Any]:
        if self.sft_baseline_path.exists():
            return _load_json(self.sft_baseline_path)
        if self.sft_baseline_manifest is not None:
            manifest = _load_json(self.sft_baseline_manifest)
            metrics = _metrics_from_eval_manifest(manifest)
            payload = {
                "timestamp": datetime.now().astimezone().isoformat(),
                "source_manifest": str(self.sft_baseline_manifest),
                "eval_run_id": Path(str(self.sft_baseline_manifest)).parent.name,
                "score": score_metrics(metrics),
                "metrics": metrics,
            }
            self.sft_baseline_path.parent.mkdir(parents=True, exist_ok=True)
            self.sft_baseline_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            return payload
        self.logger.info(
            "evaluating sft baseline profile=%s split=val prompt_profile=%s max_new_tokens=%s",
            self.sft_profile,
            self.eval_prompt_profile,
            self.eval_max_new_tokens,
        )
        eval_dir = _new_run_from_command(
            self.root,
            "eval-sft-checkpoint",
            [
                sys.executable,
                "-m",
                "decomp_clarifier.cli",
                "eval-sft-checkpoint",
                "--training-profile",
                self.sft_profile,
                "--split",
                "val",
                "--prompt-profile",
                self.eval_prompt_profile,
                "--max-new-tokens",
                str(self.eval_max_new_tokens),
                "--no-thinking",
            ],
            self.logger,
        )
        manifest = _load_json(eval_dir / "checkpoint_eval_manifest.json")
        metrics = _metrics_from_eval_manifest(manifest)
        payload = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "eval_run_id": eval_dir.name,
            "score": score_metrics(metrics),
            "metrics": metrics,
            "manifest_path": str(eval_dir / "checkpoint_eval_manifest.json"),
        }
        self.sft_baseline_path.parent.mkdir(parents=True, exist_ok=True)
        self.sft_baseline_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return payload

    def _dataset_path_for_experiment(
        self,
        *,
        iteration: int,
        experiment: CampaignExperiment,
    ) -> tuple[Path, dict[str, Any]]:
        if not experiment.dataset_overrides:
            return self.root / "data" / "processed" / "rl" / "rl_records.jsonl", {}
        prompt_mode = str(experiment.dataset_overrides.get("prompt_mode", "compact"))
        raw_task_types = experiment.dataset_overrides.get("task_types", [])
        task_types = (
            [str(value) for value in raw_task_types]
            if isinstance(raw_task_types, list)
            else []
        )
        raw_limit = experiment.dataset_overrides.get("prompt_limit")
        prompt_limit = raw_limit if isinstance(raw_limit, int) else None
        samples = _load_function_samples(
            self.root / "data" / "processed" / "sft" / "function_dataset.jsonl",
            split="train",
        )
        records = pack_campaign_rl_records(
            samples,
            prompt_mode=prompt_mode,
            task_types=task_types,
            prompt_limit=prompt_limit,
        )
        if not records:
            raise CampaignError(
                f"campaign dataset for experiment={experiment.experiment_id!r} is empty"
            )
        dataset_path = self.dataset_dir / f"{iteration:04d}-{experiment.experiment_id}.jsonl"
        _write_jsonl(dataset_path, records)
        task_counts: Counter[str] = Counter(record.task_type for record in records)
        metadata = {
            "path": str(dataset_path),
            "record_count": len(records),
            "prompt_mode": prompt_mode,
            "task_types": task_types,
            "prompt_limit": prompt_limit,
            "task_counts": dict(sorted(task_counts.items())),
        }
        (dataset_path.with_suffix(".manifest.json")).write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return dataset_path, metadata

    def _load_entries(self) -> list[dict[str, Any]]:
        return _read_jsonl(self.log_path)

    def _eval_manifest_candidates(
        self,
        *,
        profile_path: Path,
        checkpoint_dir: Path | None = None,
        scout: bool | None = None,
    ) -> list[tuple[Path, dict[str, Any]]]:
        candidates: list[tuple[Path, dict[str, Any]]] = []
        normalized_checkpoint = (
            _normalized_path_ref(checkpoint_dir) if checkpoint_dir is not None else None
        )
        for eval_dir in _run_dirs(self.root, "eval-grpo-checkpoint"):
            manifest_path = eval_dir / "checkpoint_eval_manifest.json"
            if not manifest_path.exists():
                continue
            try:
                manifest = _load_json(manifest_path)
            except (json.JSONDecodeError, OSError):
                continue
            if not _profile_ref_matches(
                manifest_ref=manifest.get("training_profile", ""),
                root=self.root,
                profile_path=profile_path,
            ):
                continue
            if normalized_checkpoint is not None and _normalized_path_ref(
                manifest.get("checkpoint_dir", "")
            ) != normalized_checkpoint:
                continue
            if manifest.get("split") != "val":
                continue
            if manifest.get("prompt_profile") != self.eval_prompt_profile:
                continue
            if int(manifest.get("max_new_tokens", 0)) != self.eval_max_new_tokens:
                continue
            sample_count = int(manifest.get("sample_count", 0))
            if scout is True and sample_count != self.scout_sample_limit:
                continue
            if scout is False and sample_count <= self.scout_sample_limit:
                continue
            candidates.append((eval_dir, manifest))
        return sorted(candidates, key=lambda item: item[0].stat().st_mtime)

    def _append_iteration_record(
        self,
        *,
        iteration: int,
        experiment: CampaignExperiment,
        status: str,
        profile_path: Path,
        candidate_profile_payload: Mapping[str, Any],
        candidate_dataset_path: Path | None,
        candidate_dataset_metadata: Mapping[str, Any],
        train_dir: Path | None,
        scout_dir: Path | None,
        confirm_dir: Path | None,
        scout_score: float | None,
        confirm_score: float | None,
        sft_score: float,
        scout_metrics: Mapping[str, float],
        confirm_metrics: Mapping[str, float],
        notes: str,
    ) -> None:
        _append_jsonl(
            self.log_path,
            {
                "iteration": iteration,
                "timestamp": datetime.now().astimezone().isoformat(),
                "status": status,
                "experiment_id": experiment.experiment_id,
                "hypothesis": experiment.hypothesis,
                "profile_path": str(profile_path),
                "dataset_path": str(candidate_dataset_path) if candidate_dataset_path else None,
                "dataset": dict(candidate_dataset_metadata),
                "train_run_id": train_dir.name if train_dir else None,
                "scout_eval_run_id": scout_dir.name if scout_dir else None,
                "confirm_eval_run_id": confirm_dir.name if confirm_dir else None,
                "scout_score": scout_score,
                "confirm_score": confirm_score,
                "sft_score": sft_score,
                "target_improvement": self.target_improvement,
                "metrics": (
                    dict(confirm_metrics)
                    if confirm_dir is not None
                    else dict(scout_metrics)
                ),
                "config_snapshot": self._profile_snapshot(candidate_profile_payload),
                "notes": notes,
            },
        )

    def _recover_completed_iteration(
        self,
        *,
        experiment: CampaignExperiment,
        iteration: int,
        champion: Mapping[str, Any],
        champion_metrics: Mapping[str, float],
        champion_scout_metrics: Mapping[str, float],
        sft_metrics: Mapping[str, float],
        sft_score: float,
    ) -> bool:
        profile_path = self.profile_dir / _profile_file_name(
            iteration,
            experiment.experiment_id,
        )
        if not profile_path.exists():
            return False
        scout_candidates = self._eval_manifest_candidates(
            profile_path=profile_path,
            scout=True,
        )
        if not scout_candidates:
            return False
        scout_dir, scout_manifest = scout_candidates[-1]
        scout_metrics = _metrics_from_eval_manifest(scout_manifest)
        scout_score = score_metrics(scout_metrics)
        checkpoint_ref = scout_manifest.get("checkpoint_dir")
        checkpoint_dir = Path(str(checkpoint_ref)) if checkpoint_ref else None
        train_dir = checkpoint_dir.parent if checkpoint_dir is not None else None
        candidate_profile_payload = _load_yaml(profile_path)
        (
            candidate_dataset_path,
            candidate_dataset_metadata,
        ) = self._dataset_path_for_experiment(
            iteration=iteration,
            experiment=experiment,
        )

        status = "discard"
        notes = "Recovered completed scout evaluation after an interrupted campaign process."
        confirm_dir: Path | None = None
        confirm_score: float | None = None
        confirm_metrics: dict[str, float] = dict.fromkeys(_SCORE_KEYS, 0.0)
        champion_score = float(champion.get("score", 0.0))
        champion_scout_score = float(champion.get("scout_score", champion_score))

        if not _hard_keep_gates_pass(scout_metrics, champion_scout_metrics):
            notes += " Scout evaluation failed the scout hard keep gates against the champion."
        elif scout_score < champion_scout_score + self.keep_improvement:
            notes += " Scout score did not clear the scout promotion threshold."
        else:
            confirm_candidates = self._eval_manifest_candidates(
                profile_path=profile_path,
                checkpoint_dir=checkpoint_dir,
                scout=False,
            )
            if not confirm_candidates:
                if train_dir is None:
                    notes += " Could not locate a checkpoint for full validation."
                else:
                    confirm_dir = _new_run_from_command(
                        self.root,
                        "eval-grpo-checkpoint",
                        [
                            sys.executable,
                            "-m",
                            "decomp_clarifier.cli",
                            "eval-grpo-checkpoint",
                            "--checkpoint-dir",
                            str(train_dir / "model"),
                            "--training-profile",
                            str(profile_path.relative_to(self.root)),
                            "--split",
                            "val",
                            "--max-new-tokens",
                            str(self.eval_max_new_tokens),
                            "--prompt-profile",
                            self.eval_prompt_profile,
                            "--no-thinking",
                        ],
                        self.logger,
                    )
                    confirm_manifest = _load_json(
                        confirm_dir / "checkpoint_eval_manifest.json"
                    )
                    confirm_metrics = _metrics_from_eval_manifest(confirm_manifest)
            else:
                confirm_dir, confirm_manifest = confirm_candidates[-1]
                confirm_metrics = _metrics_from_eval_manifest(confirm_manifest)

            if confirm_dir is not None:
                confirm_score = score_metrics(confirm_metrics)
                target_passed = sft_target_passed(
                    confirm_metrics,
                    candidate_score=confirm_score,
                    sft_metrics=sft_metrics,
                    sft_score=sft_score,
                    target_improvement=self.target_improvement,
                )
                if (
                    _hard_keep_gates_pass(confirm_metrics, champion_metrics)
                    and (
                        confirm_score >= champion_score + self.confirm_improvement
                        or target_passed
                    )
                ):
                    status = "target_keep" if target_passed else "keep"
                    notes += (
                        " Full validation reached the SFT target and cleared all hard gates."
                        if target_passed
                        else " Full validation beat the champion and cleared all hard gates."
                    )
                    self._write_champion(
                        profile_path=profile_path,
                        scout_metrics=scout_metrics,
                        metrics=confirm_metrics,
                        scout_score=scout_score,
                        score=confirm_score,
                        train_run_id=train_dir.name if train_dir else "",
                        eval_run_id=confirm_dir.name,
                        experiment_id=experiment.experiment_id,
                        hypothesis=experiment.hypothesis,
                        config_snapshot=self._profile_snapshot(candidate_profile_payload),
                    )
                else:
                    notes += " Full validation failed the final keep threshold or a hard gate."

        self._append_iteration_record(
            iteration=iteration,
            experiment=experiment,
            status=status,
            profile_path=profile_path,
            candidate_profile_payload=candidate_profile_payload,
            candidate_dataset_path=candidate_dataset_path,
            candidate_dataset_metadata=candidate_dataset_metadata,
            train_dir=train_dir,
            scout_dir=scout_dir,
            confirm_dir=confirm_dir,
            scout_score=scout_score,
            confirm_score=confirm_score,
            sft_score=sft_score,
            scout_metrics=scout_metrics,
            confirm_metrics=confirm_metrics,
            notes=notes,
        )
        self.logger.info(
            "recovered iteration=%s experiment=%s status=%s scout=%s confirm=%s",
            iteration,
            experiment.experiment_id,
            status,
            f"{scout_score:.4f}",
            f"{confirm_score:.4f}" if confirm_score is not None else "n/a",
        )
        return True

    def _write_champion(
        self,
        *,
        profile_path: Path,
        scout_metrics: Mapping[str, float],
        metrics: Mapping[str, float],
        scout_score: float,
        score: float,
        train_run_id: str,
        eval_run_id: str,
        experiment_id: str,
        hypothesis: str,
        config_snapshot: Mapping[str, Any],
    ) -> None:
        sft_baseline = self._load_or_eval_sft_baseline()
        payload = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "profile_path": str(profile_path),
            "base_model_id": self.base_model_id,
            "sft_baseline": {
                "score": float(sft_baseline.get("score", 0.0)),
                "metrics": {
                    key: float(sft_baseline.get("metrics", {}).get(key, 0.0))
                    for key in _SCORE_KEYS
                },
                "eval_run_id": sft_baseline.get("eval_run_id"),
                "target_improvement": self.target_improvement,
            },
            "scout_score": scout_score,
            "score": score,
            "train_run_id": train_run_id,
            "eval_run_id": eval_run_id,
            "experiment_id": experiment_id,
            "hypothesis": hypothesis,
            "config_snapshot": dict(config_snapshot),
            "scout_metrics": {
                key: float(scout_metrics.get(key, 0.0)) for key in _SCORE_KEYS
            },
            "metrics": {key: float(metrics.get(key, 0.0)) for key in _SCORE_KEYS},
        }
        self.champion_path.parent.mkdir(parents=True, exist_ok=True)
        self.champion_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _bootstrap_baseline(self) -> None:
        self._write_campaign_manifest()
        self._load_or_eval_sft_baseline()
        if self.champion_path.exists():
            return
        self.logger.info(
            "bootstrapping baseline from seed_profile=%s base_model=%s",
            self.seed_profile,
            self.base_model_id,
        )
        profile_path = self.profile_dir / _profile_file_name(0, "baseline")
        _write_yaml(profile_path, self.base_profile_payload)
        train_dir = _new_run_from_command(
            self.root,
            "train-grpo",
            [
                sys.executable,
                "-m",
                "decomp_clarifier.cli",
                "train-grpo",
                "--training-profile",
                str(profile_path.relative_to(self.root)),
                "--base-model-id",
                self.base_model_id,
            ],
            self.logger,
        )
        scout_dir = _new_run_from_command(
            self.root,
            "eval-grpo-checkpoint",
            [
                sys.executable,
                "-m",
                "decomp_clarifier.cli",
                "eval-grpo-checkpoint",
                "--checkpoint-dir",
                str(train_dir / "model"),
                "--training-profile",
                str(profile_path.relative_to(self.root)),
                "--split",
                "val",
                "--sample-limit",
                str(self.scout_sample_limit),
                "--max-new-tokens",
                str(self.eval_max_new_tokens),
                "--prompt-profile",
                self.eval_prompt_profile,
                "--no-thinking",
            ],
            self.logger,
        )
        confirm_dir = _new_run_from_command(
            self.root,
            "eval-grpo-checkpoint",
            [
                sys.executable,
                "-m",
                "decomp_clarifier.cli",
                "eval-grpo-checkpoint",
                "--checkpoint-dir",
                str(train_dir / "model"),
                "--training-profile",
                str(profile_path.relative_to(self.root)),
                "--split",
                "val",
                "--max-new-tokens",
                str(self.eval_max_new_tokens),
                "--prompt-profile",
                self.eval_prompt_profile,
                "--no-thinking",
            ],
            self.logger,
        )
        scout_metrics = _metrics_from_eval_manifest(
            _load_json(scout_dir / "checkpoint_eval_manifest.json")
        )
        scout_score = score_metrics(scout_metrics)
        metrics = _metrics_from_eval_manifest(
            _load_json(confirm_dir / "checkpoint_eval_manifest.json")
        )
        score = score_metrics(metrics)
        sft_baseline = self._load_or_eval_sft_baseline()
        self._write_champion(
            profile_path=profile_path,
            scout_metrics=scout_metrics,
            metrics=metrics,
            scout_score=scout_score,
            score=score,
            train_run_id=train_dir.name,
            eval_run_id=confirm_dir.name,
            experiment_id="baseline",
            hypothesis="Baseline guarded pilot from the current Qwen 2B SFT checkpoint.",
            config_snapshot=self._profile_snapshot(self.base_profile_payload),
        )
        _append_jsonl(
            self.log_path,
            {
                "iteration": 0,
                "timestamp": datetime.now().astimezone().isoformat(),
                "status": "baseline",
                "experiment_id": "baseline",
                "hypothesis": "Baseline guarded pilot from the current Qwen 2B SFT checkpoint.",
                "profile_path": str(profile_path),
                "train_run_id": train_dir.name,
                "scout_eval_run_id": scout_dir.name,
                "confirm_eval_run_id": confirm_dir.name,
                "scout_score": scout_score,
                "confirm_score": score,
                "sft_score": float(sft_baseline.get("score", 0.0)),
                "target_improvement": self.target_improvement,
                "metrics": metrics,
                "config_snapshot": self._profile_snapshot(self.base_profile_payload),
                "notes": "Campaign baseline established from the seed profile.",
            },
        )

    def _current_champion(self) -> dict[str, Any]:
        return _load_json(self.champion_path)

    def _write_candidate_profile(
        self,
        *,
        iteration: int,
        experiment: CampaignExperiment,
        champion_profile_payload: Mapping[str, Any],
    ) -> Path:
        payload = apply_training_overrides(champion_profile_payload, experiment.overrides)
        profile_path = self.profile_dir / _profile_file_name(iteration, experiment.experiment_id)
        _write_yaml(profile_path, payload)
        return profile_path

    def run(self) -> None:
        dataset_path = self.root / "data" / "processed" / "rl" / "rl_records.jsonl"
        function_dataset_path = self.root / "data" / "processed" / "sft" / "function_dataset.jsonl"
        if sys.platform != "win32":
            raise CampaignError("GRPO campaign requires Windows")
        if not dataset_path.exists():
            raise CampaignError("missing RL dataset at data/processed/rl/rl_records.jsonl")
        if not function_dataset_path.exists():
            raise CampaignError(
                "missing function dataset at data/processed/sft/function_dataset.jsonl"
            )
        load_dotenv(self.root)
        self._bootstrap_baseline()

        while True:
            entries = self._load_entries()
            completed_iterations = sum(1 for entry in entries if entry.get("status") != "baseline")
            if completed_iterations >= self.max_iterations:
                self.logger.info("reached max_iterations=%s", self.max_iterations)
                return
            champion = self._current_champion()
            champion_metrics = {
                key: float(champion.get("metrics", {}).get(key, 0.0))
                for key in _SCORE_KEYS
            }
            champion_scout_metrics = {
                key: float(
                    champion.get("scout_metrics", {}).get(
                        key,
                        champion_metrics.get(key, 0.0),
                    )
                )
                for key in _SCORE_KEYS
            }
            sft_baseline = self._load_or_eval_sft_baseline()
            sft_metrics = {
                key: float(sft_baseline.get("metrics", {}).get(key, 0.0))
                for key in _SCORE_KEYS
            }
            sft_score = float(sft_baseline.get("score", 0.0))
            champion_score = float(champion.get("score", 0.0))
            if self.stop_on_target and sft_target_passed(
                champion_metrics,
                candidate_score=champion_score,
                sft_metrics=sft_metrics,
                sft_score=sft_score,
                target_improvement=self.target_improvement,
            ):
                self.logger.info(
                    "campaign target reached champion_score=%.4f sft_score=%.4f target_delta=%.4f",
                    champion_score,
                    sft_score,
                    self.target_improvement,
                )
                return
            latest_eval_dir = (
                self.root / "artifacts" / "runs" / champion["eval_run_id"]
                if champion.get("eval_run_id")
                else None
            )
            invalid_reasons = _invalid_reasons_from_eval_dir(latest_eval_dir)
            latest_train_dir = (
                self.root / "artifacts" / "runs" / champion["train_run_id"]
                if champion.get("train_run_id")
                else None
            )
            reward_telemetry = (
                _reward_telemetry_from_train_dir(latest_train_dir) if latest_train_dir else None
            )
            experiment = choose_campaign_experiment(
                champion_metrics,
                invalid_reasons,
                {str(entry.get("experiment_id", "")) for entry in entries},
                reward_telemetry=reward_telemetry,
                search_space=self.search_space,
            )
            if experiment is None:
                self.logger.info("no remaining hypotheses; stopping campaign")
                return
            iteration = max(int(entry.get("iteration", 0)) for entry in entries) + 1
            if self._recover_completed_iteration(
                experiment=experiment,
                iteration=iteration,
                champion=champion,
                champion_metrics=champion_metrics,
                champion_scout_metrics=champion_scout_metrics,
                sft_metrics=sft_metrics,
                sft_score=sft_score,
            ):
                continue
            champion_profile_path = Path(str(champion["profile_path"]))
            champion_profile_payload = _load_yaml(champion_profile_path)
            profile_path = self._write_candidate_profile(
                iteration=iteration,
                experiment=experiment,
                champion_profile_payload=champion_profile_payload,
            )
            candidate_profile_payload = _load_yaml(profile_path)
            self.logger.info(
                "iteration=%s experiment=%s hypothesis=%s",
                iteration,
                experiment.experiment_id,
                experiment.hypothesis,
            )

            status = "discard"
            notes = ""
            train_dir: Path | None = None
            scout_dir: Path | None = None
            confirm_dir: Path | None = None
            candidate_dataset_path: Path | None = None
            candidate_dataset_metadata: dict[str, Any] = {}
            scout_score: float | None = None
            confirm_score: float | None = None
            scout_metrics: dict[str, float] = dict.fromkeys(_SCORE_KEYS, 0.0)
            confirm_metrics: dict[str, float] = dict.fromkeys(_SCORE_KEYS, 0.0)
            try:
                (
                    candidate_dataset_path,
                    candidate_dataset_metadata,
                ) = self._dataset_path_for_experiment(
                    iteration=iteration,
                    experiment=experiment,
                )
                train_dir = _new_run_from_command(
                    self.root,
                    "train-grpo",
                    [
                        sys.executable,
                        "-m",
                        "decomp_clarifier.cli",
                        "train-grpo",
                        "--training-profile",
                        str(profile_path.relative_to(self.root)),
                        "--base-model-id",
                        self.base_model_id,
                        "--dataset-path",
                        str(candidate_dataset_path.relative_to(self.root)),
                    ],
                    self.logger,
                )
                scout_dir = _new_run_from_command(
                    self.root,
                    "eval-grpo-checkpoint",
                    [
                        sys.executable,
                        "-m",
                        "decomp_clarifier.cli",
                        "eval-grpo-checkpoint",
                        "--checkpoint-dir",
                        str(train_dir / "model"),
                        "--training-profile",
                        str(profile_path.relative_to(self.root)),
                        "--split",
                        "val",
                        "--sample-limit",
                        str(self.scout_sample_limit),
                        "--max-new-tokens",
                        str(self.eval_max_new_tokens),
                        "--prompt-profile",
                        self.eval_prompt_profile,
                        "--no-thinking",
                    ],
                    self.logger,
                )
                scout_metrics = _metrics_from_eval_manifest(
                    _load_json(scout_dir / "checkpoint_eval_manifest.json")
                )
                scout_score = score_metrics(scout_metrics)
                champion_score = float(champion.get("score", 0.0))
                champion_scout_score = float(champion.get("scout_score", champion_score))
                if not _hard_keep_gates_pass(scout_metrics, champion_scout_metrics):
                    status = "discard"
                    notes = (
                        "Scout evaluation failed the scout hard keep gates against the champion."
                    )
                elif scout_score < champion_scout_score + self.keep_improvement:
                    status = "discard"
                    notes = "Scout score did not clear the scout promotion threshold."
                else:
                    confirm_dir = _new_run_from_command(
                        self.root,
                        "eval-grpo-checkpoint",
                        [
                            sys.executable,
                            "-m",
                            "decomp_clarifier.cli",
                            "eval-grpo-checkpoint",
                            "--checkpoint-dir",
                            str(train_dir / "model"),
                            "--training-profile",
                            str(profile_path.relative_to(self.root)),
                            "--split",
                            "val",
                            "--max-new-tokens",
                            str(self.eval_max_new_tokens),
                            "--prompt-profile",
                            self.eval_prompt_profile,
                            "--no-thinking",
                        ],
                        self.logger,
                    )
                    confirm_metrics = _metrics_from_eval_manifest(
                        _load_json(confirm_dir / "checkpoint_eval_manifest.json")
                    )
                    confirm_score = score_metrics(confirm_metrics)
                    target_passed = sft_target_passed(
                        confirm_metrics,
                        candidate_score=confirm_score,
                        sft_metrics=sft_metrics,
                        sft_score=sft_score,
                        target_improvement=self.target_improvement,
                    )
                    if (
                        _hard_keep_gates_pass(confirm_metrics, champion_metrics)
                        and (
                            confirm_score >= champion_score + self.confirm_improvement
                            or target_passed
                        )
                    ):
                        status = "target_keep" if target_passed else "keep"
                        notes = (
                            "Full validation reached the SFT target and cleared all hard gates."
                            if target_passed
                            else "Full validation beat the champion and cleared all hard gates."
                        )
                        self._write_champion(
                            profile_path=profile_path,
                            scout_metrics=scout_metrics,
                            metrics=confirm_metrics,
                            scout_score=scout_score,
                            score=confirm_score,
                            train_run_id=train_dir.name,
                            eval_run_id=confirm_dir.name,
                            experiment_id=experiment.experiment_id,
                            hypothesis=experiment.hypothesis,
                            config_snapshot=self._profile_snapshot(candidate_profile_payload),
                        )
                    else:
                        status = "discard"
                        notes = "Full validation failed the final keep threshold or a hard gate."
            except Exception as exc:  # noqa: BLE001
                status = "crash"
                notes = str(exc)
                self.logger.exception("iteration crashed: %s", exc)

            self._append_iteration_record(
                iteration=iteration,
                experiment=experiment,
                status=status,
                profile_path=profile_path,
                candidate_profile_payload=candidate_profile_payload,
                candidate_dataset_path=candidate_dataset_path,
                candidate_dataset_metadata=candidate_dataset_metadata,
                train_dir=train_dir,
                scout_dir=scout_dir,
                confirm_dir=confirm_dir,
                scout_score=scout_score,
                confirm_score=confirm_score,
                sft_score=sft_score,
                scout_metrics=scout_metrics,
                confirm_metrics=confirm_metrics,
                notes=notes,
            )
            self.logger.info(
                "iteration=%s experiment=%s status=%s scout=%s confirm=%s",
                iteration,
                experiment.experiment_id,
                status,
                f"{scout_score:.4f}" if scout_score is not None else "n/a",
                f"{confirm_score:.4f}" if confirm_score is not None else "n/a",
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a Qwen-only GRPO campaign loop.")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--seed-profile", type=str, default=DEFAULT_SEED_PROFILE)
    parser.add_argument("--base-model-id", type=str, default=None)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--scout-sample-limit", type=int, default=DEFAULT_SCOUT_SAMPLE_LIMIT)
    parser.add_argument("--confirm-improvement", type=float, default=DEFAULT_CONFIRM_IMPROVEMENT)
    parser.add_argument("--keep-improvement", type=float, default=DEFAULT_KEEP_IMPROVEMENT)
    parser.add_argument("--target-improvement", type=float, default=DEFAULT_TARGET_IMPROVEMENT)
    parser.add_argument("--sft-profile", type=str, default="sft_qwen35_2b")
    parser.add_argument("--sft-baseline-manifest", type=Path, default=None)
    parser.add_argument(
        "--eval-prompt-profile",
        choices=("stage", "compact", "full"),
        default=DEFAULT_EVAL_PROMPT_PROFILE,
    )
    parser.add_argument("--eval-max-new-tokens", type=int, default=DEFAULT_EVAL_MAX_NEW_TOKENS)
    parser.add_argument("--no-stop-on-target", action="store_true")
    parser.add_argument(
        "--search-space",
        choices=("default", "long300", "post_target"),
        default=DEFAULT_SEARCH_SPACE,
    )
    parser.add_argument("--log-file", type=Path, default=None)
    args = parser.parse_args(argv)

    root = _repo_root(args.root)
    tag = args.tag or f"qwen-grpo-{tag_for_now(datetime.now().astimezone())}"
    log_file = args.log_file or (root / "artifacts" / "logs" / f"{tag}.log")
    logger = build_logger(log_file)
    try:
        campaign = GrpoCampaign(
            root=root,
            tag=tag,
            seed_profile=args.seed_profile,
            base_model_id=args.base_model_id,
            logger=logger,
            max_iterations=args.max_iterations,
            scout_sample_limit=args.scout_sample_limit,
            confirm_improvement=args.confirm_improvement,
            keep_improvement=args.keep_improvement,
            target_improvement=args.target_improvement,
            eval_prompt_profile=args.eval_prompt_profile,
            eval_max_new_tokens=args.eval_max_new_tokens,
            sft_profile=args.sft_profile,
            sft_baseline_manifest=args.sft_baseline_manifest,
            stop_on_target=not args.no_stop_on_target,
            search_space=args.search_space,
        )
        campaign.run()
    except CampaignError as exc:
        logger.error("campaign stopped: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
