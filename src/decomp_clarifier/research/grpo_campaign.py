from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from decomp_clarifier.paths import ProjectPaths
from decomp_clarifier.research.autoresearch import (
    RewardTelemetrySnapshot,
    classify_invalid_prediction_rows,
    score_metrics,
)
from decomp_clarifier.settings import load_dotenv

DEFAULT_SEED_PROFILE = "configs/training/grpo_qwen35_2b_guarded_pilot.yaml"
DEFAULT_SCOUT_SAMPLE_LIMIT = 50
DEFAULT_KEEP_IMPROVEMENT = 0.005
DEFAULT_CONFIRM_IMPROVEMENT = 0.01
DEFAULT_SEARCH_SPACE = "default"
_SCORE_KEYS = (
    "json_valid_rate",
    "field_complete_rate",
    "readability_score",
    "compile_success_rate",
    "behavior_success_rate",
    "naming_score",
)


class CampaignError(RuntimeError):
    """Raised when the GRPO campaign cannot continue safely."""


@dataclass(frozen=True)
class CampaignExperiment:
    experiment_id: str
    hypothesis: str
    short_description: str
    overrides: dict[str, Any]


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
                "format and signature bias may improve the JSON-valid tail without shrinking budget."
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


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _metrics_from_eval_manifest(manifest: Mapping[str, Any]) -> dict[str, float]:
    metrics = manifest.get("metrics", {})
    return {key: float(metrics.get(key, 0.0)) for key in _SCORE_KEYS}


def _hard_keep_gates_pass(metrics: Mapping[str, float], champion: Mapping[str, float]) -> bool:
    return (
        float(metrics.get("compile_success_rate", 0.0))
        >= float(champion.get("compile_success_rate", 0.0)) - 0.01
        and float(metrics.get("behavior_success_rate", 0.0))
        >= float(champion.get("behavior_success_rate", 0.0)) - 0.01
        and float(metrics.get("json_valid_rate", 0.0))
        >= float(champion.get("json_valid_rate", 0.0)) - 0.02
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


def apply_training_overrides(base_payload: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    def merge(left: Any, right: Any) -> Any:
        if isinstance(left, dict) and isinstance(right, Mapping):
            merged = dict(left)
            for key, value in right.items():
                merged[key] = merge(merged.get(key), value)
            return merged
        return right

    return merge(dict(base_payload), overrides)


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
        self.search_space = search_space
        self.campaign_dir = root / "research" / "campaigns" / tag
        self.profile_dir = self.campaign_dir / "profiles"
        self.log_path = self.campaign_dir / "experiment_log.jsonl"
        self.champion_path = self.campaign_dir / "champion.json"
        self.base_profile_payload = _load_yaml(self._resolve_profile_path(seed_profile))
        self.base_model_id = (
            base_model_id if base_model_id else str(_latest_completed_sft_checkpoint(root))
        )

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

    def _load_entries(self) -> list[dict[str, Any]]:
        return _read_jsonl(self.log_path)

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
        payload = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "profile_path": str(profile_path),
            "base_model_id": self.base_model_id,
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
        self.champion_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _bootstrap_baseline(self) -> None:
        if self.champion_path.exists():
            return
        self.logger.info(
            "bootstrapping baseline from seed_profile=%s base_model=%s",
            self.seed_profile,
            self.base_model_id,
        )
        profile_path = self.profile_dir / _profile_file_name(0, "baseline")
        _write_yaml(profile_path, self.base_profile_payload)
        eval_max_new_tokens = _profile_eval_max_new_tokens(self.base_profile_payload)
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
                str(eval_max_new_tokens),
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
                str(eval_max_new_tokens),
            ],
            self.logger,
        )
        scout_metrics = _metrics_from_eval_manifest(_load_json(scout_dir / "checkpoint_eval_manifest.json"))
        scout_score = score_metrics(scout_metrics)
        metrics = _metrics_from_eval_manifest(_load_json(confirm_dir / "checkpoint_eval_manifest.json"))
        score = score_metrics(metrics)
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
        if sys.platform != "win32":
            raise CampaignError("GRPO campaign requires Windows")
        if not dataset_path.exists():
            raise CampaignError("missing RL dataset at data/processed/rl/rl_records.jsonl")
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
                key: float(champion.get("scout_metrics", {}).get(key, champion_metrics.get(key, 0.0)))
                for key in _SCORE_KEYS
            }
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
            champion_profile_path = Path(str(champion["profile_path"]))
            champion_profile_payload = _load_yaml(champion_profile_path)
            profile_path = self._write_candidate_profile(
                iteration=iteration,
                experiment=experiment,
                champion_profile_payload=champion_profile_payload,
            )
            candidate_profile_payload = _load_yaml(profile_path)
            eval_max_new_tokens = _profile_eval_max_new_tokens(candidate_profile_payload)
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
            scout_score: float | None = None
            confirm_score: float | None = None
            scout_metrics: dict[str, float] = dict.fromkeys(_SCORE_KEYS, 0.0)
            confirm_metrics: dict[str, float] = dict.fromkeys(_SCORE_KEYS, 0.0)
            try:
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
                        str(eval_max_new_tokens),
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
                    notes = "Scout evaluation failed the scout hard keep gates against the champion."
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
                            str(eval_max_new_tokens),
                        ],
                        self.logger,
                    )
                    confirm_metrics = _metrics_from_eval_manifest(
                        _load_json(confirm_dir / "checkpoint_eval_manifest.json")
                    )
                    confirm_score = score_metrics(confirm_metrics)
                    if (
                        _hard_keep_gates_pass(confirm_metrics, champion_metrics)
                        and confirm_score >= champion_score + self.confirm_improvement
                    ):
                        status = "keep"
                        notes = "Full validation beat the champion and cleared all hard gates."
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

            _append_jsonl(
                self.log_path,
                {
                    "iteration": iteration,
                    "timestamp": datetime.now().astimezone().isoformat(),
                    "status": status,
                    "experiment_id": experiment.experiment_id,
                    "hypothesis": experiment.hypothesis,
                    "profile_path": str(profile_path),
                    "train_run_id": train_dir.name if train_dir else None,
                    "scout_eval_run_id": scout_dir.name if scout_dir else None,
                    "confirm_eval_run_id": confirm_dir.name if confirm_dir else None,
                    "scout_score": scout_score,
                    "confirm_score": confirm_score,
                    "metrics": confirm_metrics if confirm_dir is not None else scout_metrics,
                    "config_snapshot": self._profile_snapshot(candidate_profile_payload),
                    "notes": notes,
                },
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
    parser.add_argument(
        "--search-space",
        choices=("default", "long300"),
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
            search_space=args.search_space,
        )
        campaign.run()
    except CampaignError as exc:
        logger.error("campaign stopped: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
