from __future__ import annotations

from pathlib import Path

from decomp_clarifier.research.grpo_campaign import (
    apply_training_overrides,
    choose_campaign_experiment,
)
from decomp_clarifier.settings import load_training_config


def test_choose_campaign_experiment_prioritizes_completion_contract_for_json_gaps() -> None:
    choice = choose_campaign_experiment(
        {
            "json_valid_rate": 0.91,
            "compile_success_rate": 0.65,
            "behavior_success_rate": 0.52,
        },
        {"json_parse_error": 4},
        set(),
    )
    assert choice is not None
    assert choice.experiment_id == "completion_256_contract_v1"


def test_choose_campaign_experiment_moves_to_long_horizon_after_early_hypotheses() -> None:
    choice = choose_campaign_experiment(
        {
            "json_valid_rate": 0.95,
            "compile_success_rate": 0.67,
            "behavior_success_rate": 0.53,
        },
        {},
        {
            "completion_256_contract_v1",
            "safety_signature_rebalance_v1",
            "batch2_completion256_v1",
        },
    )
    assert choice is not None
    assert choice.experiment_id == "long_horizon_stable_v1"


def test_apply_training_overrides_merges_nested_reward_weights() -> None:
    payload = {
        "training": {
            "max_steps": 100,
            "reward_weights": {
                "format": 1.0,
                "compile": 3.0,
            },
        }
    }
    updated = apply_training_overrides(
        payload,
        {
            "training": {
                "max_steps": 200,
                "reward_weights": {
                    "format": 1.15,
                    "signature": 1.25,
                },
            }
        },
    )
    assert updated["training"]["max_steps"] == 200
    assert updated["training"]["reward_weights"] == {
        "format": 1.15,
        "compile": 3.0,
        "signature": 1.25,
    }


def test_load_training_config_accepts_direct_yaml_path(tmp_path: Path) -> None:
    root = tmp_path
    profile_path = root / "research" / "campaigns" / "demo" / "candidate.yaml"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(
        """
model:
  base_model_id: demo-model
training:
  max_steps: 123
  reward_weights:
    format: 1.1
""".strip(),
        encoding="utf-8",
    )

    config = load_training_config(root, str(profile_path.relative_to(root)))

    assert config.model.base_model_id == "demo-model"
    assert config.training.max_steps == 123
    assert config.training.reward_weights["format"] == 1.1
