from __future__ import annotations

from pathlib import Path

from decomp_clarifier.research.grpo_campaign import (
    _post_target_campaign_candidates,
    _profile_eval_max_new_tokens,
    _profile_ref_matches,
    apply_training_overrides,
    choose_campaign_experiment,
    pack_campaign_rl_records,
    sft_target_passed,
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


def test_choose_campaign_experiment_tries_invalid_scope_guard_after_completion_contract() -> None:
    choice = choose_campaign_experiment(
        {
            "json_valid_rate": 0.92,
            "compile_success_rate": 0.63,
            "behavior_success_rate": 0.49,
        },
        {"json_parse_error": 6},
        {"completion_256_contract_v1", "safety_signature_rebalance_v1"},
    )
    assert choice is not None
    assert choice.experiment_id == "invalid_scope_guard_v1"


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
            "behavior_heavy_scope_v1",
        },
    )
    assert choice is not None
    assert choice.experiment_id == "long_horizon_scope_v1"


def test_choose_campaign_experiment_moves_to_exploration_after_scope_horizon() -> None:
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
            "behavior_heavy_scope_v1",
            "long_horizon_scope_v1",
            "long_horizon_stable_v1",
        },
    )
    assert choice is not None
    assert choice.experiment_id == "exploration_boost_v1"


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


def test_profile_eval_max_new_tokens_uses_training_completion_length() -> None:
    assert _profile_eval_max_new_tokens({"training": {"max_completion_length": 256}}) == 256
    assert _profile_eval_max_new_tokens({"training": {}}) == 384


def test_profile_ref_matches_relative_and_absolute_paths(tmp_path: Path) -> None:
    root = tmp_path
    profile_path = root / "research" / "campaigns" / "demo" / "profiles" / "0001-demo.yaml"
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text("training: {}\n", encoding="utf-8")

    assert _profile_ref_matches(
        manifest_ref="research\\campaigns\\demo\\profiles\\0001-demo.yaml",
        root=root,
        profile_path=profile_path,
    )
    assert _profile_ref_matches(
        manifest_ref=str(profile_path),
        root=root,
        profile_path=profile_path,
    )
    assert not _profile_ref_matches(
        manifest_ref="research/campaigns/demo/profiles/0002-other.yaml",
        root=root,
        profile_path=profile_path,
    )


def test_choose_campaign_experiment_long300_returns_first_untried_candidate() -> None:
    choice = choose_campaign_experiment(
        {
            "json_valid_rate": 0.93,
            "compile_success_rate": 0.64,
            "behavior_success_rate": 0.51,
        },
        {"json_parse_error": 7},
        set(),
        search_space="long300",
    )
    assert choice is not None
    assert choice.experiment_id == "long300_lr_low_v1"


def test_choose_campaign_experiment_long300_skips_completed_candidates() -> None:
    choice = choose_campaign_experiment(
        {
            "json_valid_rate": 0.93,
            "compile_success_rate": 0.64,
            "behavior_success_rate": 0.51,
        },
        {"json_parse_error": 7},
        {"long300_lr_low_v1", "long300_lr_high_v1"},
        search_space="long300",
    )
    assert choice is not None
    assert choice.experiment_id == "long300_gradaccum2_v1"


def test_choose_campaign_experiment_post_target_returns_first_untried_candidate() -> None:
    choice = choose_campaign_experiment(
        {
            "json_valid_rate": 1.0,
            "compile_success_rate": 0.65,
            "behavior_success_rate": 0.50,
        },
        {},
        set(),
        search_space="post_target",
    )
    assert choice is not None
    assert choice.experiment_id == "post_scope_penalty_175_q25"


def test_choose_campaign_experiment_post_target_skips_completed_candidates() -> None:
    choice = choose_campaign_experiment(
        {
            "json_valid_rate": 1.0,
            "compile_success_rate": 0.65,
            "behavior_success_rate": 0.50,
        },
        {},
        {"post_scope_penalty_175_q25", "post_scope_penalty_225_q25"},
        search_space="post_target",
    )
    assert choice is not None
    assert choice.experiment_id == "post_scope_penalty_250_q25"


def test_choose_campaign_experiment_post_target_continues_with_q200_tranche() -> None:
    candidates = _post_target_campaign_candidates()
    candidate_ids = [candidate.experiment_id for candidate in candidates]
    first_q200_index = candidate_ids.index("post2_strict_data_q200")

    choice = choose_campaign_experiment(
        {
            "json_valid_rate": 1.0,
            "compile_success_rate": 0.65,
            "behavior_success_rate": 0.50,
        },
        {},
        set(candidate_ids[:first_q200_index]),
        search_space="post_target",
    )

    assert choice is not None
    assert choice.experiment_id == "post2_strict_data_q200"
    assert choice.overrides["training"]["max_steps"] == 200
    assert choice.dataset_overrides["prompt_mode"] == "context_plus_strict"


def test_choose_campaign_experiment_prioritizes_prompt_alignment_after_json_is_healthy() -> None:
    choice = choose_campaign_experiment(
        {
            "json_valid_rate": 0.99,
            "compile_success_rate": 0.67,
            "behavior_success_rate": 0.54,
        },
        {},
        set(),
    )
    assert choice is not None
    assert choice.experiment_id == "prompt_align_full_v1"


def test_choose_campaign_experiment_uses_constant_guard_after_context_plus_trials() -> None:
    choice = choose_campaign_experiment(
        {
            "json_valid_rate": 1.0,
            "compile_success_rate": 0.60,
            "behavior_success_rate": 0.42,
        },
        {},
        {
            "prompt_align_full_v1",
            "prompt_align_context_plus_v1",
            "prompt_align_context_plus_no_rename_v1",
        },
    )
    assert choice is not None
    assert choice.experiment_id == "context_plus_constant_guard_v1"


def test_choose_campaign_experiment_uses_strict_json_after_constant_guard() -> None:
    choice = choose_campaign_experiment(
        {
            "json_valid_rate": 0.98,
            "compile_success_rate": 0.62,
            "behavior_success_rate": 0.44,
        },
        {},
        {
            "prompt_align_full_v1",
            "prompt_align_context_plus_v1",
            "prompt_align_context_plus_no_rename_v1",
            "context_plus_constant_guard_v1",
        },
    )
    assert choice is not None
    assert choice.experiment_id == "context_plus_constant_strict_json_v1"


def test_choose_campaign_experiment_uses_type_guard_after_strict_json() -> None:
    choice = choose_campaign_experiment(
        {
            "json_valid_rate": 1.0,
            "compile_success_rate": 0.63,
            "behavior_success_rate": 0.49,
        },
        {},
        {
            "prompt_align_full_v1",
            "prompt_align_context_plus_v1",
            "prompt_align_context_plus_no_rename_v1",
            "context_plus_constant_guard_v1",
            "context_plus_constant_strict_json_v1",
        },
    )
    assert choice is not None
    assert choice.experiment_id == "context_plus_strict_type_guard_v1"


def test_choose_campaign_experiment_uses_behavior_nudge_after_type_guard() -> None:
    choice = choose_campaign_experiment(
        {
            "json_valid_rate": 1.0,
            "compile_success_rate": 0.63,
            "behavior_success_rate": 0.49,
        },
        {},
        {
            "prompt_align_full_v1",
            "prompt_align_context_plus_v1",
            "prompt_align_context_plus_no_rename_v1",
            "context_plus_constant_guard_v1",
            "context_plus_constant_strict_json_v1",
            "context_plus_strict_type_guard_v1",
        },
    )
    assert choice is not None
    assert choice.experiment_id == "context_plus_strict_behavior_nudge_v1"


def test_choose_campaign_experiment_uses_long500_after_behavior_nudge() -> None:
    choice = choose_campaign_experiment(
        {
            "json_valid_rate": 1.0,
            "compile_success_rate": 0.63,
            "behavior_success_rate": 0.49,
        },
        {},
        {
            "prompt_align_full_v1",
            "prompt_align_context_plus_v1",
            "prompt_align_context_plus_no_rename_v1",
            "context_plus_constant_guard_v1",
            "context_plus_constant_strict_json_v1",
            "context_plus_strict_type_guard_v1",
            "context_plus_strict_behavior_nudge_v1",
        },
    )
    assert choice is not None
    assert choice.experiment_id == "context_plus_strict_long500_v1"


def test_sft_target_passed_requires_score_gain_and_safety_gates() -> None:
    sft_metrics = {
        "json_valid_rate": 0.99,
        "compile_success_rate": 0.66,
        "behavior_success_rate": 0.53,
    }
    assert sft_target_passed(
        {
            "json_valid_rate": 0.99,
            "compile_success_rate": 0.66,
            "behavior_success_rate": 0.54,
        },
        candidate_score=0.765,
        sft_metrics=sft_metrics,
        sft_score=0.742,
        target_improvement=0.02,
    )
    assert not sft_target_passed(
        {
            "json_valid_rate": 0.99,
            "compile_success_rate": 0.62,
            "behavior_success_rate": 0.54,
        },
        candidate_score=0.765,
        sft_metrics=sft_metrics,
        sft_score=0.742,
        target_improvement=0.02,
    )


def test_pack_campaign_rl_records_supports_full_prompt(sample_dataset_samples) -> None:
    records = pack_campaign_rl_records(
        sample_dataset_samples,
        prompt_mode="full",
        task_types=["full_clarify", "rename"],
        prompt_limit=3,
    )
    assert len(records) == 3
    assert {record.task_type for record in records} <= {"full_clarify", "rename"}
    assert "Assembly:" in records[0].prompt
    assert records[0].prompt_messages[0].content == records[0].prompt
    assert sample_dataset_samples[0].source_function_name in records[0].allowed_callees


def test_pack_campaign_rl_records_supports_strict_context_prompt(
    sample_dataset_samples,
) -> None:
    records = pack_campaign_rl_records(
        sample_dataset_samples,
        prompt_mode="context_plus_strict",
        task_types=["full_clarify"],
        prompt_limit=1,
    )

    assert len(records) == 1
    assert "Keep renamings small" in records[0].prompt
    assert "Assembly:" not in records[0].prompt
