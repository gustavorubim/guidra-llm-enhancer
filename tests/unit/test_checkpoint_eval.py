from __future__ import annotations

import pytest
import yaml

from decomp_clarifier.dataset.prompt_formatter import (
    format_context_plus_prompt,
    format_context_plus_strict_prompt,
    format_prompt,
    format_rl_prompt,
)
from decomp_clarifier.evaluation.checkpoint_eval import (
    find_latest_completed_checkpoint,
    resolve_checkpoint_prompt_formatter,
)


def test_find_latest_completed_checkpoint_filters_by_training_profile(temp_paths) -> None:
    first_run = temp_paths.run_dir("train-sft-20260415-010000")
    second_run = temp_paths.run_dir("train-sft-20260415-020000")
    third_run = temp_paths.run_dir("train-sft-20260415-030000")
    for run_dir, profile in (
        (first_run, "sft_qwen35_2b_12gb"),
        (second_run, "sft_qwen35_4b_12gb"),
        (third_run, "sft_qwen35_4b_12gb_1000"),
    ):
        model_dir = run_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "sft_training_manifest.json").write_text("{}", encoding="utf-8")
        (run_dir / "resolved_config.yaml").write_text(
            f"training_profile: {profile}\n",
            encoding="utf-8",
        )

    assert find_latest_completed_checkpoint(
        temp_paths,
        "sft",
        training_profile="sft_qwen35_4b_12gb_1000",
    ) == third_run / "model"


def test_resolve_checkpoint_prompt_formatter() -> None:
    assert resolve_checkpoint_prompt_formatter("grpo", "stage") is format_rl_prompt
    assert resolve_checkpoint_prompt_formatter("sft", "stage") is format_prompt
    assert resolve_checkpoint_prompt_formatter("sft", "compact") is format_rl_prompt
    assert resolve_checkpoint_prompt_formatter("grpo", "full") is format_prompt
    assert (
        resolve_checkpoint_prompt_formatter("grpo", "context_plus")
        is format_context_plus_prompt
    )
    assert (
        resolve_checkpoint_prompt_formatter("grpo", "context_plus_strict")
        is format_context_plus_strict_prompt
    )
    with pytest.raises(ValueError, match="prompt_profile"):
        resolve_checkpoint_prompt_formatter("grpo", "bad")


def test_find_latest_completed_checkpoint_rejects_ambiguous_unlabeled_profiles(temp_paths) -> None:
    for run_id in ("train-grpo-20260415-010000", "train-grpo-20260415-020000"):
        run_dir = temp_paths.run_dir(run_id)
        model_dir = run_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "grpo_training_manifest.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="training_profile"):
        find_latest_completed_checkpoint(
            temp_paths,
            "grpo",
            training_profile="grpo_qwen35_4b_12gb_1000",
        )


def test_find_latest_completed_checkpoint_skips_raw_grpo_for_sft_sourced_profile(
    temp_paths,
) -> None:
    profile_name = "grpo_test_profile"
    source_profile = "sft_test_profile"
    config_dir = temp_paths.root / "configs" / "training"
    config_dir.mkdir(parents=True)
    (config_dir / f"{profile_name}.yaml").write_text(
        yaml.safe_dump({"model": {"source_training_profile": source_profile}}),
        encoding="utf-8",
    )
    sft_checkpoint = temp_paths.run_dir("train-sft-20260415-000000") / "model"
    sft_checkpoint.mkdir(parents=True)
    (sft_checkpoint / "sft_training_manifest.json").write_text("{}", encoding="utf-8")

    good_run = temp_paths.run_dir("train-grpo-20260415-010000")
    good_model_dir = good_run / "model"
    good_model_dir.mkdir(parents=True)
    (good_model_dir / "grpo_training_manifest.json").write_text("{}", encoding="utf-8")
    (good_run / "resolved_config.yaml").write_text(
        yaml.safe_dump(
            {
                "training_profile": profile_name,
                "training": {
                    "model": {
                        "base_model_id": str(sft_checkpoint),
                        "source_training_profile": source_profile,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    raw_run = temp_paths.run_dir("train-grpo-20260415-020000")
    raw_model_dir = raw_run / "model"
    raw_model_dir.mkdir(parents=True)
    (raw_model_dir / "grpo_training_manifest.json").write_text("{}", encoding="utf-8")
    (raw_run / "resolved_config.yaml").write_text(
        yaml.safe_dump(
            {
                "training_profile": profile_name,
                "training": {
                    "model": {
                        "base_model_id": "Qwen/Qwen3.5-2B",
                        "source_training_profile": None,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    assert (
        find_latest_completed_checkpoint(
            temp_paths,
            "grpo",
            training_profile=profile_name,
        )
        == good_model_dir
    )
