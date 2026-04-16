from __future__ import annotations

import pytest

from decomp_clarifier.evaluation.checkpoint_eval import find_latest_completed_checkpoint


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
