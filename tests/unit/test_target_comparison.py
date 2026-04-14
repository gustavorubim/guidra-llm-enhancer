from __future__ import annotations

import pytest

from decomp_clarifier.evaluation.target_comparison import (
    TARGET_COLUMNS,
    build_target_comparison_systems,
    find_latest_checkpoint_eval_manifest,
    render_target_comparison_table,
)


def test_build_target_comparison_systems_and_render_table() -> None:
    sft_manifest = {
        "stage": "sft",
        "metrics": {
            "json_valid_rate": 0.9,
            "readability_score": 0.82,
            "naming_score": 0.61,
            "compile_success_rate": 0.73,
            "behavior_success_rate": 0.9,
        },
        "baseline_metrics": {
            "raw_ghidra": {
                "json_valid_rate": 1.0,
                "readability_score": 0.53,
                "naming_score": 0.0,
                "compile_success_rate": 0.02,
                "behavior_success_rate": 0.0,
            },
            "naming_only": {
                "json_valid_rate": 1.0,
                "readability_score": 0.53,
                "naming_score": 0.22,
                "compile_success_rate": 0.02,
                "behavior_success_rate": 0.0,
            },
            "prompt_only_cleanup": {
                "json_valid_rate": 1.0,
                "readability_score": 0.72,
                "naming_score": 0.14,
                "compile_success_rate": 0.32,
                "behavior_success_rate": 0.36,
            },
            "generation_model": {
                "json_valid_rate": 1.0,
                "readability_score": 0.55,
                "naming_score": 0.01,
                "compile_success_rate": 0.06,
                "behavior_success_rate": 0.0,
            },
            "strong_model": {
                "json_valid_rate": 1.0,
                "readability_score": 0.53,
                "naming_score": 0.0,
                "compile_success_rate": 0.06,
                "behavior_success_rate": 0.0,
            },
        },
    }
    grpo_manifest = {
        "stage": "grpo",
        "metrics": {
            "json_valid_rate": 0.88,
            "readability_score": 0.83,
            "naming_score": 0.52,
            "compile_success_rate": 0.59,
            "behavior_success_rate": 0.87,
        },
        "baseline_metrics": sft_manifest["baseline_metrics"],
    }

    systems = build_target_comparison_systems(sft_manifest, grpo_manifest)

    assert list(systems) == TARGET_COLUMNS
    assert systems["base_qwen"] == {}
    assert systems["base_qwen_openrouter"] == {}

    table = render_target_comparison_table(systems)

    assert (
        table.splitlines()[0]
        == "| Metric | raw_ghidra | naming_only | base_qwen | base_qwen_openrouter | sft | grpo | "
        "prompt_only_cleanup | generation_model | strong_model |"
    )
    assert (
        "| json_valid_rate | 1.000 | 1.000 | -- | -- | 0.900 | 0.880 | 1.000 | 1.000 | 1.000 |"
        in table
    )
    assert (
        "| behavior_success_rate | 0.000 | 0.000 | -- | -- | "
        "0.900 | 0.870 | 0.360 | 0.000 | 0.000 |"
        in table
    )


def test_build_target_comparison_systems_rejects_conflicting_baselines() -> None:
    sft_manifest = {
        "metrics": {},
        "baseline_metrics": {"raw_ghidra": {"json_valid_rate": 1.0}},
    }
    grpo_manifest = {
        "metrics": {},
        "baseline_metrics": {"raw_ghidra": {"json_valid_rate": 0.5}},
    }

    with pytest.raises(ValueError, match="Conflicting baseline metric"):
        build_target_comparison_systems(sft_manifest, grpo_manifest)


def test_find_latest_checkpoint_eval_manifest(tmp_path) -> None:
    earlier = (
        tmp_path
        / "artifacts"
        / "runs"
        / "eval-sft-checkpoint-20260413-204842"
        / "checkpoint_eval_manifest.json"
    )
    later = (
        tmp_path
        / "artifacts"
        / "runs"
        / "eval-sft-checkpoint-20260414-001500"
        / "checkpoint_eval_manifest.json"
    )
    earlier.parent.mkdir(parents=True, exist_ok=True)
    later.parent.mkdir(parents=True, exist_ok=True)
    earlier.write_text("{}", encoding="utf-8")
    later.write_text("{}", encoding="utf-8")

    assert find_latest_checkpoint_eval_manifest(tmp_path, "sft") == later
