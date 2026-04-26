from __future__ import annotations

import json
from collections import Counter
from datetime import datetime

import pytest

from decomp_clarifier.research.autoresearch import (
    RewardTelemetrySnapshot,
    _next_candidate_branch,
    _render_runtime_prompt_contract,
    choose_experiment,
    classify_invalid_prediction_rows,
    parse_program_control,
    score_metrics,
    tag_for_date,
)


def test_tag_for_date_uses_lowercase_month_day() -> None:
    assert tag_for_date(datetime(2026, 4, 14)) == "apr14-grpo"


def test_parse_program_control_reads_stop_section() -> None:
    program = """
## STOP

Status: continue
Reason: active

## Mission
"""
    assert parse_program_control(program) == ("continue", "active")


def test_score_metrics_matches_research_formula() -> None:
    metrics = {
        "behavior_success_rate": 0.8,
        "compile_success_rate": 0.6,
        "json_valid_rate": 0.9,
        "readability_score": 0.7,
        "naming_score": 0.5,
    }
    assert score_metrics(metrics) == pytest.approx(0.725)


def test_classify_invalid_prediction_rows_distinguishes_parse_and_schema_errors() -> None:
    rows = [
        {
            "json_valid": False,
            "raw_text": (
                '{"summary":"x","confidence":1.0,"renamings":["param_1","cfg"],'
                '"cleaned_c":"int x(void){return 0;}"}'
            ),
        },
        {
            "json_valid": False,
            "raw_text": (
                '{"summary":"x","confidence":1.0,"renamings":{},'
                '"cleaned_c":"int x(void){return 0;}"'
            ),
        },
        {"json_valid": True, "raw_text": json.dumps({"summary": "ok"})},
    ]
    assert classify_invalid_prediction_rows(rows) == Counter(
        {"renamings_list": 1, "json_parse_error": 1}
    )


def test_choose_experiment_prioritizes_hard_safety_gate_for_compile_gap() -> None:
    choice = choose_experiment(
        {
            "json_valid_rate": 0.87,
            "compile_success_rate": 0.59,
            "behavior_success_rate": 0.86,
        },
        {"json_parse_error": 9, "renamings_list": 4},
        set(),
        recent_entries=[
            {
                "metrics": {
                    "compile_success_rate": 0.5,
                    "behavior_success_rate": 0.84,
                    "json_valid_rate": 0.84,
                }
            }
        ],
    )
    assert choice.experiment_id == "reward_hard_safety_gate_v1"


def test_choose_experiment_uses_prompt_contract_when_safety_is_healthy() -> None:
    choice = choose_experiment(
        {
            "json_valid_rate": 0.87,
            "compile_success_rate": 0.72,
            "behavior_success_rate": 0.9,
        },
        {"json_parse_error": 9},
        set(),
    )
    assert choice.experiment_id == "runtime_prompt_contract_v1"


def test_rendered_runtime_prompt_contract_is_valid_python() -> None:
    rendered = _render_runtime_prompt_contract(1)
    compile(rendered, "generated_grpo_data.py", "exec")


def test_choose_experiment_prefers_rollout_cooling_after_reward_collapse() -> None:
    choice = choose_experiment(
        {
            "json_valid_rate": 0.98,
            "compile_success_rate": 0.7,
            "behavior_success_rate": 0.9,
        },
        {},
        set(),
        reward_telemetry=RewardTelemetrySnapshot(
            reward_mean=10.4,
            reward_std=0.3,
            gate_factor_mean=1.0,
            compile_mean=1.0,
            behavior_mean=1.0,
            behavior_from_execution_mean=1.0,
            json_valid_mean=1.0,
            signature_mean=0.9,
        ),
    )
    assert choice.experiment_id == "rollout_cooling_v1"


def test_choose_experiment_moves_to_signature_bias_after_safety_and_cooling_rounds() -> None:
    choice = choose_experiment(
        {
            "json_valid_rate": 0.98,
            "compile_success_rate": 0.66,
            "behavior_success_rate": 0.87,
        },
        {},
        {
            "reward_hard_safety_gate_v1",
            "reward_hard_safety_gate_v2",
            "reward_hard_safety_gate_v3",
            "rollout_cooling_v1",
            "rollout_cooling_v2",
            "rollout_cooling_v3",
        },
    )
    assert choice.experiment_id == "reward_signature_bias_v1"


def test_next_candidate_branch_skips_existing_stale_branch() -> None:
    iteration, branch = _next_candidate_branch(
        "apr14-grpo",
        [{"iteration": 6}],
        [
            "autoresearch/apr14-grpo",
            "autoresearch-tmp/apr14-grpo-0007",
        ],
    )
    assert iteration == 8
    assert branch == "autoresearch-tmp/apr14-grpo-0008"
