from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_summary_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "build_model_matrix_summary.py"
    spec = importlib.util.spec_from_file_location("build_model_matrix_summary", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_summary_payload_uses_expected_columns_and_manifest_map() -> None:
    module = _load_summary_module()

    payload = module._build_summary_payload(
        {
            "sft_qwen35_2b": {
                "__manifest_path": "artifacts/runs/eval-sft-2b/checkpoint_eval_manifest.json",
                "metrics": {"json_valid_rate": 0.75},
            },
            "grpo_qwen35_2b": {
                "__manifest_path": "artifacts/runs/eval-grpo-2b/checkpoint_eval_manifest.json",
                "metrics": {"json_valid_rate": 0.8},
            },
        }
    )

    assert payload["columns"][: len(module.BASELINE_COLUMNS)] == module.BASELINE_COLUMNS
    assert payload["columns"][len(module.BASELINE_COLUMNS) :] == module.SUMMARY_PROFILE_COLUMNS
    assert payload["eval_manifests"]["sft_qwen35_2b"].endswith("checkpoint_eval_manifest.json")
    assert payload["eval_manifests"]["grpo_qwen35_2b"].endswith("checkpoint_eval_manifest.json")
    assert "| Metric |" in payload["table_markdown"]
