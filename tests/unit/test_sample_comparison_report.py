from __future__ import annotations

import json
from pathlib import Path

from decomp_clarifier.baselines import raw_ghidra
from decomp_clarifier.evaluation.sample_comparison_report import build_sample_comparison_report
from decomp_clarifier.schemas.model_io import ClarifiedFunctionOutput, PredictionRecord


def _prediction(
    sample_id: str,
    system: str,
    cleaned_c: str,
    *,
    summary: str | None = None,
    json_valid: bool = True,
) -> PredictionRecord:
    return PredictionRecord(
        sample_id=sample_id,
        system=system,
        output=ClarifiedFunctionOutput(
            summary=summary or system,
            confidence=1.0,
            renamings={},
            cleaned_c=cleaned_c,
        ),
        json_valid=json_valid,
        raw_text=cleaned_c if not json_valid else None,
    )


def _write_records(path: Path, records: list[PredictionRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(record.model_dump_json() for record in records) + "\n",
        encoding="utf-8",
    )


def _write_eval_run(
    run_dir: Path,
    records: list[PredictionRecord],
    *,
    sample_count: int,
) -> Path:
    predictions_path = run_dir / "predictions.jsonl"
    _write_records(predictions_path, records)
    manifest_path = run_dir / "checkpoint_eval_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "sample_count": sample_count,
                "split": "all",
                "artifacts": {"predictions": str(predictions_path)},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return predictions_path


def test_build_sample_comparison_report_writes_artifacts(
    temp_paths,
    sample_dataset_samples,
) -> None:
    dataset_path = temp_paths.processed_sft_dir / "function_dataset.jsonl"
    dataset_rows = [
        sample.model_dump_json()
        for sample in sample_dataset_samples[:3]
    ]
    dataset_path.write_text("\n".join(dataset_rows) + "\n", encoding="utf-8")

    baseline_path = temp_paths.runs_dir / "baseline-20260421-000001" / "baseline_predictions.jsonl"
    baseline_records: list[PredictionRecord] = []
    for sample in sample_dataset_samples[:3]:
        baseline_records.extend(
            [
                PredictionRecord(
                    sample_id=sample.sample_id,
                    system="raw_ghidra",
                    output=raw_ghidra.predict(sample),
                ),
                _prediction(
                    sample.sample_id,
                    "prompt_only_cleanup",
                    f"int {sample.source_function_name}_cleanup(void) {{ return 1; }}",
                ),
                _prediction(
                    sample.sample_id,
                    "base_qwen_openrouter",
                    f"int {sample.source_function_name}_qwen(void) {{ return 2; }}",
                ),
            ]
        )
    _write_records(baseline_path, baseline_records)

    sft_best_dir = temp_paths.runs_dir / "eval-sft-checkpoint-best"
    _write_eval_run(
        sft_best_dir,
        [
            _prediction(
                sample.sample_id,
                "sft_checkpoint",
                f"int {sample.source_function_name}_sft(void) {{ return 3; }}",
            )
            for sample in sample_dataset_samples[:3]
        ],
        sample_count=3,
    )

    sft_scout_dir = temp_paths.runs_dir / "eval-sft-checkpoint-scout"
    _write_eval_run(
        sft_scout_dir,
        [
            _prediction(
                sample_dataset_samples[0].sample_id,
                "sft_checkpoint",
                "int scout(void) { return 9; }",
            )
        ],
        sample_count=1,
    )

    grpo_best_dir = temp_paths.runs_dir / "eval-grpo-checkpoint-best"
    _write_eval_run(
        grpo_best_dir,
        [
            _prediction(
                sample.sample_id,
                "grpo_checkpoint",
                f"int {sample.source_function_name}_grpo(void) {{ return 4; }}",
            )
            for sample in sample_dataset_samples[:3]
        ],
        sample_count=3,
    )

    artifacts = build_sample_comparison_report(
        paths=temp_paths,
        split="all",
        sample_count=2,
        seed=7,
    )

    manifest_payload = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    markdown = artifacts.markdown_path.read_text(encoding="utf-8")
    report_payload = json.loads(artifacts.json_path.read_text(encoding="utf-8"))

    assert artifacts.markdown_path.exists()
    assert artifacts.json_path.exists()
    assert manifest_payload["inputs"]["qwen_system"] == "base_qwen_openrouter"
    assert Path(manifest_payload["inputs"]["sft_manifest_path"]).parent.name == "eval-sft-checkpoint-best"
    assert "## Summary Comparison" in markdown
    assert "### Original Source" in markdown
    assert "### Qwen Via Prompt (base_qwen_openrouter)" in markdown
    assert len(report_payload["samples"]) == 2
    assert set(report_payload["metrics"]) == {
        "decompiled",
        "original_model",
        "qwen_via_prompt",
        "sft_model",
        "grpo_model",
    }
