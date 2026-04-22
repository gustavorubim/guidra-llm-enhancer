from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from decomp_clarifier.evaluation.checkpoint_eval import (
    evaluate_prediction_records,
    load_dataset_split,
)
from decomp_clarifier.evaluation.report_builder import build_report
from decomp_clarifier.paths import ProjectPaths
from decomp_clarifier.schemas.dataset import FunctionDatasetSample
from decomp_clarifier.schemas.model_io import PredictionRecord


@dataclass(frozen=True)
class SampleComparisonReportArtifacts:
    run_dir: Path
    manifest_path: Path
    markdown_path: Path
    json_path: Path


def _report_run_id() -> str:
    return f"sample-comparison-report-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return payload


def _candidate_eval_prediction_paths(
    paths: ProjectPaths,
    stage: str,
) -> list[tuple[int, float, Path, Path | None]]:
    candidates: list[tuple[int, float, Path, Path | None]] = []
    manifests = sorted(paths.runs_dir.glob(f"eval-{stage}-checkpoint-*/checkpoint_eval_manifest.json"))
    for manifest_path in manifests:
        payload = _load_json(manifest_path)
        artifacts = payload.get("artifacts", {})
        predictions_value = artifacts.get("predictions") if isinstance(artifacts, dict) else None
        predictions_path = Path(predictions_value) if isinstance(predictions_value, str) else None
        if predictions_path is not None and not predictions_path.is_absolute():
            predictions_path = paths.resolve(predictions_path)
        if predictions_path is None or not predictions_path.exists():
            predictions_path = manifest_path.parent / "predictions.jsonl"
        if not predictions_path.exists():
            continue
        sample_count = payload.get("sample_count", 0)
        sample_total = int(sample_count) if isinstance(sample_count, int | float) else 0
        candidates.append(
            (
                sample_total,
                manifest_path.stat().st_mtime,
                predictions_path,
                manifest_path,
            )
        )
    if candidates:
        return candidates

    predictions = sorted(paths.runs_dir.glob(f"eval-{stage}-checkpoint-*/predictions.jsonl"))
    return [
        (
            0,
            prediction_path.stat().st_mtime,
            prediction_path,
            None,
        )
        for prediction_path in predictions
    ]


def resolve_latest_eval_predictions(
    paths: ProjectPaths,
    *,
    stage: str,
    explicit_path: Path | None = None,
) -> tuple[Path, Path | None]:
    if explicit_path is not None:
        manifest_path = explicit_path.parent / "checkpoint_eval_manifest.json"
        return explicit_path, manifest_path if manifest_path.exists() else None

    candidates = _candidate_eval_prediction_paths(paths, stage)
    if not candidates:
        raise FileNotFoundError(f"No eval-{stage}-checkpoint predictions found.")
    _sample_count, _mtime, predictions_path, manifest_path = max(
        candidates,
        key=lambda item: (item[0], item[1]),
    )
    return predictions_path, manifest_path


def resolve_latest_baseline_predictions(
    paths: ProjectPaths,
    *,
    explicit_path: Path | None = None,
) -> Path:
    if explicit_path is not None:
        return explicit_path
    candidates = sorted(paths.runs_dir.glob("baseline-*/baseline_predictions.jsonl"))
    if not candidates:
        raise FileNotFoundError("No baseline predictions found.")
    return candidates[-1]


def load_prediction_records(path: Path) -> list[PredictionRecord]:
    records: list[PredictionRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(PredictionRecord.model_validate_json(line))
    return records


def _records_by_sample(records: list[PredictionRecord]) -> dict[str, PredictionRecord]:
    indexed: dict[str, PredictionRecord] = {}
    for record in records:
        if record.sample_id in indexed:
            raise ValueError(f"Duplicate prediction for sample_id={record.sample_id!r}.")
        indexed[record.sample_id] = record
    return indexed


def _baseline_predictions_by_system(path: Path) -> dict[str, dict[str, PredictionRecord]]:
    grouped: dict[str, dict[str, PredictionRecord]] = {}
    for record in load_prediction_records(path):
        system_records = grouped.setdefault(record.system, {})
        if record.sample_id in system_records:
            raise ValueError(
                f"Duplicate baseline prediction for system={record.system!r} "
                f"sample_id={record.sample_id!r}."
            )
        system_records[record.sample_id] = record
    return grouped


def _default_qwen_system(systems: dict[str, dict[str, PredictionRecord]]) -> str:
    for name in ("base_qwen", "base_qwen_openrouter"):
        if name in systems:
            return name
    raise ValueError("Baseline predictions do not contain base_qwen or base_qwen_openrouter.")


def _metric_value(value: bool | float) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return f"{value:.3f}"


def _render_metrics_table(
    labels: list[str],
    metrics_by_label: dict[str, dict[str, float]],
) -> str:
    metric_order = [
        "json_valid_rate",
        "field_complete_rate",
        "readability_score",
        "naming_score",
        "compile_success_rate",
        "behavior_success_rate",
    ]
    header = "| Metric | " + " | ".join(labels) + " |"
    separator = "|:---|" + "|".join("---:" for _ in labels) + "|"
    rows = [header, separator]
    for metric_name in metric_order:
        values = [
            f"{metrics_by_label.get(label, {}).get(metric_name, 0.0):.3f}" for label in labels
        ]
        rows.append("| " + " | ".join([metric_name, *values]) + " |")
    return "\n".join(rows)


def _render_sample_scorecard(
    sample_id: str,
    labels: list[str],
    evaluations_by_label: dict[str, dict[str, Any]],
) -> str:
    header = "| System | JSON | Compile | Behavior | Readability | Naming |"
    separator = "|:---|:---:|:---:|:---:|---:|---:|"
    rows = [header, separator]
    for label in labels:
        evaluation = evaluations_by_label[label][sample_id]
        rows.append(
            "| "
            + " | ".join(
                [
                    label,
                    _metric_value(evaluation.json_valid),
                    _metric_value(evaluation.compile_success),
                    _metric_value(evaluation.behavior_success),
                    _metric_value(evaluation.readability_score),
                    _metric_value(evaluation.naming_score),
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def _system_section(
    *,
    title: str,
    record: PredictionRecord,
) -> list[str]:
    lines = [
        f"### {title}",
        "",
        f"- Summary: {record.output.summary or 'None'}",
        f"- JSON valid: `{record.json_valid}`",
        f"- Confidence: `{record.output.confidence:.3f}`",
        "",
        "```c",
        record.output.cleaned_c,
        "```",
        "",
    ]
    if not record.json_valid and record.raw_text:
        lines.extend(
            [
                "Raw model text:",
                "```text",
                record.raw_text,
                "```",
                "",
            ]
        )
    return lines


def build_sample_comparison_report(
    *,
    paths: ProjectPaths,
    split: str = "val",
    sample_count: int = 20,
    seed: int | None = None,
    baseline_predictions_path: Path | None = None,
    sft_predictions_path: Path | None = None,
    grpo_predictions_path: Path | None = None,
    original_system: str = "prompt_only_cleanup",
    qwen_system: str | None = None,
) -> SampleComparisonReportArtifacts:
    run_id = _report_run_id()
    run_dir = paths.run_dir(run_id)

    dataset_path = paths.processed_sft_dir / "function_dataset.jsonl"
    baseline_path = resolve_latest_baseline_predictions(
        paths,
        explicit_path=baseline_predictions_path,
    )
    sft_path, sft_manifest_path = resolve_latest_eval_predictions(
        paths,
        stage="sft",
        explicit_path=sft_predictions_path,
    )
    grpo_path, grpo_manifest_path = resolve_latest_eval_predictions(
        paths,
        stage="grpo",
        explicit_path=grpo_predictions_path,
    )

    samples = load_dataset_split(dataset_path, split=split)
    samples_by_id = {sample.sample_id: sample for sample in samples}

    baseline_by_system = _baseline_predictions_by_system(baseline_path)
    if original_system not in baseline_by_system:
        raise ValueError(f"Baseline predictions do not contain system={original_system!r}.")
    resolved_qwen_system = qwen_system or _default_qwen_system(baseline_by_system)
    if resolved_qwen_system not in baseline_by_system:
        raise ValueError(f"Baseline predictions do not contain system={resolved_qwen_system!r}.")
    if "raw_ghidra" not in baseline_by_system:
        raise ValueError("Baseline predictions do not contain system='raw_ghidra'.")

    sft_records = _records_by_sample(load_prediction_records(sft_path))
    grpo_records = _records_by_sample(load_prediction_records(grpo_path))

    common_sample_ids = sorted(
        set(samples_by_id)
        & set(baseline_by_system["raw_ghidra"])
        & set(baseline_by_system[original_system])
        & set(baseline_by_system[resolved_qwen_system])
        & set(sft_records)
        & set(grpo_records)
    )
    if len(common_sample_ids) < sample_count:
        raise ValueError(
            f"Only {len(common_sample_ids)} common samples are available for the requested "
            f"comparison, but sample_count={sample_count}."
        )

    resolved_seed = seed if seed is not None else random.SystemRandom().randrange(1, 2**31)
    rng = random.Random(resolved_seed)
    selected_sample_ids = rng.sample(common_sample_ids, sample_count)
    selected_samples_by_id = {sample_id: samples_by_id[sample_id] for sample_id in selected_sample_ids}

    labels_in_order = [
        "decompiled",
        "original_model",
        "qwen_via_prompt",
        "sft_model",
        "grpo_model",
    ]
    records_by_label = {
        "decompiled": {
            sample_id: baseline_by_system["raw_ghidra"][sample_id] for sample_id in selected_sample_ids
        },
        "original_model": {
            sample_id: baseline_by_system[original_system][sample_id]
            for sample_id in selected_sample_ids
        },
        "qwen_via_prompt": {
            sample_id: baseline_by_system[resolved_qwen_system][sample_id]
            for sample_id in selected_sample_ids
        },
        "sft_model": {sample_id: sft_records[sample_id] for sample_id in selected_sample_ids},
        "grpo_model": {sample_id: grpo_records[sample_id] for sample_id in selected_sample_ids},
    }

    metrics_by_label: dict[str, dict[str, float]] = {}
    evaluations_by_label: dict[str, dict[str, Any]] = {}
    for label in labels_in_order:
        ordered_records = [records_by_label[label][sample_id] for sample_id in selected_sample_ids]
        evaluations = evaluate_prediction_records(selected_samples_by_id, ordered_records)
        metrics_by_label[label] = build_report(label, evaluations).metrics
        evaluations_by_label[label] = {item.sample_id: item for item in evaluations}

    markdown_lines = [
        f"# Sample Comparison Report: {run_id}",
        "",
        f"- split: `{split}`",
        f"- sample_count: `{sample_count}`",
        f"- random_seed: `{resolved_seed}`",
        f"- baseline_predictions: `{baseline_path}`",
        f"- original_system: `{original_system}`",
        f"- qwen_system: `{resolved_qwen_system}`",
        f"- sft_predictions: `{sft_path}`",
        f"- grpo_predictions: `{grpo_path}`",
        "",
        "## Summary Comparison",
        "",
        _render_metrics_table(labels_in_order, metrics_by_label),
        "",
    ]

    json_samples: list[dict[str, Any]] = []
    for sample_id in selected_sample_ids:
        sample = selected_samples_by_id[sample_id]
        markdown_lines.extend(
            [
                f"## {sample.sample_id}",
                "",
                f"- project_id: `{sample.project_id}`",
                f"- function: `{sample.source_function_name}`",
                f"- task_type: `{sample.task_type}`",
                f"- compiler: `{sample.compiler}`",
                f"- opt_level: `{sample.opt_level}`",
                "",
                _render_sample_scorecard(sample_id, labels_in_order, evaluations_by_label),
                "",
                "### Original Source",
                "",
                "```c",
                sample.source_code,
                "```",
                "",
                "### Decompiled",
                "",
                "```c",
                sample.ghidra_decompiled_code,
                "```",
                "",
            ]
        )
        markdown_lines.extend(
            _system_section(
                title=f"Original Model ({original_system})",
                record=records_by_label["original_model"][sample_id],
            )
        )
        markdown_lines.extend(
            _system_section(
                title=f"Qwen Via Prompt ({resolved_qwen_system})",
                record=records_by_label["qwen_via_prompt"][sample_id],
            )
        )
        markdown_lines.extend(
            _system_section(
                title="SFT Model",
                record=records_by_label["sft_model"][sample_id],
            )
        )
        markdown_lines.extend(
            _system_section(
                title="GRPO Model",
                record=records_by_label["grpo_model"][sample_id],
            )
        )

        json_samples.append(
            {
                "sample_id": sample.sample_id,
                "project_id": sample.project_id,
                "source_function_name": sample.source_function_name,
                "task_type": sample.task_type,
                "original_source": sample.source_code,
                "decompiled": sample.ghidra_decompiled_code,
                "systems": {
                    label: {
                        "system": records_by_label[label][sample_id].system,
                        "json_valid": records_by_label[label][sample_id].json_valid,
                        "summary": records_by_label[label][sample_id].output.summary,
                        "cleaned_c": records_by_label[label][sample_id].output.cleaned_c,
                        "evaluation": evaluations_by_label[label][sample_id].model_dump(
                            mode="python"
                        ),
                    }
                    for label in labels_in_order
                },
            }
        )

    markdown_path = run_dir / "sample_comparison_report.md"
    markdown_path.write_text("\n".join(markdown_lines).rstrip() + "\n", encoding="utf-8")

    json_path = run_dir / "sample_comparison_report.json"
    json_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "split": split,
                "sample_count": sample_count,
                "random_seed": resolved_seed,
                "dataset_path": str(dataset_path),
                "baseline_predictions_path": str(baseline_path),
                "sft_predictions_path": str(sft_path),
                "grpo_predictions_path": str(grpo_path),
                "original_system": original_system,
                "qwen_system": resolved_qwen_system,
                "metrics": metrics_by_label,
                "samples": json_samples,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    manifest_path = run_dir / "sample_comparison_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "inputs": {
                    "dataset_path": str(dataset_path),
                    "baseline_predictions_path": str(baseline_path),
                    "sft_predictions_path": str(sft_path),
                    "grpo_predictions_path": str(grpo_path),
                    "sft_manifest_path": str(sft_manifest_path) if sft_manifest_path else None,
                    "grpo_manifest_path": str(grpo_manifest_path) if grpo_manifest_path else None,
                    "split": split,
                    "sample_count": sample_count,
                    "random_seed": resolved_seed,
                    "original_system": original_system,
                    "qwen_system": resolved_qwen_system,
                },
                "artifacts": {
                    "markdown": str(markdown_path),
                    "json": str(json_path),
                },
                "selected_sample_ids": selected_sample_ids,
                "generated_at_epoch_s": time.time(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return SampleComparisonReportArtifacts(
        run_dir=run_dir,
        manifest_path=manifest_path,
        markdown_path=markdown_path,
        json_path=json_path,
    )
