from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import yaml

from decomp_clarifier.dataset.prompt_formatter import format_prompt, format_rl_prompt
from decomp_clarifier.evaluation.readability_eval import readability_improvement
from decomp_clarifier.evaluation.report_builder import build_report, write_report
from decomp_clarifier.inference.checkpoint_predictor import CheckpointPredictor
from decomp_clarifier.inference.explain import summarize_improvements
from decomp_clarifier.paths import ProjectPaths
from decomp_clarifier.schemas.dataset import FunctionDatasetSample
from decomp_clarifier.schemas.evaluation import ReportExample, SampleEvaluation
from decomp_clarifier.schemas.model_io import ClarifiedFunctionOutput, PredictionRecord
from decomp_clarifier.settings import TrainingConfig, load_training_config
from decomp_clarifier.training.grpo.verifier import verify_output


@dataclass(frozen=True)
class CheckpointEvalArtifacts:
    manifest_path: Path
    predictions_path: Path
    evaluations_path: Path
    inspection_markdown_path: Path
    inspection_jsonl_path: Path
    comparison_markdown_path: Path
    report_markdown_path: Path
    report_html_path: Path
    report_json_path: Path


def find_latest_completed_checkpoint(paths: ProjectPaths, stage: str) -> Path:
    manifest_name = (
        "sft_training_manifest.json" if stage == "sft" else "grpo_training_manifest.json"
    )
    manifests = sorted(paths.runs_dir.glob(f"train-{stage}-*/model/{manifest_name}"))
    if not manifests:
        raise FileNotFoundError(f"No completed train-{stage} checkpoint manifest found.")
    return manifests[-1].parent


def normalize_checkpoint_dir(checkpoint_dir: Path) -> Path:
    if checkpoint_dir.is_dir() and (checkpoint_dir / "model").is_dir():
        return checkpoint_dir / "model"
    return checkpoint_dir


def load_checkpoint_training_config(
    root: Path, checkpoint_dir: Path, training_profile: str
) -> TrainingConfig:
    resolved_path = checkpoint_dir.parent / "resolved_config.yaml"
    if resolved_path.exists():
        payload = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
        training_payload = payload.get("training")
        if isinstance(training_payload, dict):
            config = TrainingConfig.model_validate(training_payload)
            config.model.base_model_id = str(checkpoint_dir)
            return config
    config = load_training_config(root, training_profile)
    config.model.base_model_id = str(checkpoint_dir)
    return config


def load_dataset_split(
    dataset_path: Path, *, split: str, sample_limit: int | None = None
) -> list[FunctionDatasetSample]:
    samples: list[FunctionDatasetSample] = []
    seen_ids: set[str] = set()
    for line in dataset_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sample = FunctionDatasetSample.model_validate_json(line)
        if split != "all" and sample.split != split:
            continue
        if sample.sample_id in seen_ids:
            raise ValueError(f"Duplicate sample_id in dataset split: {sample.sample_id}")
        seen_ids.add(sample.sample_id)
        samples.append(sample)
        if sample_limit is not None and len(samples) >= sample_limit:
            break
    return samples


def evaluate_prediction_records(
    samples_by_id: dict[str, FunctionDatasetSample],
    records: Iterable[PredictionRecord],
) -> list[SampleEvaluation]:
    evaluations: list[SampleEvaluation] = []
    for record in records:
        sample = samples_by_id.get(record.sample_id)
        if sample is None:
            continue
        verification = verify_output(sample, record.output)
        evaluations.append(
            SampleEvaluation(
                sample_id=record.sample_id,
                system=record.system,
                json_valid=record.json_valid,
                field_complete=verification.field_complete,
                placeholder_ratio=verification.placeholder_ratio,
                readability_score=verification.readability_score,
                naming_score=verification.naming_score,
                compile_success=verification.compile_success,
                behavior_success=verification.behavior_success,
                notes=[],
            )
        )
    return evaluations


def load_baseline_reports(
    paths: ProjectPaths, samples_by_id: dict[str, FunctionDatasetSample]
) -> dict[str, dict[str, float]]:
    baseline_paths = sorted(paths.runs_dir.glob("baseline-*/baseline_predictions.jsonl"))
    if not baseline_paths:
        return {}

    records: list[PredictionRecord] = []
    for line in baseline_paths[-1].read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload["sample_id"] not in samples_by_id:
            continue
        records.append(
            PredictionRecord(
                sample_id=payload["sample_id"],
                system=payload["system"],
                output=ClarifiedFunctionOutput.model_validate(payload["output"]),
                raw_text=None,
                json_valid=True,
            )
        )

    reports: dict[str, dict[str, float]] = {}
    systems = sorted({record.system for record in records})
    for system in systems:
        system_records = [record for record in records if record.system == system]
        evaluations = evaluate_prediction_records(samples_by_id, system_records)
        reports[system] = build_report(f"baseline-{system}", evaluations).metrics
    return reports


def _inspection_item(
    sample: FunctionDatasetSample, record: PredictionRecord, evaluation: SampleEvaluation
) -> dict[str, object]:
    return {
        "sample": sample,
        "record": record,
        "evaluation": evaluation,
        "readability_improvement": readability_improvement(
            record.output.cleaned_c, sample.ghidra_decompiled_code
        ),
    }


def select_inspection_items(
    samples_by_id: dict[str, FunctionDatasetSample],
    records: list[PredictionRecord],
    evaluations: list[SampleEvaluation],
    *,
    limit: int,
) -> list[dict[str, object]]:
    records_by_id = {record.sample_id: record for record in records}
    items = [
        _inspection_item(
            samples_by_id[evaluation.sample_id],
            records_by_id[evaluation.sample_id],
            evaluation,
        )
        for evaluation in evaluations
        if evaluation.sample_id in records_by_id and evaluation.sample_id in samples_by_id
    ]

    successes = sorted(
        (
            item
            for item in items
            if item["evaluation"].compile_success or item["evaluation"].behavior_success
        ),
        key=lambda item: (
            int(item["evaluation"].behavior_success),
            int(item["evaluation"].compile_success),
            float(item["readability_improvement"]),
        ),
        reverse=True,
    )
    failures = sorted(
        (
            item
            for item in items
            if not (item["evaluation"].compile_success and item["evaluation"].behavior_success)
        ),
        key=lambda item: (
            int(item["evaluation"].compile_success),
            int(item["evaluation"].behavior_success),
            float(item["readability_improvement"]),
        ),
    )

    selected: list[dict[str, object]] = []
    seen: set[str] = set()
    target_successes = max(1, limit // 2) if limit > 1 else 1
    target_failures = limit - target_successes
    success_count = 0
    failure_count = 0

    pools = ((successes, target_successes), (failures, target_failures), (items, limit))
    for pool, target in pools:
        for item in pool:
            sample_id = item["sample"].sample_id
            if sample_id in seen:
                continue
            selected.append(item)
            seen.add(sample_id)
            if pool is successes:
                success_count += 1
                if success_count >= target:
                    break
            if pool is failures:
                failure_count += 1
                if failure_count >= target:
                    break
            if len(selected) >= limit:
                break
        if len(selected) >= limit:
            break
    return selected[:limit]


def write_inspection_samples(
    items: list[dict[str, object]], markdown_path: Path, jsonl_path: Path
) -> None:
    markdown_lines = ["# Inspection Samples", ""]
    jsonl_lines: list[str] = []
    for item in items:
        sample = item["sample"]
        record = item["record"]
        evaluation = item["evaluation"]
        improvement = float(item["readability_improvement"])
        notes = summarize_improvements(sample, record.output)

        markdown_lines.extend(
            [
                f"## {sample.sample_id} [{sample.split}/{sample.task_type}]",
                "",
                f"- Source function: `{sample.source_function_name}`",
                f"- Compile success: `{evaluation.compile_success}`",
                f"- Behavior success: `{evaluation.behavior_success}`",
                f"- JSON valid: `{record.json_valid}`",
                f"- Readability score: `{evaluation.readability_score:.3f}`",
                f"- Naming score: `{evaluation.naming_score:.3f}`",
                f"- Placeholder ratio: `{evaluation.placeholder_ratio:.3f}`",
                f"- Readability improvement vs decompiled: `{improvement:.3f}`",
                f"- Notes: {'; '.join(notes) if notes else 'None'}",
                "",
                "### Original Source",
                "```c",
                sample.source_code,
                "```",
                "",
                "### Decompiled",
                "```c",
                sample.ghidra_decompiled_code,
                "```",
                "",
                "### Reconstructed",
                "```c",
                record.output.cleaned_c,
                "```",
                "",
            ]
        )

        jsonl_lines.append(
            json.dumps(
                {
                    "sample_id": sample.sample_id,
                    "split": sample.split,
                    "task_type": sample.task_type,
                    "source_function_name": sample.source_function_name,
                    "compile_success": evaluation.compile_success,
                    "behavior_success": evaluation.behavior_success,
                    "json_valid": record.json_valid,
                    "readability_score": evaluation.readability_score,
                    "naming_score": evaluation.naming_score,
                    "placeholder_ratio": evaluation.placeholder_ratio,
                    "readability_improvement": improvement,
                    "raw_text": record.raw_text,
                    "summary": record.output.summary,
                    "original_source": sample.source_code,
                    "decompiled": sample.ghidra_decompiled_code,
                    "reconstructed": record.output.cleaned_c,
                },
                sort_keys=True,
            )
        )

    markdown_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
    jsonl_path.write_text("\n".join(jsonl_lines) + ("\n" if jsonl_lines else ""), encoding="utf-8")


def render_comparison_markdown(
    *,
    run_id: str,
    stage: str,
    checkpoint_dir: Path,
    split: str,
    report_metrics: dict[str, float],
    baseline_metrics: dict[str, dict[str, float]],
    sample_count: int,
) -> str:
    lines = [
        f"# {stage.upper()} Checkpoint Evaluation: {run_id}",
        "",
        f"- checkpoint: `{checkpoint_dir}`",
        f"- split: `{split}`",
        f"- evaluated samples: `{sample_count}`",
        "",
        "## Checkpoint Metrics",
        "",
    ]
    lines.extend(f"- {name}: {value:.3f}" for name, value in report_metrics.items())
    if baseline_metrics:
        lines.extend(["", "## Baseline Comparison", ""])
        for system, metrics in sorted(baseline_metrics.items()):
            lines.append(f"### {system}")
            lines.append("")
            lines.extend(f"- {name}: {value:.3f}" for name, value in metrics.items())
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def run_checkpoint_evaluation(
    *,
    root: Path,
    paths: ProjectPaths,
    run_id: str,
    run_dir: Path,
    stage: str,
    checkpoint_dir: Path,
    training_profile: str,
    split: str,
    sample_limit: int | None,
    inspection_sample_count: int,
    max_new_tokens: int,
    temperature: float,
) -> CheckpointEvalArtifacts:
    checkpoint_dir = normalize_checkpoint_dir(checkpoint_dir)
    config = load_checkpoint_training_config(root, checkpoint_dir, training_profile)
    dataset_path = paths.processed_sft_dir / "function_dataset.jsonl"
    samples = load_dataset_split(dataset_path, split=split, sample_limit=sample_limit)
    if not samples:
        raise ValueError(f"No dataset samples found for split '{split}'.")
    samples_by_id = {sample.sample_id: sample for sample in samples}

    prompt_formatter = format_rl_prompt if stage == "grpo" else format_prompt
    predictor = CheckpointPredictor(
        checkpoint_dir,
        config,
        prompt_formatter=prompt_formatter,
    )
    system = f"{stage}_checkpoint"
    records = [
        predictor.predict(
            sample,
            system=system,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        for sample in samples
    ]

    predictions_path = run_dir / "predictions.jsonl"
    predictions_path.write_text(
        "\n".join(record.model_dump_json() for record in records) + "\n",
        encoding="utf-8",
    )

    evaluations = evaluate_prediction_records(samples_by_id, records)
    evaluations_path = run_dir / "sample_evaluations.jsonl"
    evaluations_path.write_text(
        "\n".join(item.model_dump_json() for item in evaluations) + "\n",
        encoding="utf-8",
    )

    report = build_report(run_id, evaluations)
    report.examples = [
        ReportExample(
            sample_id=sample.sample_id,
            project_id=sample.project_id,
            source_function_name=sample.source_function_name,
            raw_ghidra=sample.ghidra_decompiled_code,
            candidate=records[index].output.cleaned_c,
            original_source=sample.target_clean_code,
            note="; ".join(summarize_improvements(sample, records[index].output)),
        )
        for index, sample in enumerate(samples[:5])
    ]
    report_markdown_path, report_html_path, report_json_path = write_report(
        report, run_dir / "reports"
    )

    inspection_items = select_inspection_items(
        samples_by_id,
        records,
        evaluations,
        limit=inspection_sample_count,
    )
    inspection_markdown_path = run_dir / "inspection_samples.md"
    inspection_jsonl_path = run_dir / "inspection_samples.jsonl"
    write_inspection_samples(inspection_items, inspection_markdown_path, inspection_jsonl_path)

    baseline_metrics = load_baseline_reports(paths, samples_by_id)
    comparison_markdown_path = run_dir / "comparison.md"
    comparison_markdown_path.write_text(
        render_comparison_markdown(
            run_id=run_id,
            stage=stage,
            checkpoint_dir=checkpoint_dir,
            split=split,
            report_metrics=report.metrics,
            baseline_metrics=baseline_metrics,
            sample_count=len(samples),
        ),
        encoding="utf-8",
    )

    manifest_path = run_dir / "checkpoint_eval_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "stage": stage,
                "checkpoint_dir": str(checkpoint_dir),
                "split": split,
                "sample_count": len(samples),
                "training_profile": training_profile,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "metrics": report.metrics,
                "baseline_metrics": baseline_metrics,
                "artifacts": {
                    "predictions": str(predictions_path),
                    "sample_evaluations": str(evaluations_path),
                    "report_markdown": str(report_markdown_path),
                    "report_html": str(report_html_path),
                    "report_json": str(report_json_path),
                    "inspection_markdown": str(inspection_markdown_path),
                    "inspection_jsonl": str(inspection_jsonl_path),
                    "comparison_markdown": str(comparison_markdown_path),
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return CheckpointEvalArtifacts(
        manifest_path=manifest_path,
        predictions_path=predictions_path,
        evaluations_path=evaluations_path,
        inspection_markdown_path=inspection_markdown_path,
        inspection_jsonl_path=inspection_jsonl_path,
        comparison_markdown_path=comparison_markdown_path,
        report_markdown_path=report_markdown_path,
        report_html_path=report_html_path,
        report_json_path=report_json_path,
    )
