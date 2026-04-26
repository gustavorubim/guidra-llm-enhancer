from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from decomp_clarifier.dataset.prompt_formatter import format_prompt  # noqa: E402
from decomp_clarifier.evaluation.checkpoint_eval import (  # noqa: E402
    evaluate_prediction_records,
    find_latest_completed_checkpoint,
    load_checkpoint_training_config,
    load_dataset_split,
    normalize_checkpoint_dir,
)
from decomp_clarifier.evaluation.report_builder import build_report, write_report  # noqa: E402
from decomp_clarifier.inference.agentic_repair import AgenticRepairPredictor  # noqa: E402
from decomp_clarifier.inference.checkpoint_predictor import CheckpointPredictor  # noqa: E402
from decomp_clarifier.paths import ProjectPaths  # noqa: E402
from decomp_clarifier.schemas.dataset import FunctionDatasetSample  # noqa: E402
from decomp_clarifier.schemas.model_io import PredictionRecord  # noqa: E402
from decomp_clarifier.settings import load_app_config, load_training_config  # noqa: E402

METRIC_WEIGHTS = {
    "behavior_success_rate": 0.30,
    "compile_success_rate": 0.25,
    "field_complete_rate": 0.20,
    "readability_score": 0.15,
    "naming_score": 0.10,
}
CONDITIONS = (
    "raw_qwen_direct",
    "raw_qwen_agentic_thinking",
    "gdpo_direct",
    "gdpo_agentic_no_thinking",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a raw-Qwen versus GDPO ablation with optional verifier-backed repair loops."
        )
    )
    parser.add_argument("--app-profile", default="default")
    parser.add_argument("--raw-model-id", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--raw-training-profile", default="sft_qwen35_2b")
    parser.add_argument("--gdpo-profile", default="grpo_qwen35_2b_gdpo_300")
    parser.add_argument("--gdpo-checkpoint-dir")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--sample-limit",
        type=int,
        help="Limit samples for a pilot run. Omit for the full split.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-repair-attempts", type=int, default=2)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITIONS,
        default=list(CONDITIONS),
        help="Subset of ablation conditions to run.",
    )
    return parser.parse_args()


def _run_id() -> str:
    return "agentic-ablation-" + datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _append_text_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(line.rstrip("\n") + "\n")


def _load_partial_predictions(path: Path) -> list[PredictionRecord]:
    if not path.exists():
        return []
    records: list[PredictionRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(PredictionRecord.model_validate_json(line))
    return records


def _load_partial_attempts(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    attempts: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            attempts.append(json.loads(line))
    return attempts


def _score(metrics: dict[str, float]) -> float:
    return sum(float(metrics.get(name, 0.0)) * weight for name, weight in METRIC_WEIGHTS.items())


def _progress_interval(sample_count: int) -> int:
    if sample_count <= 10:
        return 1
    return max(1, sample_count // 10)


def _evaluate_records(
    *,
    condition_dir: Path,
    condition: str,
    samples: list[FunctionDatasetSample],
    records: list[PredictionRecord],
    attempts: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    condition_dir.mkdir(parents=True, exist_ok=True)
    samples_by_id = {sample.sample_id: sample for sample in samples}
    predictions_path = condition_dir / "predictions.jsonl"
    predictions_path.write_text(
        "\n".join(record.model_dump_json() for record in records) + "\n",
        encoding="utf-8",
    )
    attempts_path = condition_dir / "attempts.jsonl"
    _write_jsonl(attempts_path, attempts)
    evaluations = evaluate_prediction_records(
        samples_by_id,
        records,
        progress_label=f"{condition} verifier",
    )
    evaluations_path = condition_dir / "sample_evaluations.jsonl"
    evaluations_path.write_text(
        "\n".join(item.model_dump_json() for item in evaluations) + "\n",
        encoding="utf-8",
    )
    report = build_report(condition, evaluations)
    report_markdown_path, report_html_path, report_json_path = write_report(
        report,
        condition_dir / "reports",
    )
    manifest = {
        "condition": condition,
        "config": config,
        "sample_count": len(samples),
        "metrics": report.metrics,
        "score": _score(report.metrics),
        "artifacts": {
            "predictions": str(predictions_path),
            "attempts": str(attempts_path),
            "sample_evaluations": str(evaluations_path),
            "report_markdown": str(report_markdown_path),
            "report_html": str(report_html_path),
            "report_json": str(report_json_path),
        },
    }
    manifest_path = condition_dir / "agentic_condition_manifest.json"
    _write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _run_direct_condition(
    *,
    condition_dir: Path,
    condition: str,
    predictor: CheckpointPredictor,
    samples: list[FunctionDatasetSample],
    max_new_tokens: int,
    temperature: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    condition_dir.mkdir(parents=True, exist_ok=True)
    incremental_predictions_path = condition_dir / "predictions.partial.jsonl"
    incremental_attempts_path = condition_dir / "attempts.partial.jsonl"
    records = _load_partial_predictions(incremental_predictions_path)
    attempts = _load_partial_attempts(incremental_attempts_path)
    completed_sample_ids = {record.sample_id for record in records}
    interval = _progress_interval(len(samples))
    started_at = time.perf_counter()
    for index, sample in enumerate(samples, start=1):
        if sample.sample_id in completed_sample_ids:
            continue
        record = predictor.predict(
            sample,
            system=condition,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        records.append(record)
        attempts.append(
            {
                "sample_id": sample.sample_id,
                "attempts": [
                    {
                        "attempt_index": 0,
                        "raw_text": record.raw_text,
                        "json_valid": record.json_valid,
                    }
                ],
            }
        )
        _append_text_line(incremental_predictions_path, record.model_dump_json())
        _append_text_line(
            incremental_attempts_path,
            json.dumps(attempts[-1], sort_keys=True),
        )
        if index == 1 or index == len(samples) or index % interval == 0:
            elapsed = max(time.perf_counter() - started_at, 1e-9)
            print(
                f"{condition} progress {index}/{len(samples)} "
                f"sample_id={sample.sample_id} rate={index / elapsed:.2f}/s",
                flush=True,
            )
    return _evaluate_records(
        condition_dir=condition_dir,
        condition=condition,
        samples=samples,
        records=records,
        attempts=attempts,
        config=config,
    )


def _run_agentic_condition(
    *,
    condition_dir: Path,
    condition: str,
    predictor: AgenticRepairPredictor,
    samples: list[FunctionDatasetSample],
    max_new_tokens: int,
    temperature: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    condition_dir.mkdir(parents=True, exist_ok=True)
    incremental_predictions_path = condition_dir / "predictions.partial.jsonl"
    incremental_attempts_path = condition_dir / "attempts.partial.jsonl"
    records = _load_partial_predictions(incremental_predictions_path)
    attempts = _load_partial_attempts(incremental_attempts_path)
    completed_sample_ids = {record.sample_id for record in records}
    interval = _progress_interval(len(samples))
    started_at = time.perf_counter()
    for index, sample in enumerate(samples, start=1):
        if sample.sample_id in completed_sample_ids:
            continue
        prediction = predictor.predict(
            sample,
            system=condition,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        records.append(prediction.record)
        attempts.append(
            {
                "sample_id": sample.sample_id,
                "attempts": [attempt.model_dump() for attempt in prediction.attempts],
            }
        )
        _append_text_line(incremental_predictions_path, prediction.record.model_dump_json())
        _append_text_line(
            incremental_attempts_path,
            json.dumps(attempts[-1], sort_keys=True),
        )
        if index == 1 or index == len(samples) or index % interval == 0:
            elapsed = max(time.perf_counter() - started_at, 1e-9)
            print(
                f"{condition} progress {index}/{len(samples)} "
                f"sample_id={sample.sample_id} rate={index / elapsed:.2f}/s",
                flush=True,
            )
    return _evaluate_records(
        condition_dir=condition_dir,
        condition=condition,
        samples=samples,
        records=records,
        attempts=attempts,
        config=config,
    )


def _load_evaluations(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    path = Path(str(manifest["artifacts"]["sample_evaluations"]))
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[str(row["sample_id"])] = row
    return rows


def _pairwise_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_rows = _load_evaluations(before)
    after_rows = _load_evaluations(after)
    common_ids = sorted(set(before_rows) & set(after_rows))
    result: dict[str, Any] = {"common_sample_count": len(common_ids)}
    for key in ("json_valid", "compile_success", "behavior_success"):
        result[f"{key}_gains"] = sum(
            not before_rows[item][key] and bool(after_rows[item][key])
            for item in common_ids
        )
        result[f"{key}_losses"] = sum(
            bool(before_rows[item][key]) and not after_rows[item][key]
            for item in common_ids
        )
    for key in ("readability_score", "naming_score"):
        result[f"{key}_mean_delta"] = (
            sum(float(after_rows[item][key]) - float(before_rows[item][key]) for item in common_ids)
            / len(common_ids)
            if common_ids
            else 0.0
        )
    return result


def _render_metrics_table(manifests: dict[str, dict[str, Any]]) -> str:
    lines = [
        "| Condition | Score | Behavior | Compile | JSON | Readability | Naming |",
        "|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, manifest in manifests.items():
        metrics = manifest["metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    condition,
                    f"{manifest['score']:.4f}",
                    f"{metrics['behavior_success_rate']:.4f}",
                    f"{metrics['compile_success_rate']:.4f}",
                    f"{metrics['json_valid_rate']:.4f}",
                    f"{metrics['readability_score']:.4f}",
                    f"{metrics['naming_score']:.4f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _render_delta_table(deltas: dict[str, dict[str, Any]]) -> str:
    lines = [
        "| Comparison | Common | JSON +/- | Compile +/- | Behavior +/- | Readability delta | "
        "Naming delta |",
        "|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, delta in deltas.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    str(delta["common_sample_count"]),
                    f"{delta['json_valid_gains']}/{delta['json_valid_losses']}",
                    f"{delta['compile_success_gains']}/{delta['compile_success_losses']}",
                    f"{delta['behavior_success_gains']}/{delta['behavior_success_losses']}",
                    f"{delta['readability_score_mean_delta']:.6f}",
                    f"{delta['naming_score_mean_delta']:.6f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _write_ablation_report(
    *,
    run_dir: Path,
    args: argparse.Namespace,
    manifests: dict[str, dict[str, Any]],
) -> tuple[Path, Path]:
    deltas: dict[str, dict[str, Any]] = {}
    if "raw_qwen_direct" in manifests and "raw_qwen_agentic_thinking" in manifests:
        deltas["raw_qwen_direct -> raw_qwen_agentic_thinking"] = _pairwise_delta(
            manifests["raw_qwen_direct"],
            manifests["raw_qwen_agentic_thinking"],
        )
    if "gdpo_direct" in manifests and "gdpo_agentic_no_thinking" in manifests:
        deltas["gdpo_direct -> gdpo_agentic_no_thinking"] = _pairwise_delta(
            manifests["gdpo_direct"],
            manifests["gdpo_agentic_no_thinking"],
        )
    if "raw_qwen_agentic_thinking" in manifests and "gdpo_direct" in manifests:
        deltas["raw_qwen_agentic_thinking -> gdpo_direct"] = _pairwise_delta(
            manifests["raw_qwen_agentic_thinking"],
            manifests["gdpo_direct"],
        )
    markdown = "\n".join(
        [
            "# Agentic Ablation Report",
            "",
            "## Configuration",
            "",
            f"- Raw model: `{args.raw_model_id}`",
            f"- GDPO profile: `{args.gdpo_profile}`",
            f"- Split: `{args.split}`",
            f"- Sample limit: `{args.sample_limit}`",
            f"- Max new tokens: `{args.max_new_tokens}`",
            f"- Max repair attempts: `{args.max_repair_attempts}`",
            f"- Temperature: `{args.temperature}`",
            "",
            "## Metrics",
            "",
            _render_metrics_table(manifests),
            "",
            "## Pairwise Deltas",
            "",
            _render_delta_table(deltas) if deltas else "No pairwise deltas available.",
            "",
            "## Condition Manifests",
            "",
            *[
                f"- {condition}: `{manifest['manifest_path']}`"
                for condition, manifest in manifests.items()
            ],
            "",
        ]
    )
    markdown_path = run_dir / "agentic_ablation_report.md"
    json_path = run_dir / "agentic_ablation_report.json"
    markdown_path.write_text(markdown, encoding="utf-8")
    _write_json(
        json_path,
        {
            "config": {
                "raw_model_id": args.raw_model_id,
                "gdpo_profile": args.gdpo_profile,
                "split": args.split,
                "sample_limit": args.sample_limit,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "max_repair_attempts": args.max_repair_attempts,
            },
            "metric_weights": METRIC_WEIGHTS,
            "conditions": manifests,
            "deltas": deltas,
        },
    )
    return markdown_path, json_path


def main() -> None:
    args = parse_args()
    app_config = load_app_config(ROOT, args.app_profile)
    paths = ProjectPaths.from_config(ROOT, app_config)
    paths.ensure()
    run_id = _run_id()
    run_dir = paths.run_dir(run_id)
    dataset_path = paths.processed_sft_dir / "function_dataset.jsonl"
    samples = load_dataset_split(dataset_path, split=args.split, sample_limit=args.sample_limit)
    if not samples:
        raise ValueError(f"No samples found for split={args.split!r}")

    manifests: dict[str, dict[str, Any]] = {}
    if "raw_qwen_direct" in args.conditions:
        raw_config = load_training_config(ROOT, args.raw_training_profile)
        raw_config.model.base_model_id = args.raw_model_id
        predictor = CheckpointPredictor(
            args.raw_model_id,
            raw_config,
            prompt_formatter=format_prompt,
            enable_thinking=False,
        )
        manifests["raw_qwen_direct"] = _run_direct_condition(
            condition_dir=run_dir / "raw_qwen_direct",
            condition="raw_qwen_direct",
            predictor=predictor,
            samples=samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            config={"model": args.raw_model_id, "agentic": False, "thinking": False},
        )

    if "raw_qwen_agentic_thinking" in args.conditions:
        raw_config = load_training_config(ROOT, args.raw_training_profile)
        raw_config.model.base_model_id = args.raw_model_id
        predictor = CheckpointPredictor(
            args.raw_model_id,
            raw_config,
            prompt_formatter=format_prompt,
            enable_thinking=True,
        )
        agentic_predictor = AgenticRepairPredictor(
            predictor,
            prompt_formatter=format_prompt,
            max_repair_attempts=args.max_repair_attempts,
        )
        manifests["raw_qwen_agentic_thinking"] = _run_agentic_condition(
            condition_dir=run_dir / "raw_qwen_agentic_thinking",
            condition="raw_qwen_agentic_thinking",
            predictor=agentic_predictor,
            samples=samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            config={"model": args.raw_model_id, "agentic": True, "thinking": True},
        )

    gdpo_checkpoint = normalize_checkpoint_dir(
        Path(args.gdpo_checkpoint_dir)
        if args.gdpo_checkpoint_dir
        else find_latest_completed_checkpoint(paths, "grpo", training_profile=args.gdpo_profile)
    )
    if "gdpo_direct" in args.conditions:
        gdpo_config = load_checkpoint_training_config(ROOT, gdpo_checkpoint, args.gdpo_profile)
        predictor = CheckpointPredictor(
            gdpo_checkpoint,
            gdpo_config,
            prompt_formatter=format_prompt,
            enable_thinking=False,
        )
        manifests["gdpo_direct"] = _run_direct_condition(
            condition_dir=run_dir / "gdpo_direct",
            condition="gdpo_direct",
            predictor=predictor,
            samples=samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            config={"model": str(gdpo_checkpoint), "agentic": False, "thinking": False},
        )

    if "gdpo_agentic_no_thinking" in args.conditions:
        gdpo_config = load_checkpoint_training_config(ROOT, gdpo_checkpoint, args.gdpo_profile)
        predictor = CheckpointPredictor(
            gdpo_checkpoint,
            gdpo_config,
            prompt_formatter=format_prompt,
            enable_thinking=False,
        )
        agentic_predictor = AgenticRepairPredictor(
            predictor,
            prompt_formatter=format_prompt,
            max_repair_attempts=args.max_repair_attempts,
        )
        manifests["gdpo_agentic_no_thinking"] = _run_agentic_condition(
            condition_dir=run_dir / "gdpo_agentic_no_thinking",
            condition="gdpo_agentic_no_thinking",
            predictor=agentic_predictor,
            samples=samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            config={"model": str(gdpo_checkpoint), "agentic": True, "thinking": False},
        )

    report_markdown, report_json = _write_ablation_report(
        run_dir=run_dir,
        args=args,
        manifests=manifests,
    )
    _write_json(
        run_dir / "agentic_ablation_manifest.json",
        {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "report_markdown": str(report_markdown),
            "report_json": str(report_json),
            "conditions": {
                condition: manifest["manifest_path"] for condition, manifest in manifests.items()
            },
        },
    )
    print(f"Agentic ablation report: {report_markdown}")
    print(f"Agentic ablation json: {report_json}")


if __name__ == "__main__":
    main()
