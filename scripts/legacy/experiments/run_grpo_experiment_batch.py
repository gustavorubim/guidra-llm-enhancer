from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from decomp_clarifier.c_source import extract_called_functions
from decomp_clarifier.dataset.prompt_formatter import format_rl_prompt
from decomp_clarifier.evaluation.checkpoint_eval import (
    evaluate_prediction_records,
    find_latest_completed_checkpoint,
    load_dataset_split,
)
from decomp_clarifier.evaluation.report_builder import build_report
from decomp_clarifier.inference.checkpoint_predictor import CheckpointPredictor
from decomp_clarifier.logging import configure_logging
from decomp_clarifier.paths import ProjectPaths
from decomp_clarifier.schemas.dataset import FunctionDatasetSample, PackedRLRecord
from decomp_clarifier.settings import load_app_config, load_dotenv, load_training_config
from decomp_clarifier.training.grpo.train import run_grpo_training

ScoreDict = dict[str, float]
PromptBuilder = Callable[[FunctionDatasetSample], str]


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    description: str
    task_types: tuple[str, ...]
    train_prompt_mode: str
    natural_eval_prompt_mode: str
    common_eval_prompt_mode: str
    include_source_function_in_allowed_calls: bool = False
    max_steps: int = 150
    max_completion_length: int = 384
    eval_max_new_tokens: int = 384


EXPERIMENTS: tuple[ExperimentSpec, ...] = (
    ExperimentSpec(
        name="A0_clean_baseline",
        description=(
            "Train-only split hygiene, no reward/prompt/data intervention beyond the "
            "clean experiment harness."
        ),
        task_types=("full_clarify", "cleanup", "rename"),
        train_prompt_mode="compact",
        natural_eval_prompt_mode="compact",
        common_eval_prompt_mode="compact",
    ),
    ExperimentSpec(
        name="A1_self_call_fix",
        description="Allow the current function name in the reward-side allowed call set.",
        task_types=("full_clarify", "cleanup", "rename"),
        train_prompt_mode="compact",
        natural_eval_prompt_mode="compact",
        common_eval_prompt_mode="compact",
        include_source_function_in_allowed_calls=True,
    ),
    ExperimentSpec(
        name="A2_curated_dataset",
        description="Remove rename-only rows and train only on full_clarify + cleanup.",
        task_types=("full_clarify", "cleanup"),
        train_prompt_mode="compact",
        natural_eval_prompt_mode="compact",
        common_eval_prompt_mode="compact",
    ),
    ExperimentSpec(
        name="A3_context_restore",
        description="Restore additional binary-grounded metadata in the RL prompt.",
        task_types=("full_clarify", "cleanup", "rename"),
        train_prompt_mode="context_plus",
        natural_eval_prompt_mode="context_plus",
        common_eval_prompt_mode="compact",
    ),
    ExperimentSpec(
        name="A4_longer_completion",
        description="Increase rollout/eval completion budget to reduce truncation.",
        task_types=("full_clarify", "cleanup", "rename"),
        train_prompt_mode="compact",
        natural_eval_prompt_mode="compact",
        common_eval_prompt_mode="compact",
        max_completion_length=512,
        eval_max_new_tokens=512,
    ),
    ExperimentSpec(
        name="A5_more_steps",
        description="Double the GRPO step budget while keeping the baseline setup fixed.",
        task_types=("full_clarify", "cleanup", "rename"),
        train_prompt_mode="compact",
        natural_eval_prompt_mode="compact",
        common_eval_prompt_mode="compact",
        max_steps=300,
    ),
)


def format_context_plus_prompt(sample: FunctionDatasetSample) -> str:
    return "\n".join(
        [
            "You are a binary-grounded code clarification assistant.",
            f"Task: {sample.task_type}",
            "Return exactly one JSON object with keys summary, confidence, renamings, cleaned_c.",
            "Do not include markdown, commentary, XML tags, or <think> blocks.",
            "",
            "Decompiler:",
            sample.ghidra_decompiled_code,
            "",
            f"Strings: {json.dumps(sample.strings)}",
            f"Imports: {json.dumps(sample.imports)}",
            f"Callers: {json.dumps(sample.callers)}",
            f"Callees: {json.dumps(sample.callees)}",
            f"Semantic summary: {sample.semantic_summary}",
            "JSON:",
        ]
    )


PROMPT_BUILDERS: dict[str, PromptBuilder] = {
    "compact": format_rl_prompt,
    "context_plus": format_context_plus_prompt,
}


def _now_id(prefix: str) -> str:
    return f"{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def _score(metrics: dict[str, float]) -> float:
    return (
        0.30 * metrics.get("behavior_success_rate", 0.0)
        + 0.25 * metrics.get("compile_success_rate", 0.0)
        + 0.20 * metrics.get("json_valid_rate", 0.0)
        + 0.15 * metrics.get("readability_score", 0.0)
        + 0.10 * metrics.get("naming_score", 0.0)
    )


def _empty_cuda_cache() -> None:
    try:
        import torch  # type: ignore[import-not-found]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    gc.collect()


def _load_samples(dataset_path: Path, *, split: str) -> list[FunctionDatasetSample]:
    samples = load_dataset_split(dataset_path, split=split)
    return sorted(samples, key=lambda sample: sample.sample_id)


def _allowed_callees(
    sample: FunctionDatasetSample, *, include_source_function: bool
) -> list[str]:
    ordered = [*sample.callees, *extract_called_functions(sample.target_clean_code)]
    if include_source_function:
        ordered.append(sample.source_function_name)
    return list(dict.fromkeys(ordered))


def _pack_rl_records(
    samples: list[FunctionDatasetSample],
    *,
    prompt_builder: PromptBuilder,
    include_source_function_in_allowed_calls: bool,
    task_types: tuple[str, ...],
) -> list[PackedRLRecord]:
    allowed_task_types = set(task_types)
    filtered_samples = [sample for sample in samples if sample.task_type in allowed_task_types]
    return [
        PackedRLRecord(
            sample_id=sample.sample_id,
            task_type=sample.task_type,
            prompt=prompt_builder(sample),
            source_function_name=sample.source_function_name,
            raw_code=sample.ghidra_decompiled_code,
            compile_reference_source=sample.compile_reference_source or sample.source_code,
            target_clean_code=sample.target_clean_code,
            target_renamings=json.dumps(sample.rename_map_target, sort_keys=True),
            allowed_imports=json.dumps(sample.imports),
            allowed_callees=json.dumps(
                _allowed_callees(
                    sample,
                    include_source_function=include_source_function_in_allowed_calls,
                )
            ),
        )
        for sample in filtered_samples
    ]


def _write_jsonl(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            row.model_dump_json() if hasattr(row, "model_dump_json") else json.dumps(row)
            for row in rows
        )
        + ("\n" if rows else ""),
        encoding="utf-8",
    )


def _evaluate_checkpoint(
    *,
    name: str,
    checkpoint_dir: Path,
    config: Any,
    samples: list[FunctionDatasetSample],
    prompt_builder: PromptBuilder,
    max_new_tokens: int,
    output_dir: Path,
    logger: logging.Logger,
) -> dict[str, Any]:
    logger.info(
        "evaluating checkpoint name=%s prompt_mode=%s samples=%s max_new_tokens=%s",
        name,
        next(key for key, value in PROMPT_BUILDERS.items() if value is prompt_builder),
        len(samples),
        max_new_tokens,
    )
    predictor = CheckpointPredictor(
        checkpoint_dir,
        config,
        prompt_formatter=prompt_builder,
    )
    records = [
        predictor.predict(
            sample,
            system=name,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
        )
        for sample in samples
    ]
    del predictor
    _empty_cuda_cache()

    _write_jsonl(output_dir / "predictions.jsonl", records)
    samples_by_id = {sample.sample_id: sample for sample in samples}
    evaluations = evaluate_prediction_records(samples_by_id, records, logger=logger, progress_label=name)
    _write_jsonl(output_dir / "sample_evaluations.jsonl", evaluations)
    report = build_report(name, evaluations)
    metrics = dict(report.metrics)
    metrics["study_score"] = _score(metrics)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "metrics": metrics,
        "predictions": str(output_dir / "predictions.jsonl"),
        "sample_evaluations": str(output_dir / "sample_evaluations.jsonl"),
    }


def _write_batch_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "experiment",
        "dataset_rows",
        "train_prompt_mode",
        "common_score",
        "common_json_valid",
        "common_compile",
        "common_behavior",
        "common_readability",
        "common_naming",
        "natural_score",
        "natural_json_valid",
        "natural_compile",
        "natural_behavior",
        "natural_readability",
        "natural_naming",
    ]
    lines = [
        "# GRPO Experiment Batch Summary",
        "",
        "| Experiment | Rows | Train Prompt | Common Score | Common JSON | Common Compile | Common Behavior | Common Readability | Common Naming | Natural Score | Natural JSON | Natural Compile | Natural Behavior | Natural Readability | Natural Naming |",
        "|:---|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        common = row["common_eval"]["metrics"]
        natural = row["natural_eval"]["metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    row["experiment"],
                    str(row["dataset_rows"]),
                    row["train_prompt_mode"],
                    f"{common['study_score']:.3f}",
                    f"{common['json_valid_rate']:.3f}",
                    f"{common['compile_success_rate']:.3f}",
                    f"{common['behavior_success_rate']:.3f}",
                    f"{common['readability_score']:.3f}",
                    f"{common['naming_score']:.3f}",
                    f"{natural['study_score']:.3f}",
                    f"{natural['json_valid_rate']:.3f}",
                    f"{natural['compile_success_rate']:.3f}",
                    f"{natural['behavior_success_rate']:.3f}",
                    f"{natural['readability_score']:.3f}",
                    f"{natural['naming_score']:.3f}",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_final_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _configure_training_config(
    root: Path,
    *,
    base_checkpoint: Path,
    experiment: ExperimentSpec,
) -> Any:
    config = load_training_config(root, "grpo_qwen35_2b")
    config.model.base_model_id = str(base_checkpoint)
    config.training.max_steps = experiment.max_steps
    config.training.save_steps = experiment.max_steps
    config.training.max_completion_length = experiment.max_completion_length
    return config


def _run_sft_reference_eval(
    *,
    root: Path,
    dataset_path: Path,
    val_samples: list[FunctionDatasetSample],
    sft_checkpoint: Path,
    batch_dir: Path,
    logger: logging.Logger,
) -> dict[str, Any]:
    config = load_training_config(root, "sft_qwen35_2b")
    config.model.base_model_id = str(sft_checkpoint)
    output_dir = batch_dir / "sft_reference_compact_eval"
    result = _evaluate_checkpoint(
        name="sft_reference_compact",
        checkpoint_dir=sft_checkpoint,
        config=config,
        samples=val_samples,
        prompt_builder=PROMPT_BUILDERS["compact"],
        max_new_tokens=384,
        output_dir=output_dir,
        logger=logger,
    )
    _write_final_manifest(
        output_dir / "manifest.json",
        {
            "checkpoint": str(sft_checkpoint),
            "dataset_path": str(dataset_path),
            "prompt_mode": "compact",
            **result,
        },
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a sequential GRPO experiment batch.")
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=[experiment.name for experiment in EXPERIMENTS],
        help="Subset of experiment names to run.",
    )
    args = parser.parse_args()

    root = ProjectPaths.discover()
    load_dotenv(root)
    app_config = load_app_config(root)
    paths = ProjectPaths.from_config(root, app_config)
    paths.ensure()

    batch_id = _now_id("grpo-exp-batch")
    batch_dir = paths.runs_dir / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(app_config.run.log_level, paths.log_file(batch_id), True)

    dataset_path = paths.processed_sft_dir / "function_dataset.jsonl"
    train_samples = _load_samples(dataset_path, split="train")
    val_samples = _load_samples(dataset_path, split="val")
    sft_checkpoint = find_latest_completed_checkpoint(paths, "sft")
    logger.info(
        "starting grpo experiment batch id=%s train_samples=%s val_samples=%s sft_checkpoint=%s",
        batch_id,
        len(train_samples),
        len(val_samples),
        sft_checkpoint,
    )

    selected_experiments = [
        experiment for experiment in EXPERIMENTS if experiment.name in set(args.experiments)
    ]
    if not selected_experiments:
        raise ValueError("No experiments selected.")

    sft_reference = _run_sft_reference_eval(
        root=root,
        dataset_path=dataset_path,
        val_samples=val_samples,
        sft_checkpoint=sft_checkpoint,
        batch_dir=batch_dir,
        logger=logger,
    )

    rows: list[dict[str, Any]] = []
    for experiment in selected_experiments:
        exp_started_at = time.perf_counter()
        exp_dir = batch_dir / experiment.name
        exp_dir.mkdir(parents=True, exist_ok=True)
        logger.info("starting experiment=%s description=%s", experiment.name, experiment.description)

        prompt_builder = PROMPT_BUILDERS[experiment.train_prompt_mode]
        records = _pack_rl_records(
            train_samples,
            prompt_builder=prompt_builder,
            include_source_function_in_allowed_calls=(
                experiment.include_source_function_in_allowed_calls
            ),
            task_types=experiment.task_types,
        )
        dataset_file = exp_dir / "rl_records.train.jsonl"
        _write_jsonl(dataset_file, records)

        config = _configure_training_config(
            root,
            base_checkpoint=sft_checkpoint,
            experiment=experiment,
        )
        (exp_dir / "config_snapshot.json").write_text(
            json.dumps(
                {
                    "experiment": asdict(experiment),
                    "training_config": config.model_dump(mode="python"),
                    "dataset_rows": len(records),
                    "train_task_counts": {
                        task_type: sum(1 for record in records if record.task_type == task_type)
                        for task_type in sorted({record.task_type for record in records})
                    },
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        manifest_path = run_grpo_training(dataset_file, exp_dir / "model", config)
        logger.info("completed training experiment=%s manifest=%s", experiment.name, manifest_path)

        common_eval = _evaluate_checkpoint(
            name=f"{experiment.name}_common",
            checkpoint_dir=exp_dir / "model",
            config=config,
            samples=val_samples,
            prompt_builder=PROMPT_BUILDERS[experiment.common_eval_prompt_mode],
            max_new_tokens=experiment.eval_max_new_tokens,
            output_dir=exp_dir / "eval_common",
            logger=logger,
        )
        natural_eval = (
            common_eval
            if experiment.common_eval_prompt_mode == experiment.natural_eval_prompt_mode
            else _evaluate_checkpoint(
                name=f"{experiment.name}_natural",
                checkpoint_dir=exp_dir / "model",
                config=config,
                samples=val_samples,
                prompt_builder=PROMPT_BUILDERS[experiment.natural_eval_prompt_mode],
                max_new_tokens=experiment.eval_max_new_tokens,
                output_dir=exp_dir / "eval_natural",
                logger=logger,
            )
        )

        elapsed = time.perf_counter() - exp_started_at
        row = {
            "experiment": experiment.name,
            "description": experiment.description,
            "dataset_rows": len(records),
            "train_prompt_mode": experiment.train_prompt_mode,
            "manifest_path": str(manifest_path),
            "common_eval": common_eval,
            "natural_eval": natural_eval,
            "elapsed_seconds": elapsed,
        }
        rows.append(row)
        _write_final_manifest(exp_dir / "manifest.json", row)
        _write_final_manifest(
            batch_dir / "summary.json",
            {
                "batch_id": batch_id,
                "sft_reference_compact": sft_reference,
                "experiments": rows,
            },
        )
        _write_batch_summary(batch_dir / "summary.md", rows)
        logger.info(
            "finished experiment=%s elapsed_seconds=%.1f common_score=%.3f natural_score=%.3f",
            experiment.name,
            elapsed,
            row["common_eval"]["metrics"]["study_score"],
            row["natural_eval"]["metrics"]["study_score"],
        )
        _empty_cuda_cache()

    _write_final_manifest(
        batch_dir / "final_manifest.json",
        {
            "batch_id": batch_id,
            "started_at": batch_id,
            "root": str(root),
            "dataset_path": str(dataset_path),
            "sft_checkpoint": str(sft_checkpoint),
            "sft_reference_compact": sft_reference,
            "experiments": rows,
        },
    )
    _write_batch_summary(batch_dir / "summary.md", rows)
    logger.info("completed grpo experiment batch summary=%s", batch_dir / "summary.md")
    print(batch_dir)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    main()
