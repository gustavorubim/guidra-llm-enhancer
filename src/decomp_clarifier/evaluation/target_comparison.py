from __future__ import annotations

import json
from pathlib import Path
from typing import Any

BASELINE_COLUMNS = [
    "raw_ghidra",
    "naming_only",
    "base_qwen",
    "base_qwen_openrouter",
    "prompt_only_cleanup",
    "generation_model",
    "strong_model",
]

TARGET_COLUMNS = [
    "raw_ghidra",
    "naming_only",
    "base_qwen",
    "base_qwen_openrouter",
    "sft",
    "grpo",
    "prompt_only_cleanup",
    "generation_model",
    "strong_model",
]

TARGET_METRICS = [
    "json_valid_rate",
    "readability_score",
    "naming_score",
    "compile_success_rate",
    "behavior_success_rate",
]


def find_latest_checkpoint_eval_manifest(root: Path, stage: str) -> Path:
    if stage not in {"sft", "grpo"}:
        raise ValueError(f"Unsupported checkpoint eval stage: {stage}")
    manifests = sorted(
        root.glob(f"artifacts/runs/eval-{stage}-checkpoint-*/checkpoint_eval_manifest.json")
    )
    if not manifests:
        raise FileNotFoundError(f"No eval-{stage}-checkpoint manifest found under {root}")
    return manifests[-1]


def load_checkpoint_eval_manifest(
    manifest_path: Path, *, expected_stage: str | None = None
) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping payload in {manifest_path}")
    if expected_stage is not None:
        actual_stage = payload.get("stage")
        if actual_stage != expected_stage:
            raise ValueError(
                f"Expected stage {expected_stage!r} in {manifest_path}, got {actual_stage!r}"
            )
    return payload


def _coerce_metrics(raw_metrics: object, *, context: str) -> dict[str, float]:
    if not isinstance(raw_metrics, dict):
        raise ValueError(f"Expected mapping for {context}")
    metrics: dict[str, float] = {}
    for metric_name, raw_value in raw_metrics.items():
        if not isinstance(metric_name, str):
            raise ValueError(f"Expected string metric name in {context}, got {metric_name!r}")
        if isinstance(raw_value, bool) or not isinstance(raw_value, int | float):
            raise ValueError(
                f"Expected numeric value for {context}.{metric_name}, got {raw_value!r}"
            )
        metrics[metric_name] = float(raw_value)
    return metrics


def _merge_baseline_metrics(manifests: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    merged: dict[str, dict[str, float]] = {}
    for manifest in manifests:
        baseline_metrics = _coerce_metrics_map(
            manifest.get("baseline_metrics", {}), context="baseline_metrics"
        )
        for system_name, metrics in baseline_metrics.items():
            existing = merged.setdefault(system_name, {})
            for metric_name, value in metrics.items():
                previous = existing.get(metric_name)
                if previous is not None and abs(previous - value) > 1e-12:
                    raise ValueError(
                        "Conflicting baseline metric for "
                        f"{system_name}.{metric_name}: {previous} != {value}"
                    )
                existing[metric_name] = value
    return merged


def _coerce_metrics_map(raw_metrics_map: object, *, context: str) -> dict[str, dict[str, float]]:
    if not isinstance(raw_metrics_map, dict):
        raise ValueError(f"Expected mapping for {context}")
    metrics_map: dict[str, dict[str, float]] = {}
    for system_name, raw_metrics in raw_metrics_map.items():
        if not isinstance(system_name, str):
            raise ValueError(f"Expected string system name in {context}, got {system_name!r}")
        metrics_map[system_name] = _coerce_metrics(
            raw_metrics, context=f"{context}.{system_name}"
        )
    return metrics_map


def build_target_comparison_systems(
    sft_manifest: dict[str, Any],
    grpo_manifest: dict[str, Any],
    *,
    extra_manifests: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, float]]:
    manifests = {"sft": sft_manifest, "grpo": grpo_manifest, **(extra_manifests or {})}
    return build_target_comparison_systems_from_manifests(
        manifests,
        baseline_columns=[
            "raw_ghidra",
            "naming_only",
            "base_qwen",
            "base_qwen_openrouter",
            "prompt_only_cleanup",
            "generation_model",
            "strong_model",
        ],
        pinned_columns=TARGET_COLUMNS,
    )


def build_target_comparison_systems_from_manifests(
    manifests_by_label: dict[str, dict[str, Any]],
    *,
    baseline_columns: list[str] | None = None,
    pinned_columns: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    baseline_columns = baseline_columns or BASELINE_COLUMNS
    pinned_columns = pinned_columns or baseline_columns
    baseline_metrics = _merge_baseline_metrics(list(manifests_by_label.values()))
    unexpected_baselines = sorted(
        name
        for name in baseline_metrics
        if name not in pinned_columns and name not in manifests_by_label
    )
    ordered_columns = [*pinned_columns, *manifests_by_label, *unexpected_baselines]
    systems = {column: {} for column in ordered_columns}
    for system_name, metrics in baseline_metrics.items():
        if system_name not in systems:
            systems[system_name] = {}
        systems[system_name] = dict(metrics)
    for label, manifest in manifests_by_label.items():
        systems[label] = _coerce_metrics(manifest.get("metrics", {}), context=f"{label}.metrics")
    return systems


def render_target_comparison_table(
    systems: dict[str, dict[str, float]], *, columns: list[str] | None = None
) -> str:
    ordered_columns = columns or list(systems)
    header = "| Metric | " + " | ".join(ordered_columns) + " |"
    separator = "|:---|" + "|".join("---:" for _ in ordered_columns) + "|"
    rows = [header, separator]
    for metric_name in TARGET_METRICS:
        values = []
        for system_name in ordered_columns:
            value = systems.get(system_name, {}).get(metric_name)
            values.append(f"{value:.3f}" if value is not None else "--")
        rows.append("| " + " | ".join([metric_name, *values]) + " |")
    return "\n".join(rows)
