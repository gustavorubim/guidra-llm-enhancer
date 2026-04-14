from __future__ import annotations

import json
from pathlib import Path

from decomp_clarifier.evaluation.metrics import aggregate_metric
from decomp_clarifier.schemas.evaluation import EvaluationReport, SampleEvaluation

_COLUMN_ORDER = [
    "raw_ghidra",
    "naming_only",
    "base_qwen",
    "base_qwen_openrouter",
    "sft_checkpoint",
    "grpo_checkpoint",
    "prompt_only_cleanup",
    "generation_model",
    "strong_model",
]

_METRIC_ORDER = [
    "json_valid_rate",
    "field_complete_rate",
    "readability_score",
    "readability_improvement",
    "naming_score",
    "compile_success_rate",
    "behavior_success_rate",
]


def build_report(run_id: str, evaluations: list[SampleEvaluation]) -> EvaluationReport:
    metrics = {
        "json_valid_rate": aggregate_metric(evaluations, "json_valid"),
        "field_complete_rate": aggregate_metric(evaluations, "field_complete"),
        "readability_score": aggregate_metric(evaluations, "readability_score"),
        "naming_score": aggregate_metric(evaluations, "naming_score"),
        "compile_success_rate": aggregate_metric(evaluations, "compile_success"),
        "behavior_success_rate": aggregate_metric(evaluations, "behavior_success"),
    }
    return EvaluationReport(run_id=run_id, metrics=metrics, samples=evaluations)


def _ordered_system_names(systems: dict[str, dict[str, float]]) -> list[str]:
    known = [name for name in _COLUMN_ORDER if name in systems]
    unknown = sorted(name for name in systems if name not in _COLUMN_ORDER)
    return [*known, *unknown]


def _ordered_metric_names(systems: dict[str, dict[str, float]]) -> list[str]:
    present = {metric for values in systems.values() for metric in values}
    ordered = [metric for metric in _METRIC_ORDER if metric in present]
    extras = sorted(metric for metric in present if metric not in _METRIC_ORDER)
    return [*ordered, *extras]


def render_comparison_table(systems: dict[str, dict[str, float]]) -> str:
    if not systems:
        return ""
    system_names = _ordered_system_names(systems)
    metric_names = _ordered_metric_names(systems)
    header = "| Metric | " + " | ".join(system_names) + " |"
    separator = "|:---|" + "|".join("---:" for _ in system_names) + "|"
    rows = [header, separator]
    for metric_name in metric_names:
        values = []
        for system_name in system_names:
            value = systems.get(system_name, {}).get(metric_name)
            values.append(f"{value:.3f}" if value is not None else "--")
        rows.append("| " + " | ".join([metric_name, *values]) + " |")
    return "\n".join(rows)


def render_comparison_html_table(systems: dict[str, dict[str, float]]) -> str:
    if not systems:
        return "<table><thead><tr><th>Metric</th></tr></thead><tbody></tbody></table>"
    system_names = _ordered_system_names(systems)
    metric_names = _ordered_metric_names(systems)
    header_cells = "".join(f"<th>{name}</th>" for name in system_names)
    body_rows = []
    for metric_name in metric_names:
        value_cells = []
        for system_name in system_names:
            value = systems.get(system_name, {}).get(metric_name)
            value_cells.append(f"<td>{value:.3f}</td>" if value is not None else "<td>--</td>")
        body_rows.append(f"<tr><td>{metric_name}</td>{''.join(value_cells)}</tr>")
    return (
        "<table>"
        f"<thead><tr><th>Metric</th>{header_cells}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
    )


def render_markdown(report: EvaluationReport) -> str:
    lines = [
        f"# Evaluation Report: {report.run_id}",
        "",
        "## Metrics",
        "",
    ]
    lines.append(render_comparison_table({report.run_id: report.metrics}))
    lines.extend(["", "## Samples", ""])
    for sample in report.samples[:10]:
        lines.append(
            f"- {sample.sample_id} [{sample.system}] "
            f"readability={sample.readability_score:.3f} "
            f"naming={sample.naming_score:.3f}"
        )
    return "\n".join(lines) + "\n"


def render_html(report: EvaluationReport) -> str:
    return (
        "<html><body>"
        f"<h1>Evaluation Report: {report.run_id}</h1>"
        f"{render_comparison_html_table({report.run_id: report.metrics})}"
        "</body></html>"
    )


def write_report(report: EvaluationReport, output_dir: Path) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / f"{report.run_id}.md"
    html_path = output_dir / f"{report.run_id}.html"
    json_path = output_dir / f"{report.run_id}.json"
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    html_path.write_text(render_html(report), encoding="utf-8")
    json_path.write_text(
        json.dumps(report.model_dump(mode="python"), indent=2, sort_keys=True), encoding="utf-8"
    )
    return markdown_path, html_path, json_path
