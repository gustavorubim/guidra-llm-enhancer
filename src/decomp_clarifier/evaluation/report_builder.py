from __future__ import annotations

import json
from pathlib import Path

from decomp_clarifier.evaluation.metrics import aggregate_metric
from decomp_clarifier.schemas.evaluation import EvaluationReport, SampleEvaluation


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


def render_markdown(report: EvaluationReport) -> str:
    lines = [
        f"# Evaluation Report: {report.run_id}",
        "",
        "## Metrics",
        "",
    ]
    lines.extend([f"- {name}: {value:.3f}" for name, value in report.metrics.items()])
    lines.extend(["", "## Samples", ""])
    for sample in report.samples[:10]:
        lines.append(
            f"- {sample.sample_id} [{sample.system}] "
            f"readability={sample.readability_score:.3f} "
            f"naming={sample.naming_score:.3f}"
        )
    return "\n".join(lines) + "\n"


def render_html(report: EvaluationReport) -> str:
    rows = "\n".join(
        f"<tr><td>{name}</td><td>{value:.3f}</td></tr>" for name, value in report.metrics.items()
    )
    return (
        "<html><body>"
        f"<h1>Evaluation Report: {report.run_id}</h1>"
        "<table>"
        "<thead><tr><th>Metric</th><th>Value</th></tr></thead>"
        f"<tbody>{rows}</tbody>"
        "</table>"
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
