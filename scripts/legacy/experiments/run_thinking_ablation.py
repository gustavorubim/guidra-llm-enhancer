from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from decomp_clarifier.adapters.subprocess_utils import run_subprocess  # noqa: E402
from decomp_clarifier.evaluation.target_comparison import (  # noqa: E402
    build_target_comparison_systems_from_manifests,
    load_checkpoint_eval_manifest,
    render_target_comparison_table,
)
from decomp_clarifier.paths import ProjectPaths  # noqa: E402
from decomp_clarifier.settings import load_app_config  # noqa: E402

METRIC_WEIGHTS = {
    "behavior_success_rate": 0.30,
    "compile_success_rate": 0.25,
    "field_complete_rate": 0.20,
    "readability_score": 0.15,
    "naming_score": 0.10,
}
ABLATED_COLUMNS = [
    "raw_ghidra",
    "prompt_only_cleanup",
    "sft_no_thinking",
    "sft_thinking",
    "gdpo_no_thinking",
    "gdpo_thinking",
]


class _TeeWriter:
    def __init__(self, *writers: TextIO) -> None:
        self._writers = writers

    def write(self, data: str) -> int:
        for writer in self._writers:
            writer.write(data)
        return len(data)

    def flush(self) -> None:
        for writer in self._writers:
            writer.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the SFT/GDPO thinking ablation with a controlled prompt and token budget, "
            "then write a combined report."
        )
    )
    parser.add_argument("--app-profile", default="default")
    parser.add_argument("--sft-profile", default="sft_qwen35_2b")
    parser.add_argument("--grpo-profile", default="grpo_qwen35_2b_gdpo_300")
    parser.add_argument("--sft-checkpoint-dir")
    parser.add_argument("--grpo-checkpoint-dir")
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--inspection-sample-count", type=int, default=8)
    parser.add_argument("--prompt-profile", default="full", choices=("stage", "compact", "full"))
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Use the same larger budget for all four conditions so thinking has room.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not stream child command output live; only write per-condition logs.",
    )
    return parser.parse_args()


def _run_id() -> str:
    return "thinking-ablation-" + datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")


def _pythonpath_env() -> dict[str, str]:
    env = dict(os.environ)
    root_src = str(ROOT / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = root_src if not existing else os.pathsep.join([root_src, existing])
    return env


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _parse_manifest_path(stdout: str) -> Path:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Eval command did not emit a checkpoint eval manifest path.")
    return Path(lines[-1])


def _eval_command(args: argparse.Namespace, *, stage: str, thinking: bool) -> list[str]:
    command_name = "eval-sft-checkpoint" if stage == "sft" else "eval-grpo-checkpoint"
    training_profile = args.sft_profile if stage == "sft" else args.grpo_profile
    checkpoint_dir = args.sft_checkpoint_dir if stage == "sft" else args.grpo_checkpoint_dir
    command = [
        sys.executable,
        "-m",
        "decomp_clarifier.cli",
        command_name,
        "--training-profile",
        training_profile,
        "--app-profile",
        args.app_profile,
        "--split",
        args.split,
        "--inspection-sample-count",
        str(args.inspection_sample_count),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
        "--prompt-profile",
        args.prompt_profile,
    ]
    if checkpoint_dir:
        command.extend(["--checkpoint-dir", checkpoint_dir])
    if args.sample_limit is not None:
        command.extend(["--sample-limit", str(args.sample_limit)])
    if thinking:
        command.append("--thinking")
    return command


def _run_condition(
    *,
    run_dir: Path,
    env: dict[str, str],
    args: argparse.Namespace,
    label: str,
    stage: str,
    thinking: bool,
) -> dict[str, Any]:
    command = _eval_command(args, stage=stage, thinking=thinking)
    log_path = run_dir / f"{label}.log"
    print(f"[{label}] running {' '.join(command[1:])}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        stdout_sink = log_file if args.quiet else _TeeWriter(sys.stdout, log_file)
        stderr_sink = log_file if args.quiet else _TeeWriter(sys.stderr, log_file)
        result = run_subprocess(
            command,
            cwd=ROOT,
            env=env,
            stdout_sink=stdout_sink,
            stderr_sink=stderr_sink,
        )
    result.raise_for_error()
    manifest_path = _parse_manifest_path(result.stdout)
    manifest = load_checkpoint_eval_manifest(manifest_path, expected_stage=stage)
    return {
        "label": label,
        "stage": stage,
        "thinking": thinking,
        "command": command,
        "log_path": str(log_path),
        "manifest_path": str(manifest_path),
        "manifest": manifest,
    }


def _score(metrics: dict[str, Any]) -> float:
    return sum(float(metrics.get(name, 0.0)) * weight for name, weight in METRIC_WEIGHTS.items())


def _load_evaluations(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    evaluations_path = Path(str(manifest["artifacts"]["sample_evaluations"]))
    rows: dict[str, dict[str, Any]] = {}
    for line in evaluations_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[str(row["sample_id"])] = row
    return rows


def _pairwise_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_rows = _load_evaluations(before)
    after_rows = _load_evaluations(after)
    common_ids = sorted(set(before_rows) & set(after_rows))
    result: dict[str, Any] = {"common_sample_count": len(common_ids)}
    for key in ("json_valid", "compile_success", "behavior_success"):
        gains = sum(
            not before_rows[item][key] and bool(after_rows[item][key])
            for item in common_ids
        )
        losses = sum(
            bool(before_rows[item][key]) and not after_rows[item][key]
            for item in common_ids
        )
        result[f"{key}_gains"] = gains
        result[f"{key}_losses"] = losses
    for key in ("readability_score", "naming_score"):
        if common_ids:
            result[f"{key}_mean_delta"] = sum(
                float(after_rows[item][key]) - float(before_rows[item][key])
                for item in common_ids
            ) / len(common_ids)
        else:
            result[f"{key}_mean_delta"] = 0.0
    return result


def _render_score_table(manifests: dict[str, dict[str, Any]]) -> str:
    lines = [
        "| Condition | Score | Behavior | Compile | JSON | Readability | Naming |",
        "|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, manifest in manifests.items():
        metrics = manifest["metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    f"{_score(metrics):.4f}",
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


def _write_report(
    *,
    run_dir: Path,
    args: argparse.Namespace,
    results: dict[str, dict[str, Any]],
) -> tuple[Path, Path]:
    manifests = {label: item["manifest"] for label, item in results.items()}
    systems = build_target_comparison_systems_from_manifests(
        manifests,
        pinned_columns=ABLATED_COLUMNS,
    )
    deltas = {
        "sft_no_thinking -> sft_thinking": _pairwise_delta(
            manifests["sft_no_thinking"], manifests["sft_thinking"]
        ),
        "gdpo_no_thinking -> gdpo_thinking": _pairwise_delta(
            manifests["gdpo_no_thinking"], manifests["gdpo_thinking"]
        ),
        "sft_no_thinking -> gdpo_no_thinking": _pairwise_delta(
            manifests["sft_no_thinking"], manifests["gdpo_no_thinking"]
        ),
        "sft_thinking -> gdpo_thinking": _pairwise_delta(
            manifests["sft_thinking"], manifests["gdpo_thinking"]
        ),
    }
    markdown = "\n".join(
        [
            "# Thinking Ablation Report",
            "",
            "## Configuration",
            "",
            f"- SFT profile: `{args.sft_profile}`",
            f"- GDPO profile: `{args.grpo_profile}`",
            f"- Split: `{args.split}`",
            f"- Prompt profile: `{args.prompt_profile}`",
            f"- Max new tokens: `{args.max_new_tokens}`",
            f"- Temperature: `{args.temperature}`",
            f"- Sample limit: `{args.sample_limit}`",
            "",
            "## Weighted Scores",
            "",
            _render_score_table(manifests),
            "",
            "## Target Metrics",
            "",
            render_target_comparison_table(systems, columns=ABLATED_COLUMNS),
            "",
            "## Pairwise Deltas",
            "",
            _render_delta_table(deltas),
            "",
            "## Eval Manifests",
            "",
            *[
                f"- {label}: `{item['manifest_path']}`"
                for label, item in results.items()
            ],
            "",
        ]
    )
    report_markdown = run_dir / "thinking_ablation_report.md"
    report_json = run_dir / "thinking_ablation_report.json"
    report_markdown.write_text(markdown, encoding="utf-8")
    _write_json(
        report_json,
        {
            "config": {
                "sft_profile": args.sft_profile,
                "grpo_profile": args.grpo_profile,
                "split": args.split,
                "prompt_profile": args.prompt_profile,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "sample_limit": args.sample_limit,
            },
            "metric_weights": METRIC_WEIGHTS,
            "scores": {
                label: _score(item["manifest"]["metrics"]) for label, item in results.items()
            },
            "deltas": deltas,
            "eval_manifests": {
                label: item["manifest_path"] for label, item in results.items()
            },
            "systems": systems,
        },
    )
    return report_markdown, report_json


def main() -> None:
    args = parse_args()
    app_config = load_app_config(ROOT, args.app_profile)
    paths = ProjectPaths.from_config(ROOT, app_config)
    run_id = _run_id()
    run_dir = paths.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    env = _pythonpath_env()
    conditions = (
        ("sft_no_thinking", "sft", False),
        ("sft_thinking", "sft", True),
        ("gdpo_no_thinking", "grpo", False),
        ("gdpo_thinking", "grpo", True),
    )
    results = {
        label: _run_condition(
            run_dir=run_dir,
            env=env,
            args=args,
            label=label,
            stage=stage,
            thinking=thinking,
        )
        for label, stage, thinking in conditions
    }
    report_markdown, report_json = _write_report(run_dir=run_dir, args=args, results=results)
    _write_json(
        run_dir / "thinking_ablation_manifest.json",
        {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "report_markdown": str(report_markdown),
            "report_json": str(report_json),
            "eval_manifests": {
                label: item["manifest_path"] for label, item in results.items()
            },
        },
    )
    print(f"Thinking ablation report: {report_markdown}")
    print(f"Thinking ablation json: {report_json}")


if __name__ == "__main__":
    main()
