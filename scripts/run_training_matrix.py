from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from decomp_clarifier.adapters.subprocess_utils import run_subprocess  # noqa: E402
from decomp_clarifier.evaluation.target_comparison import (  # noqa: E402
    BASELINE_COLUMNS,
    build_target_comparison_systems_from_manifests,
    load_checkpoint_eval_manifest,
    render_target_comparison_table,
)
from decomp_clarifier.paths import ProjectPaths  # noqa: E402
from decomp_clarifier.settings import load_app_config  # noqa: E402

SFT_PROFILES = (
    "sft_qwen35_2b",
    "sft_qwen35_4b",
    "sft_gemma4_e2b_it",
    "sft_gemma4_e4b_it",
)
GRPO_PROFILES = (
    "grpo_qwen35_2b",
    "grpo_qwen35_4b",
    "grpo_gemma4_e2b_it",
    "grpo_gemma4_e4b_it",
)
SUMMARY_PROFILE_COLUMNS = [
    "sft_qwen35_2b",
    "grpo_qwen35_2b",
    "sft_qwen35_4b",
    "grpo_qwen35_4b",
    "sft_gemma4_e2b_it",
    "grpo_gemma4_e2b_it",
    "sft_gemma4_e4b_it",
    "grpo_gemma4_e4b_it",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the full 2B/4B Qwen + Gemma SFT/GRPO matrix, run all checkpoint "
            "evaluations, and refresh the combined performance summary."
        )
    )
    parser.add_argument("--app-profile", default="default")
    parser.add_argument("--split", default="val")
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--inspection-sample-count", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def _run_id() -> str:
    return "train-matrix-" + datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")


def _pythonpath_env() -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    root_src = str(ROOT / "src")
    env["PYTHONPATH"] = root_src if not existing else os.pathsep.join([root_src, existing])
    return env


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _parse_manifest_path(stdout: str) -> str:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError("CLI step did not emit a manifest path")
    return lines[-1]


def _run_cli_step(
    *,
    root: Path,
    run_dir: Path,
    env: dict[str, str],
    step_index: int,
    label: str,
    cli_args: list[str],
) -> dict[str, Any]:
    command = [sys.executable, "-m", "decomp_clarifier.cli", *cli_args]
    result = run_subprocess(command, cwd=root, env=env)
    stdout_path = run_dir / "steps" / f"{step_index:02d}-{label}.stdout.log"
    stderr_path = run_dir / "steps" / f"{step_index:02d}-{label}.stderr.log"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")

    step_record: dict[str, Any] = {
        "label": label,
        "command": command,
        "returncode": result.returncode,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }
    if result.returncode != 0:
        step_record["status"] = "failed"
        step_record["stdout_tail"] = "\n".join(result.stdout.splitlines()[-20:])
        step_record["stderr_tail"] = "\n".join(result.stderr.splitlines()[-20:])
        raise RuntimeError(json.dumps(step_record, indent=2))

    step_record["status"] = "completed"
    step_record["manifest_path"] = _parse_manifest_path(result.stdout)
    return step_record


def _build_summary_payload(
    manifests_by_profile: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    systems = build_target_comparison_systems_from_manifests(manifests_by_profile)
    columns = [*BASELINE_COLUMNS, *SUMMARY_PROFILE_COLUMNS]
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "columns": columns,
        "eval_manifests": {
            profile: manifest.get("__manifest_path", "") for profile, manifest in manifests_by_profile.items()
        },
        "systems": systems,
        "table_markdown": render_target_comparison_table(systems, columns=columns),
    }


def main() -> None:
    args = parse_args()
    root = ProjectPaths.discover(start=ROOT)
    app_config = load_app_config(root, name=args.app_profile)
    paths = ProjectPaths.from_config(root, app_config)
    paths.ensure()

    run_id = _run_id()
    run_dir = paths.run_dir(run_id)
    env = _pythonpath_env()
    manifest_path = run_dir / "matrix_run_manifest.json"

    state: dict[str, Any] = {
        "run_id": run_id,
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "root": str(root),
        "app_profile": args.app_profile,
        "split": args.split,
        "sample_limit": args.sample_limit,
        "inspection_sample_count": args.inspection_sample_count,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "sft_profiles": list(SFT_PROFILES),
        "grpo_profiles": list(GRPO_PROFILES),
        "steps": [],
    }
    _write_json(manifest_path, state)

    try:
        step_index = 1
        for profile in SFT_PROFILES:
            step = _run_cli_step(
                root=root,
                run_dir=run_dir,
                env=env,
                step_index=step_index,
                label=f"train-{profile}",
                cli_args=["train-sft", "--training-profile", profile, "--app-profile", args.app_profile],
            )
            state["steps"].append(step)
            _write_json(manifest_path, state)
            step_index += 1

        for profile in GRPO_PROFILES:
            step = _run_cli_step(
                root=root,
                run_dir=run_dir,
                env=env,
                step_index=step_index,
                label=f"train-{profile}",
                cli_args=["train-grpo", "--training-profile", profile, "--app-profile", args.app_profile],
            )
            state["steps"].append(step)
            _write_json(manifest_path, state)
            step_index += 1

        eval_manifest_paths: dict[str, str] = {}
        for profile in SFT_PROFILES:
            cli_args = [
                "eval-sft-checkpoint",
                "--training-profile",
                profile,
                "--split",
                args.split,
                "--inspection-sample-count",
                str(args.inspection_sample_count),
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--temperature",
                str(args.temperature),
                "--app-profile",
                args.app_profile,
            ]
            if args.sample_limit is not None:
                cli_args.extend(["--sample-limit", str(args.sample_limit)])
            step = _run_cli_step(
                root=root,
                run_dir=run_dir,
                env=env,
                step_index=step_index,
                label=f"eval-{profile}",
                cli_args=cli_args,
            )
            state["steps"].append(step)
            eval_manifest_paths[profile] = step["manifest_path"]
            _write_json(manifest_path, state)
            step_index += 1

        for profile in GRPO_PROFILES:
            cli_args = [
                "eval-grpo-checkpoint",
                "--training-profile",
                profile,
                "--split",
                args.split,
                "--inspection-sample-count",
                str(args.inspection_sample_count),
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--temperature",
                str(args.temperature),
                "--app-profile",
                args.app_profile,
            ]
            if args.sample_limit is not None:
                cli_args.extend(["--sample-limit", str(args.sample_limit)])
            step = _run_cli_step(
                root=root,
                run_dir=run_dir,
                env=env,
                step_index=step_index,
                label=f"eval-{profile}",
                cli_args=cli_args,
            )
            state["steps"].append(step)
            eval_manifest_paths[profile] = step["manifest_path"]
            _write_json(manifest_path, state)
            step_index += 1

        manifests_by_profile: dict[str, dict[str, Any]] = {}
        for profile, path in eval_manifest_paths.items():
            manifest = load_checkpoint_eval_manifest(paths.resolve(path))
            manifest["__manifest_path"] = str(paths.resolve(path))
            manifests_by_profile[profile] = manifest

        summary_payload = _build_summary_payload(manifests_by_profile)
        matrix_markdown_path = paths.reports_dir / "model_matrix_summary.md"
        matrix_json_path = paths.reports_dir / "model_matrix_summary.json"
        target_markdown_path = paths.reports_dir / "target_comparison_table.md"
        target_json_path = paths.reports_dir / "target_comparison_table.json"
        matrix_markdown_path.write_text(
            summary_payload["table_markdown"] + "\n",
            encoding="utf-8",
        )
        _write_json(matrix_json_path, summary_payload)
        target_markdown_path.write_text(
            summary_payload["table_markdown"] + "\n",
            encoding="utf-8",
        )
        _write_json(target_json_path, summary_payload)

        state["summary_report"] = {
            "model_matrix_markdown": str(matrix_markdown_path),
            "model_matrix_json": str(matrix_json_path),
            "target_comparison_markdown": str(target_markdown_path),
            "target_comparison_json": str(target_json_path),
        }
        state["completed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        state["status"] = "completed"
        _write_json(manifest_path, state)

        print(f"Matrix run manifest: {manifest_path}")
        print(f"Summary markdown: {matrix_markdown_path}")
        print(f"Summary json: {matrix_json_path}")
    except Exception as exc:  # noqa: BLE001
        state["completed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        state["status"] = "failed"
        state["error"] = str(exc)
        _write_json(manifest_path, state)
        raise


if __name__ == "__main__":
    main()
