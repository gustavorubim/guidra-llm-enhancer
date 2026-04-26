from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from decomp_clarifier.evaluation.target_comparison import (  # noqa: E402
    BASELINE_COLUMNS,
    build_target_comparison_systems_from_manifests,
    load_checkpoint_eval_manifest,
    render_target_comparison_table,
)
from decomp_clarifier.paths import ProjectPaths  # noqa: E402
from decomp_clarifier.settings import load_app_config  # noqa: E402

SUMMARY_PROFILE_COLUMNS = [
    "sft_qwen35_2b",
    "grpo_qwen35_2b_champion_300",
    "sft_gemma4_e2b_it",
    "grpo_gemma4_e2b_it",
    "sft_qwen35_4b",
    "grpo_qwen35_4b",
    "sft_gemma4_e4b_it",
    "grpo_gemma4_e4b_it",
]


def _parse_eval_manifest(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError(
            "Expected --eval-manifest in the form profile=path/to/checkpoint_eval_manifest.json"
        )
    return label.strip(), Path(raw_path.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the matrix summary and target comparison reports from labeled "
            "checkpoint eval manifests."
        )
    )
    parser.add_argument("--app-profile", default="default")
    parser.add_argument(
        "--eval-manifest",
        action="append",
        type=_parse_eval_manifest,
        default=[],
        help="Append a labeled eval manifest as profile=path/to/checkpoint_eval_manifest.json.",
    )
    return parser.parse_args()


def _build_summary_payload(manifests_by_profile: dict[str, dict[str, object]]) -> dict[str, object]:
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
    if not args.eval_manifest:
        raise ValueError("At least one --eval-manifest profile=path value is required.")

    root = ProjectPaths.discover(start=ROOT)
    app_config = load_app_config(root, name=args.app_profile)
    paths = ProjectPaths.from_config(root, app_config)
    paths.ensure()

    manifests_by_profile: dict[str, dict[str, object]] = {}
    for profile, raw_path in args.eval_manifest:
        resolved_path = paths.resolve(raw_path)
        manifest = load_checkpoint_eval_manifest(resolved_path)
        manifest["__manifest_path"] = str(resolved_path)
        manifests_by_profile[profile] = manifest

    summary_payload = _build_summary_payload(manifests_by_profile)
    matrix_markdown_path = paths.reports_dir / "model_matrix_summary.md"
    matrix_json_path = paths.reports_dir / "model_matrix_summary.json"
    target_markdown_path = paths.reports_dir / "target_comparison_table.md"
    target_json_path = paths.reports_dir / "target_comparison_table.json"

    matrix_markdown_path.write_text(summary_payload["table_markdown"] + "\n", encoding="utf-8")
    matrix_json_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")
    target_markdown_path.write_text(summary_payload["table_markdown"] + "\n", encoding="utf-8")
    target_json_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Model matrix markdown: {matrix_markdown_path}")
    print(f"Model matrix json: {matrix_json_path}")
    print(f"Target comparison markdown: {target_markdown_path}")
    print(f"Target comparison json: {target_json_path}")


if __name__ == "__main__":
    main()
