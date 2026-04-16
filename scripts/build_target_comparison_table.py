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
    TARGET_COLUMNS,
    TARGET_METRICS,
    build_target_comparison_systems,
    find_latest_checkpoint_eval_manifest,
    load_checkpoint_eval_manifest,
    render_target_comparison_table,
)
from decomp_clarifier.paths import ProjectPaths  # noqa: E402
from decomp_clarifier.settings import load_app_config  # noqa: E402


def _resolve_output_path(paths: ProjectPaths, value: Path | None, default: Path) -> Path:
    resolved = default if value is None else paths.resolve(value)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _parse_extra_manifest(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError(
            "Expected --extra-manifest in the form label=path/to/checkpoint_eval_manifest.json"
        )
    return label.strip(), Path(raw_path.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a single markdown comparison table that merges existing baseline, "
            "SFT checkpoint, and GRPO checkpoint metrics."
        )
    )
    parser.add_argument(
        "--sft-manifest",
        type=Path,
        help="Path to checkpoint_eval_manifest.json from an eval-sft-checkpoint run.",
    )
    parser.add_argument(
        "--grpo-manifest",
        type=Path,
        help="Path to checkpoint_eval_manifest.json from an eval-grpo-checkpoint run.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        help="Markdown output path. Defaults to artifacts/reports/target_comparison_table.md.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="JSON output path. Defaults to artifacts/reports/target_comparison_table.json.",
    )
    parser.add_argument(
        "--extra-manifest",
        action="append",
        type=_parse_extra_manifest,
        default=[],
        help="Append an extra labeled checkpoint column as label=path/to/checkpoint_eval_manifest.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = ProjectPaths.discover(start=ROOT)
    app_config = load_app_config(root, name="default")
    paths = ProjectPaths.from_config(root, app_config)
    paths.ensure()

    sft_manifest_path = (
        paths.resolve(args.sft_manifest)
        if args.sft_manifest is not None
        else find_latest_checkpoint_eval_manifest(root, "sft")
    )
    grpo_manifest_path = (
        paths.resolve(args.grpo_manifest)
        if args.grpo_manifest is not None
        else find_latest_checkpoint_eval_manifest(root, "grpo")
    )

    sft_manifest = load_checkpoint_eval_manifest(sft_manifest_path, expected_stage="sft")
    grpo_manifest = load_checkpoint_eval_manifest(grpo_manifest_path, expected_stage="grpo")
    extra_manifest_paths: dict[str, Path] = {}
    extra_manifests = {}
    for label, extra_path in args.extra_manifest:
        if label in TARGET_COLUMNS:
            raise ValueError(f"Extra manifest label {label!r} conflicts with a built-in column.")
        if label in extra_manifest_paths:
            raise ValueError(f"Duplicate extra manifest label: {label!r}")
        resolved_extra_path = paths.resolve(extra_path)
        extra_manifest_paths[label] = resolved_extra_path
        extra_manifests[label] = load_checkpoint_eval_manifest(resolved_extra_path)
    columns = [*TARGET_COLUMNS, *extra_manifest_paths]
    systems = build_target_comparison_systems(
        sft_manifest,
        grpo_manifest,
        extra_manifests=extra_manifests,
    )
    table = render_target_comparison_table(systems, columns=columns)

    markdown_path = _resolve_output_path(
        paths,
        args.output_md,
        paths.reports_dir / "target_comparison_table.md",
    )
    json_path = _resolve_output_path(
        paths,
        args.output_json,
        paths.reports_dir / "target_comparison_table.json",
    )

    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "sources": {
            "sft_manifest": str(sft_manifest_path),
            "grpo_manifest": str(grpo_manifest_path),
            "extra_manifests": {
                label: str(path) for label, path in extra_manifest_paths.items()
            },
        },
        "columns": columns,
        "metrics": TARGET_METRICS,
        "systems": systems,
        "table_markdown": table,
    }

    markdown_path.write_text(table + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(table)
    print()
    print(f"Markdown: {markdown_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
