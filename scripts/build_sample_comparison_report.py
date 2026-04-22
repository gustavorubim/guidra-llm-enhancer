from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from decomp_clarifier.evaluation.sample_comparison_report import (  # noqa: E402
    build_sample_comparison_report,
)
from decomp_clarifier.paths import ProjectPaths  # noqa: E402
from decomp_clarifier.settings import load_app_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a random side-by-side report comparing original source, decompiled output, "
            "prompt baselines, SFT, and GRPO on a shared sample subset."
        )
    )
    parser.add_argument("--app-profile", default="default")
    parser.add_argument("--split", default="val")
    parser.add_argument("--sample-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--baseline-predictions", type=Path, default=None)
    parser.add_argument("--sft-predictions", type=Path, default=None)
    parser.add_argument("--grpo-predictions", type=Path, default=None)
    parser.add_argument("--original-system", default="prompt_only_cleanup")
    parser.add_argument("--qwen-system", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = ProjectPaths.discover(start=ROOT)
    app_config = load_app_config(root, name=args.app_profile)
    paths = ProjectPaths.from_config(root, app_config)
    paths.ensure()

    artifacts = build_sample_comparison_report(
        paths=paths,
        split=args.split,
        sample_count=args.sample_count,
        seed=args.seed,
        baseline_predictions_path=paths.resolve(args.baseline_predictions)
        if args.baseline_predictions
        else None,
        sft_predictions_path=paths.resolve(args.sft_predictions) if args.sft_predictions else None,
        grpo_predictions_path=paths.resolve(args.grpo_predictions)
        if args.grpo_predictions
        else None,
        original_system=args.original_system,
        qwen_system=args.qwen_system,
    )

    print(f"Manifest: {artifacts.manifest_path}")
    print(f"Markdown report: {artifacts.markdown_path}")
    print(f"JSON report: {artifacts.json_path}")


if __name__ == "__main__":
    main()
