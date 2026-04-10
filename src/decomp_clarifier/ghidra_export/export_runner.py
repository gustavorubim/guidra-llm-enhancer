from __future__ import annotations

import json
from pathlib import Path

from decomp_clarifier.adapters.ghidra_headless import GhidraHeadlessAdapter
from decomp_clarifier.schemas.compiler import CompileManifest


class GhidraExportRunner:
    def __init__(self, adapter: GhidraHeadlessAdapter) -> None:
        self.adapter = adapter

    def export_manifest(self, manifest: CompileManifest, output_root: Path) -> Path:
        if not manifest.binaries:
            raise ValueError(
                f"compile manifest for {manifest.project_id} does not contain binaries"
            )
        binary_path = Path(manifest.binaries[0].path)
        export_dir = output_root / manifest.project_id
        result = self.adapter.run(
            binary_path=binary_path,
            output_dir=export_dir,
            project_name=f"proj_{manifest.project_id}",
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr or result.stdout or "Ghidra export failed")
        (export_dir / "ghidra_headless.log").write_text(
            "\n".join([result.stdout, result.stderr]).strip(),
            encoding="utf-8",
        )
        required_paths = [
            export_dir / "project_manifest.json",
            export_dir / "functions.jsonl",
        ]
        missing = [path.name for path in required_paths if not path.exists()]
        if missing:
            raise RuntimeError(
                f"Ghidra export did not produce required artifacts: {', '.join(missing)}"
            )
        (export_dir / "source_compile_manifest.json").write_text(
            json.dumps(manifest.model_dump(mode="python"), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return export_dir
