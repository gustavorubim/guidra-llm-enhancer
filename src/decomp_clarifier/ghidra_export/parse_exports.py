from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from decomp_clarifier.schemas.ghidra import GhidraFunctionRow, GhidraProjectManifest


@dataclass(frozen=True)
class ParsedGhidraProject:
    manifest: GhidraProjectManifest
    functions: list[GhidraFunctionRow]


def parse_ghidra_export_dir(path: Path) -> ParsedGhidraProject:
    manifest = GhidraProjectManifest.model_validate_json(
        (path / "project_manifest.json").read_text(encoding="utf-8")
    )
    functions: list[GhidraFunctionRow] = []
    for line in (path / "functions.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            functions.append(GhidraFunctionRow.model_validate(json.loads(line)))
    return ParsedGhidraProject(manifest=manifest, functions=functions)
