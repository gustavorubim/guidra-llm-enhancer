from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_rl_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def prompt_from_record(record: dict[str, Any]) -> str:
    return record["prompt"]


def reward_fields_from_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_function_name": record.get("source_function_name", ""),
        "raw_code": record.get("raw_code", ""),
        "compile_reference_source": record.get(
            "compile_reference_source", record.get("target_clean_code", "")
        ),
        "target_clean_code": record.get("target_clean_code", ""),
        "target_renamings": record.get("target_renamings", "{}"),
        "allowed_imports": record.get("allowed_imports", "[]"),
        "allowed_callees": record.get("allowed_callees", "[]"),
    }
