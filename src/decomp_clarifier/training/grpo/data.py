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
