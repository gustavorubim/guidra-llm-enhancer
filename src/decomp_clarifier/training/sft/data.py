from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_sft_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def combine_prompt_and_response(record: dict[str, Any], *, eos_token: str | None = None) -> str:
    suffix = eos_token or ""
    return f"{record['prompt']}\n{record['response_json']}{suffix}"
