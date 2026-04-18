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


def prompt_completion_from_record(record: dict[str, Any]) -> dict[str, Any]:
    prompt_messages = record.get("prompt_messages")
    completion_messages = record.get("completion_messages")
    if prompt_messages and completion_messages:
        return {
            "prompt": prompt_messages,
            "completion": completion_messages,
        }
    return {"text": combine_prompt_and_response(record)}
