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


PromptValue = str | list[dict[str, str]]


def prompt_from_record(record: dict[str, Any]) -> PromptValue:
    prompt_messages = record.get("prompt_messages")
    if isinstance(prompt_messages, list) and prompt_messages:
        messages: list[dict[str, str]] = []
        for message in prompt_messages:
            if not isinstance(message, dict):
                return str(record["prompt"])
            role = message.get("role")
            content = message.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                return str(record["prompt"])
            messages.append({"role": role, "content": content})
        return messages
    return str(record["prompt"])


def reward_fields_from_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_type": record.get("task_type", "full_clarify"),
        "source_function_name": record.get("source_function_name", ""),
        "raw_code": record.get("raw_code", ""),
        "compile_reference_source": record.get(
            "compile_reference_source", record.get("target_clean_code", "")
        ),
        "target_clean_code": record.get("target_clean_code", ""),
        "target_renamings": record.get("target_renamings", "{}"),
        "allowed_imports": record.get("allowed_imports", "[]"),
        "allowed_callees": record.get("allowed_callees", "[]"),
        "compiler_executable": record.get("compiler_executable"),
        "tests_ref": record.get("tests_ref") or "",
    }
