from __future__ import annotations

import json

from decomp_clarifier.schemas.dataset import ChatMessage, FunctionDatasetSample


def format_prompt(sample: FunctionDatasetSample) -> str:
    return "\n".join(
        [
            "You are a binary-grounded code clarification assistant.",
            f"Task: {sample.task_type}",
            "",
            "Decompiler:",
            f"<code>\n{sample.ghidra_decompiled_code}\n</code>",
            "",
            "Assembly:",
            f"<asm>\n{sample.assembly}\n</asm>",
            "",
            f"Strings: {json.dumps(sample.strings)}",
            f"Imports: {json.dumps(sample.imports)}",
            f"Callers: {json.dumps(sample.callers)}",
            f"Callees: {json.dumps(sample.callees)}",
            f"Semantic summary: {sample.semantic_summary}",
            "",
            "Return exactly one JSON object with keys summary, confidence, renamings, cleaned_c.",
            "Do not include markdown fences, commentary, XML tags, or <think> blocks.",
            "JSON:",
        ]
    )


def prompt_messages(prompt: str) -> list[ChatMessage]:
    return [ChatMessage(role="user", content=prompt)]


def completion_messages(response_json: str) -> list[ChatMessage]:
    return [ChatMessage(role="assistant", content=response_json)]


def format_rl_prompt(sample: FunctionDatasetSample) -> str:
    return format_prompt(sample)
