from __future__ import annotations

import json
from typing import Literal

from decomp_clarifier.schemas.model_io import ClarifiedFunctionOutput

SchemaStatus = Literal["invalid", "extractable", "strict"]
_THINKING_CLOSE_TAG = "</think>"


def extract_json_object(text: str) -> str | None:
    start_index: int | None = None
    depth = 0
    in_string = False
    escaped = False

    for index, character in enumerate(text):
        if start_index is None:
            if character == "{":
                start_index = index
                depth = 1
            continue
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[start_index : index + 1]
    return None


def _fallback_output(text: str) -> ClarifiedFunctionOutput:
    return ClarifiedFunctionOutput(
        summary="",
        confidence=0.0,
        renamings={},
        cleaned_c=text.strip(),
    )


def strip_thinking_prefix(text: str) -> str:
    stripped = text.strip()
    close_index = stripped.rfind(_THINKING_CLOSE_TAG)
    if close_index < 0:
        return stripped
    return stripped[close_index + len(_THINKING_CLOSE_TAG) :].strip()


def normalize_output_with_schema_status(
    text: str, *, strip_thinking: bool = False
) -> tuple[ClarifiedFunctionOutput, SchemaStatus]:
    normalized_text = strip_thinking_prefix(text) if strip_thinking else text
    json_fragment = extract_json_object(normalized_text)
    if json_fragment is None:
        return _fallback_output(normalized_text), "invalid"
    try:
        output = ClarifiedFunctionOutput.model_validate(json.loads(json_fragment))
        if normalized_text.strip() == json_fragment.strip():
            return output, "strict"
        return output, "extractable"
    except Exception:  # noqa: BLE001 - tolerate malformed generations during inference
        return _fallback_output(normalized_text), "invalid"


def normalize_output_with_status(
    text: str, *, strip_thinking: bool = False
) -> tuple[ClarifiedFunctionOutput, bool]:
    output, schema_status = normalize_output_with_schema_status(
        text, strip_thinking=strip_thinking
    )
    return output, schema_status == "strict"


def normalize_output(text: str) -> ClarifiedFunctionOutput:
    return normalize_output_with_status(text)[0]
