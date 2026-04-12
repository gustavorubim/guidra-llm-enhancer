from __future__ import annotations

import json

from decomp_clarifier.schemas.model_io import ClarifiedFunctionOutput


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


def normalize_output_with_status(text: str) -> tuple[ClarifiedFunctionOutput, bool]:
    json_fragment = extract_json_object(text)
    if json_fragment is None:
        return _fallback_output(text), False
    try:
        output = ClarifiedFunctionOutput.model_validate(json.loads(json_fragment))
        return output, text.strip() == json_fragment.strip()
    except Exception:  # noqa: BLE001 - tolerate malformed generations during inference
        return _fallback_output(text), False


def normalize_output(text: str) -> ClarifiedFunctionOutput:
    return normalize_output_with_status(text)[0]
