from __future__ import annotations

import re

_CONTROL_KEYWORDS = {"if", "for", "while", "switch", "return", "sizeof"}
_FUNCTION_SIGNATURE_LINE = re.compile(
    r"^\s*(?:[A-Za-z_][\w\s\*]*?\s+)?(?P<name>[A-Za-z_]\w*)\s*\([^;{}]*\)\s*\{\s*$"
)
_CALL_IDENTIFIER_PATTERN = re.compile(r"\b([A-Za-z_]\w*)\s*\(")


def looks_like_function_signature_line(line: str) -> bool:
    match = _FUNCTION_SIGNATURE_LINE.match(line)
    if match is None:
        return False
    return match.group("name") not in _CONTROL_KEYWORDS


def function_name_from_signature_line(line: str) -> str | None:
    match = _FUNCTION_SIGNATURE_LINE.match(line)
    if match is None:
        return None
    name = match.group("name")
    return None if name in _CONTROL_KEYWORDS else name


def strip_leading_non_code_lines(code: str) -> str:
    lines = code.splitlines()
    for index, line in enumerate(lines):
        if looks_like_function_signature_line(line):
            return "\n".join(lines[index:]).strip()
    return code.strip()


def extract_called_functions(code: str) -> list[str]:
    cleaned = strip_leading_non_code_lines(code)
    brace_index = cleaned.find("{")
    body = cleaned[brace_index + 1 :] if brace_index != -1 else cleaned
    current_function = None
    signature_line = cleaned.splitlines()[0] if cleaned.splitlines() else ""
    current_function = function_name_from_signature_line(signature_line)

    calls: list[str] = []
    for name in _CALL_IDENTIFIER_PATTERN.findall(body):
        if name in _CONTROL_KEYWORDS or name == current_function:
            continue
        if name not in calls:
            calls.append(name)
    return calls
