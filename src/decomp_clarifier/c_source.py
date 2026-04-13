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


def iter_function_starts(content: str) -> list[tuple[int, str]]:
    lines = content.splitlines(keepends=True)
    offsets: list[int] = []
    offset = 0
    for line in lines:
        offsets.append(offset)
        offset += len(line)

    starts: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        stripped = line.rstrip("\r\n").strip()
        if not stripped or stripped.startswith(("#", "//", "/*", "*", "*/")):
            continue
        candidate = stripped
        if not looks_like_function_signature_line(candidate) and "(" in stripped:
            candidate_parts = [stripped]
            for lookahead in range(index + 1, len(lines)):
                next_part = lines[lookahead].rstrip("\r\n").strip()
                if not next_part:
                    break
                candidate_parts.append(next_part)
                candidate = " ".join(part for part in candidate_parts if part)
                if "{" in next_part or ";" in next_part:
                    break
        if not looks_like_function_signature_line(candidate):
            continue
        name = function_name_from_signature_line(candidate)
        if name is None:
            continue
        starts.append((offsets[index], name))
    return starts


def slice_function(content: str, start_index: int) -> str:
    brace_index = content.find("{", start_index)
    if brace_index == -1:
        return ""
    depth = 0
    for index in range(brace_index, len(content)):
        character = content[index]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return content[start_index : index + 1]
    return content[start_index:]


def replace_function_definition(
    source: str, function_name: str | None, replacement_code: str
) -> str | None:
    if function_name is None:
        return None
    for start_index, name in iter_function_starts(source):
        if name != function_name:
            continue
        original = slice_function(source, start_index)
        if not original:
            return None
        end_index = start_index + len(original)
        replacement = replacement_code.strip()
        return f"{source[:start_index]}{replacement}{source[end_index:]}"
    return None


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
