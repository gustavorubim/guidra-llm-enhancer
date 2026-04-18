from __future__ import annotations

import re

_CONTROL_KEYWORDS = {"if", "for", "while", "switch", "return", "sizeof"}
_FUNCTION_SIGNATURE_LINE = re.compile(
    r"^\s*(?:[A-Za-z_][\w\s\*]*?\s+)?(?P<name>[A-Za-z_]\w*)\s*\([^;{}]*\)\s*\{\s*$"
)
_FUNCTION_HEADER = re.compile(
    r"^\s*(?P<signature>(?:[A-Za-z_][\w\s\*]*?\s+)?(?P<name>[A-Za-z_]\w*)\s*\((?P<params>[^;{}]*)\))\s*(?:\{|$)"
)
_CALL_IDENTIFIER_PATTERN = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
_CALLING_CONVENTION_PATTERN = re.compile(
    r"\b__(?:cdecl|stdcall|fastcall|thiscall|vectorcall)\b"
)


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


def extract_function_signature(code: str) -> str | None:
    cleaned = strip_leading_non_code_lines(code)
    if not cleaned:
        return None
    prefix = cleaned.split("{", 1)[0]
    header = " ".join(part.strip() for part in prefix.splitlines() if part.strip())
    match = _FUNCTION_HEADER.match(header)
    if match is None:
        return None
    name = match.group("name")
    if name in _CONTROL_KEYWORDS:
        return None
    return match.group("signature").strip()


def function_name_from_signature_text(signature: str) -> str | None:
    match = _FUNCTION_HEADER.match(signature.strip())
    if match is None:
        return None
    name = match.group("name")
    return None if name in _CONTROL_KEYWORDS else name


def parameter_count_from_signature(signature: str) -> int | None:
    match = _FUNCTION_HEADER.match(signature.strip())
    if match is None:
        return None
    params = match.group("params").strip()
    if not params or params == "void":
        return 0
    return len([param for param in params.split(",") if param.strip()])


def parameter_types_from_signature(signature: str) -> list[str] | None:
    match = _FUNCTION_HEADER.match(signature.strip())
    if match is None:
        return None
    params = match.group("params").strip()
    if not params or params == "void":
        return []

    types: list[str] = []
    for raw_param in params.split(","):
        param = raw_param.strip()
        if not param:
            continue
        param = _CALLING_CONVENTION_PATTERN.sub("", param)
        param = re.sub(r"\s+", " ", param).strip()
        param = re.sub(r"\s*\[[^\]]*\]", "[]", param)
        match_name = re.match(
            r"^(?P<prefix>.*?)(?P<name>[A-Za-z_]\w*)(?P<suffix>\s*(?:\[\])*)$",
            param,
        )
        if match_name is not None and match_name.group("prefix").strip():
            param_type = f"{match_name.group('prefix').strip()}{match_name.group('suffix')}"
        else:
            param_type = param
        param_type = re.sub(r"\s*\*\s*", "*", param_type)
        param_type = re.sub(r"\s+", " ", param_type).strip()
        types.append(param_type)
    return types


def parameter_names_from_signature(signature: str) -> list[str] | None:
    match = _FUNCTION_HEADER.match(signature.strip())
    if match is None:
        return None
    params = match.group("params").strip()
    if not params or params == "void":
        return []

    names: list[str] = []
    for raw_param in params.split(","):
        param = raw_param.strip()
        if not param or param == "...":
            continue
        param = _CALLING_CONVENTION_PATTERN.sub("", param)
        param = re.sub(r"\s+", " ", param).strip()
        param = re.sub(r"\s*\[[^\]]*\]", "[]", param)
        match_name = re.match(
            r"^(?P<prefix>.*?)(?P<name>[A-Za-z_]\w*)(?P<suffix>\s*(?:\[\])*)$",
            param,
        )
        if match_name is None:
            continue
        names.append(match_name.group("name"))
    return names


def return_type_from_signature(signature: str) -> str | None:
    stripped = signature.strip()
    match = _FUNCTION_HEADER.match(stripped)
    if match is None:
        return None
    return_prefix = stripped[: match.start("name")].strip()
    if not return_prefix:
        return None
    normalized = _CALLING_CONVENTION_PATTERN.sub("", return_prefix)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized or None


def normalize_function_signature(signature: str) -> str:
    normalized = _CALLING_CONVENTION_PATTERN.sub("", signature)
    normalized = re.sub(r"\s*\*\s*", "*", normalized)
    normalized = re.sub(r"\s*,\s*", ", ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"\(\s*", "(", normalized)
    normalized = re.sub(r"\s*\)", ")", normalized)
    return normalized.strip()


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
