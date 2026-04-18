from __future__ import annotations

import keyword
import re

from decomp_clarifier.c_source import (
    extract_function_signature,
    function_name_from_signature_text,
    parameter_names_from_signature,
    strip_leading_non_code_lines,
)

PLACEHOLDER_PATTERN = re.compile(
    r"\b(?:param_\d+|local_[0-9A-Fa-f]+|iVar\d+|uVar\d+|puVar\d+|FUN_[0-9A-Fa-f]+)\b"
)
IDENTIFIER_PATTERN = re.compile(r"\b[A-Za-z_]\w*\b")
BLOCK_COMMENT_PATTERN = re.compile(r"/\*.*?\*/", re.DOTALL)
LINE_COMMENT_PATTERN = re.compile(r"//.*?$", re.MULTILINE)
FUNCTION_PLACEHOLDER_PATTERN = re.compile(r"\bFUN_[0-9A-Fa-f]+\b")
PARAM_PLACEHOLDER_PATTERN = re.compile(r"\bparam_\d+\b")


def strip_comments(code: str) -> str:
    without_block = BLOCK_COMMENT_PATTERN.sub(" ", code)
    return LINE_COMMENT_PATTERN.sub(" ", without_block)


def extract_placeholders(code: str) -> list[str]:
    return list(dict.fromkeys(PLACEHOLDER_PATTERN.findall(code)))


def extract_meaningful_identifiers(code: str) -> list[str]:
    code = strip_comments(code)
    ignored = set(keyword.kwlist) | {
        "int",
        "char",
        "void",
        "return",
        "size_t",
        "struct",
        "const",
        "static",
        "unsigned",
        "signed",
        "short",
        "long",
        "float",
        "double",
    }
    return [
        token
        for token in dict.fromkeys(IDENTIFIER_PATTERN.findall(code))
        if token not in ignored and not token.isupper()
    ]


def derive_rename_map(source_code: str, ghidra_code: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    normalized_source = normalize_source_for_target(source_code)
    signature = extract_function_signature(normalized_source)
    if signature is None:
        return mapping

    source_function_name = function_name_from_signature_text(signature)
    if source_function_name is not None:
        for placeholder in FUNCTION_PLACEHOLDER_PATTERN.findall(ghidra_code):
            mapping[placeholder] = source_function_name

    source_param_names = parameter_names_from_signature(signature) or []
    ghidra_signature = ghidra_code.split("{", 1)[0]
    ghidra_param_placeholders = list(dict.fromkeys(PARAM_PLACEHOLDER_PATTERN.findall(ghidra_signature)))
    for placeholder, identifier in zip(ghidra_param_placeholders, source_param_names, strict=False):
        if placeholder != identifier:
            mapping[placeholder] = identifier
    return mapping


def normalize_source_for_target(code: str) -> str:
    cleaned = strip_leading_non_code_lines(code)
    return "\n".join(line.rstrip() for line in cleaned.strip().splitlines())
