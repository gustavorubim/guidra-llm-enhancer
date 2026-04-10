from __future__ import annotations

import keyword
import re

PLACEHOLDER_PATTERN = re.compile(
    r"\b(?:param_\d+|local_[0-9A-Fa-f]+|iVar\d+|uVar\d+|puVar\d+|FUN_[0-9A-Fa-f]+)\b"
)
IDENTIFIER_PATTERN = re.compile(r"\b[A-Za-z_]\w*\b")


def extract_placeholders(code: str) -> list[str]:
    return list(dict.fromkeys(PLACEHOLDER_PATTERN.findall(code)))


def extract_meaningful_identifiers(code: str) -> list[str]:
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
    placeholders = extract_placeholders(ghidra_code)
    identifiers = extract_meaningful_identifiers(source_code)
    mapping: dict[str, str] = {}
    for placeholder, identifier in zip(placeholders, identifiers, strict=False):
        if placeholder != identifier:
            mapping[placeholder] = identifier
    return mapping


def normalize_source_for_target(code: str) -> str:
    return "\n".join(line.rstrip() for line in code.strip().splitlines())
