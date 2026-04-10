from __future__ import annotations

import re
from dataclasses import dataclass

from decomp_clarifier.ghidra_export.parse_exports import ParsedGhidraProject
from decomp_clarifier.schemas.generation import GeneratedProject
from decomp_clarifier.schemas.ghidra import GhidraFunctionRow

FUNCTION_HEADER = re.compile(
    r"(?P<signature>\b[A-Za-z_][\w\s\*]*\b(?P<name>[A-Za-z_]\w*)\s*\([^;{}]*\)\s*\{)",
    re.MULTILINE,
)


@dataclass(frozen=True)
class SourceFunction:
    file_path: str
    name: str
    code: str
    order_index: int


@dataclass(frozen=True)
class AlignedFunction:
    source: SourceFunction
    ghidra: GhidraFunctionRow


def _slice_function(content: str, start_index: int) -> str:
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


def extract_source_functions(project: GeneratedProject) -> list[SourceFunction]:
    functions: list[SourceFunction] = []
    order = 0
    for file in project.files:
        if not file.path.endswith(".c"):
            continue
        for match in FUNCTION_HEADER.finditer(file.content):
            name = match.group("name")
            if name in {"if", "for", "while", "switch"}:
                continue
            code = _slice_function(file.content, match.start()).strip()
            functions.append(
                SourceFunction(file_path=file.path, name=name, code=code, order_index=order)
            )
            order += 1
    return functions


def align_functions(
    project: GeneratedProject, parsed_project: ParsedGhidraProject
) -> list[AlignedFunction]:
    source_functions = extract_source_functions(project)
    by_name = {function.name: function for function in source_functions}
    aligned: list[AlignedFunction] = []
    used_names: set[str] = set()
    remaining_ghidra: list[GhidraFunctionRow] = []
    for function in parsed_project.functions:
        source = by_name.get(function.ghidra_function_name)
        if source is None:
            remaining_ghidra.append(function)
            continue
        aligned.append(AlignedFunction(source=source, ghidra=function))
        used_names.add(source.name)

    remaining_source = [
        function for function in source_functions if function.name not in used_names
    ]
    for source, ghidra in zip(remaining_source, remaining_ghidra, strict=False):
        aligned.append(AlignedFunction(source=source, ghidra=ghidra))
    return aligned
