from __future__ import annotations

from dataclasses import dataclass

from decomp_clarifier.c_source import (
    function_name_from_signature_line,
    looks_like_function_signature_line,
)
from decomp_clarifier.ghidra_export.parse_exports import ParsedGhidraProject
from decomp_clarifier.schemas.generation import GeneratedProject
from decomp_clarifier.schemas.ghidra import GhidraFunctionRow


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


def _iter_function_starts(content: str) -> list[tuple[int, str]]:
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


def extract_source_functions(project: GeneratedProject) -> list[SourceFunction]:
    functions: list[SourceFunction] = []
    order = 0
    for file in project.files:
        if not file.path.endswith(".c"):
            continue
        for start_index, name in _iter_function_starts(file.content):
            code = _slice_function(file.content, start_index).strip()
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
