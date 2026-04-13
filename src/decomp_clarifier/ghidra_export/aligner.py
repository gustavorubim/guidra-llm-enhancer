from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from decomp_clarifier.c_source import iter_function_starts, slice_function
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


def _ghidra_row_score(function: GhidraFunctionRow) -> tuple[int, int, int, int, int, str]:
    return (
        function.instruction_count,
        function.basic_block_count,
        len(function.callees),
        len(function.callers),
        len(function.decompiled_text),
        function.function_address,
    )


def select_best_ghidra_rows(
    functions: Iterable[GhidraFunctionRow], source_names: set[str]
) -> list[GhidraFunctionRow]:
    best_by_name: dict[str, GhidraFunctionRow] = {}
    passthrough: list[GhidraFunctionRow] = []
    for function in functions:
        name = function.ghidra_function_name
        if name not in source_names:
            passthrough.append(function)
            continue
        current = best_by_name.get(name)
        if current is None or _ghidra_row_score(function) > _ghidra_row_score(current):
            best_by_name[name] = function
    return [*best_by_name.values(), *passthrough]


def extract_source_functions(project: GeneratedProject) -> list[SourceFunction]:
    functions: list[SourceFunction] = []
    order = 0
    for file in project.files:
        if not file.path.endswith(".c"):
            continue
        for start_index, name in iter_function_starts(file.content):
            code = slice_function(file.content, start_index).strip()
            functions.append(
                SourceFunction(file_path=file.path, name=name, code=code, order_index=order)
            )
            order += 1
    return functions


def align_functions(
    project: GeneratedProject, parsed_project: ParsedGhidraProject
) -> list[AlignedFunction]:
    source_functions = extract_source_functions(project)
    source_names = {function.name for function in source_functions}
    ghidra_functions = select_best_ghidra_rows(parsed_project.functions, source_names)
    by_name: dict[str, list[SourceFunction]] = {}
    for function in source_functions:
        by_name.setdefault(function.name, []).append(function)
    aligned: list[AlignedFunction] = []
    used_indices: set[int] = set()
    remaining_ghidra: list[GhidraFunctionRow] = []
    for function in ghidra_functions:
        candidates = by_name.get(function.ghidra_function_name)
        if not candidates:
            remaining_ghidra.append(function)
            continue
        source = candidates.pop(0)
        aligned.append(AlignedFunction(source=source, ghidra=function))
        used_indices.add(source.order_index)

    remaining_source = [
        function for function in source_functions if function.order_index not in used_indices
    ]
    for source, ghidra in zip(remaining_source, remaining_ghidra, strict=False):
        aligned.append(AlignedFunction(source=source, ghidra=ghidra))
    return aligned
