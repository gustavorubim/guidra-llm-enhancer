from __future__ import annotations

import re

from decomp_clarifier.schemas.generation import GeneratedProject
from decomp_clarifier.settings import GenerationValidationConfig

FUNCTION_PATTERN = re.compile(
    r"\b[A-Za-z_][\w\s\*]*\b([A-Za-z_]\w*)\s*\([^;{}]*\)\s*\{",
    re.MULTILINE,
)


class ProjectValidationError(ValueError):
    """Raised when a generated project does not satisfy repository constraints."""


def extract_function_names(source: str) -> list[str]:
    return [match.group(1) for match in FUNCTION_PATTERN.finditer(source)]


def validate_project(project: GeneratedProject, config: GenerationValidationConfig) -> None:
    source_files = [file for file in project.files if file.path.endswith(".c")]
    if len(source_files) < config.min_source_files:
        raise ProjectValidationError("project has too few source files")
    if len(source_files) > config.max_source_files:
        raise ProjectValidationError("project has too many source files")

    function_count = sum(len(extract_function_names(file.content)) for file in source_files)
    if function_count < config.min_function_count:
        raise ProjectValidationError("project does not contain enough functions")

    banned_includes = tuple(config.banned_includes)
    banned_calls = tuple(config.banned_calls)
    for file in project.source_files:
        if any(include in file.content for include in banned_includes):
            raise ProjectValidationError(f"project contains banned include in {file.path}")
        if any(re.search(rf"\b{re.escape(call)}\s*\(", file.content) for call in banned_calls):
            raise ProjectValidationError(f"project contains banned call in {file.path}")
