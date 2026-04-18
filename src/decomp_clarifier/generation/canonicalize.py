from __future__ import annotations

from decomp_clarifier.schemas.generation import GeneratedFile, GeneratedProject


def canonicalize_compiler_family(family: str) -> str:
    normalized = family.strip().lower()
    if normalized in {"clang", "gcc", "cc"}:
        return "clang"
    return normalized or "clang"


def canonicalize_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned_lines = [line.rstrip() for line in normalized.splitlines()]
    return "\n".join(cleaned_lines).strip() + "\n"


def canonicalize_project(project: GeneratedProject) -> GeneratedProject:
    files = [
        GeneratedFile(path=file.path, content=canonicalize_text(file.content))
        for file in sorted(project.files, key=lambda item: item.path)
    ]
    build = project.build.model_copy(
        update={"compiler_family": canonicalize_compiler_family(project.build.compiler_family)}
    )
    return project.model_copy(update={"files": files, "build": build})
