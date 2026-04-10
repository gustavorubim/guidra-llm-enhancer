from __future__ import annotations

from decomp_clarifier.schemas.generation import GeneratedFile, GeneratedProject


def canonicalize_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned_lines = [line.rstrip() for line in normalized.splitlines()]
    return "\n".join(cleaned_lines).strip() + "\n"


def canonicalize_project(project: GeneratedProject) -> GeneratedProject:
    files = [
        GeneratedFile(path=file.path, content=canonicalize_text(file.content))
        for file in sorted(project.files, key=lambda item: item.path)
    ]
    return project.model_copy(update={"files": files})
