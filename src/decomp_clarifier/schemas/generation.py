from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class GeneratedFile(BaseModel):
    path: str
    content: str

    @field_validator("path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        if value.startswith("/") or ".." in value.split("/"):
            raise ValueError("generated file paths must be project-relative")
        return value


class GeneratedTestCase(BaseModel):
    name: str
    input: str
    expected: str


class BuildSpec(BaseModel):
    entrypoints: list[str] = Field(default_factory=list)
    c_standard: str = "c11"
    compiler_family: str = "clang"


class FunctionIntent(BaseModel):
    function_name: str
    intent: str


class SemanticHints(BaseModel):
    project_purpose: str
    function_intents: list[FunctionIntent] = Field(default_factory=list)


class GeneratedProject(BaseModel):
    project_id: str
    summary: str
    difficulty: str
    files: list[GeneratedFile]
    tests: list[GeneratedTestCase] = Field(default_factory=list)
    build: BuildSpec = Field(default_factory=BuildSpec)
    semantic_hints: SemanticHints

    @property
    def source_files(self) -> list[GeneratedFile]:
        return [file for file in self.files if file.path.endswith(".c") or file.path.endswith(".h")]
