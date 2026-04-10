from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field


def now_utc() -> str:
    return datetime.now(tz=UTC).isoformat()


class CompileCommand(BaseModel):
    executable: str
    args: list[str] = Field(default_factory=list)
    cwd: str


class BinaryArtifact(BaseModel):
    path: str
    binary_format: str
    arch: str
    stripped: bool


class TestExecutionResult(BaseModel):
    name: str
    passed: bool
    returncode: int
    stdout: str
    stderr: str


class CompileManifest(BaseModel):
    project_id: str
    build_id: str
    compiler_family: str
    compiler_version: str
    host_os: str
    binary_format: str
    arch: str
    opt_level: str
    source_root: str
    output_root: str
    build_log: str = ""
    compile_commands: list[CompileCommand] = Field(default_factory=list)
    binaries: list[BinaryArtifact] = Field(default_factory=list)
    test_results: list[TestExecutionResult] = Field(default_factory=list)
    created_at: str = Field(default_factory=now_utc)
