from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field


def now_utc() -> str:
    return datetime.now(tz=UTC).isoformat()


class VariableRecord(BaseModel):
    name: str
    type: str
    storage: str | None = None


class GhidraFunctionRow(BaseModel):
    project_id: str
    binary_path: str
    binary_name: str
    function_address: str
    ghidra_function_name: str
    signature: str
    return_type: str
    parameters: list[VariableRecord] = Field(default_factory=list)
    local_variables: list[VariableRecord] = Field(default_factory=list)
    decompiled_text: str
    disassembly_text: str
    strings: list[str] = Field(default_factory=list)
    imports: list[str] = Field(default_factory=list)
    callees: list[str] = Field(default_factory=list)
    callers: list[str] = Field(default_factory=list)
    basic_block_count: int = 0
    instruction_count: int = 0


class GhidraProjectManifest(BaseModel):
    project_id: str
    binary_path: str
    binary_name: str
    output_dir: str
    functions_path: str
    created_at: str = Field(default_factory=now_utc)
    strings_path: str | None = None
    imports_path: str | None = None
    symbols_path: str | None = None
    decompiled_snapshot_path: str | None = None
    disassembly_snapshot_path: str | None = None
