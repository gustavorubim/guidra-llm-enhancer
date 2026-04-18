from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class FunctionDatasetSample(BaseModel):
    sample_id: str
    project_id: str
    split: str
    task_type: str
    host_os: str
    compiler: str
    compiler_executable: str | None = None
    opt_level: str
    binary_format: str
    source_function_name: str
    source_code: str
    compile_reference_source: str | None = None
    target_clean_code: str
    ghidra_function_name: str
    ghidra_decompiled_code: str
    assembly: str
    strings: list[str] = Field(default_factory=list)
    imports: list[str] = Field(default_factory=list)
    callers: list[str] = Field(default_factory=list)
    callees: list[str] = Field(default_factory=list)
    semantic_summary: str
    rename_map_target: dict[str, str] = Field(default_factory=dict)
    tests_ref: str | None = None
    difficulty: str


class PackedSFTRecord(BaseModel):
    sample_id: str
    task_type: str
    prompt: str
    response_json: str
    prompt_messages: list[ChatMessage] = Field(default_factory=list)
    completion_messages: list[ChatMessage] = Field(default_factory=list)


class PackedRLRecord(BaseModel):
    sample_id: str
    task_type: str
    prompt: str
    prompt_messages: list[ChatMessage] = Field(default_factory=list)
    source_function_name: str
    raw_code: str
    compile_reference_source: str
    target_clean_code: str
    target_renamings: str
    allowed_imports: str
    allowed_callees: str
    compiler_executable: str | None = None
    tests_ref: str | None = None


class DatasetManifest(BaseModel):
    record_count: int
    split_counts: dict[str, int]
    task_counts: dict[str, int]
    output_path: str
