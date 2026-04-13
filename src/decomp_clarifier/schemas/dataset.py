from __future__ import annotations

from pydantic import BaseModel, Field


class FunctionDatasetSample(BaseModel):
    sample_id: str
    project_id: str
    split: str
    task_type: str
    host_os: str
    compiler: str
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


class PackedRLRecord(BaseModel):
    sample_id: str
    task_type: str
    prompt: str
    source_function_name: str
    raw_code: str
    compile_reference_source: str
    target_clean_code: str
    target_renamings: str
    allowed_imports: str
    allowed_callees: str


class DatasetManifest(BaseModel):
    record_count: int
    split_counts: dict[str, int]
    task_counts: dict[str, int]
    output_path: str
