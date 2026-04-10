from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field


def now_utc() -> str:
    return datetime.now(tz=UTC).isoformat()


class SampleEvaluation(BaseModel):
    sample_id: str
    system: str
    json_valid: bool
    field_complete: bool
    placeholder_ratio: float
    readability_score: float
    naming_score: float
    compile_success: bool
    behavior_success: bool
    notes: list[str] = Field(default_factory=list)


class ReportExample(BaseModel):
    sample_id: str
    project_id: str
    source_function_name: str
    raw_ghidra: str
    candidate: str
    original_source: str
    note: str


class EvaluationReport(BaseModel):
    run_id: str
    generated_at: str = Field(default_factory=now_utc)
    metrics: dict[str, float] = Field(default_factory=dict)
    samples: list[SampleEvaluation] = Field(default_factory=list)
    examples: list[ReportExample] = Field(default_factory=list)
