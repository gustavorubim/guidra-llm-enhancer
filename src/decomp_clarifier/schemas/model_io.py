from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class PromptInput(BaseModel):
    task_type: str
    decompiled_code: str
    assembly: str
    strings: list[str] = Field(default_factory=list)
    imports: list[str] = Field(default_factory=list)
    callers: list[str] = Field(default_factory=list)
    callees: list[str] = Field(default_factory=list)
    semantic_summary: str | None = None


class ClarifiedFunctionOutput(BaseModel):
    summary: str
    confidence: float
    renamings: dict[str, str] = Field(default_factory=dict)
    cleaned_c: str

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, value: float) -> float:
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value


class PredictionRecord(BaseModel):
    sample_id: str
    system: str
    output: ClarifiedFunctionOutput
