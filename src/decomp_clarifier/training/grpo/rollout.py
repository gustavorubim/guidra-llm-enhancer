from __future__ import annotations

from decomp_clarifier.inference.formatter import normalize_output
from decomp_clarifier.schemas.model_io import ClarifiedFunctionOutput


def normalize_completion(text: str) -> ClarifiedFunctionOutput:
    return normalize_output(text)
