from __future__ import annotations

from decomp_clarifier.schemas.dataset import FunctionDatasetSample
from decomp_clarifier.schemas.model_io import ClarifiedFunctionOutput


def predict(sample: FunctionDatasetSample) -> ClarifiedFunctionOutput:
    return ClarifiedFunctionOutput(
        summary="Raw Ghidra decompiler output without clarification.",
        confidence=0.0,
        renamings={},
        cleaned_c=sample.ghidra_decompiled_code,
    )
