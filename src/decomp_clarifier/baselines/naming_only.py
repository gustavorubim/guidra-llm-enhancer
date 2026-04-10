from __future__ import annotations

import re

from decomp_clarifier.dataset.transforms import extract_placeholders
from decomp_clarifier.schemas.dataset import FunctionDatasetSample
from decomp_clarifier.schemas.model_io import ClarifiedFunctionOutput


def _semantic_stub(sample: FunctionDatasetSample) -> list[str]:
    words = [word.lower() for word in re.findall(r"[A-Za-z]+", sample.semantic_summary)]
    return [word for word in words if len(word) > 2]


def predict(sample: FunctionDatasetSample) -> ClarifiedFunctionOutput:
    words = _semantic_stub(sample)
    renamings: dict[str, str] = {}
    for index, placeholder in enumerate(
        extract_placeholders(sample.ghidra_decompiled_code), start=1
    ):
        base = words[index - 1] if index - 1 < len(words) else "value"
        suffix = "arg" if placeholder.startswith("param_") else "tmp"
        renamings[placeholder] = f"{base}_{suffix}{index}"
    return ClarifiedFunctionOutput(
        summary=f"Heuristic rename pass for {sample.ghidra_function_name}.",
        confidence=0.25,
        renamings=renamings,
        cleaned_c=sample.ghidra_decompiled_code,
    )
