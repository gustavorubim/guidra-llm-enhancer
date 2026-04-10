from __future__ import annotations

from decomp_clarifier.schemas.dataset import FunctionDatasetSample
from decomp_clarifier.schemas.model_io import ClarifiedFunctionOutput


def summarize_improvements(
    sample: FunctionDatasetSample, output: ClarifiedFunctionOutput
) -> list[str]:
    improvements: list[str] = []
    if output.renamings:
        improvements.append(f"Added {len(output.renamings)} renamings.")
    if output.cleaned_c.strip() != sample.ghidra_decompiled_code.strip():
        improvements.append("Structural cleanup changed the decompiled text.")
    if output.summary.strip():
        improvements.append("Produced a human-readable summary.")
    return improvements
