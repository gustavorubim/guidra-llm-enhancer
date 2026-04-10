from __future__ import annotations

from dataclasses import dataclass

from decomp_clarifier.evaluation.behavior_eval import behavior_similarity, is_behavior_improvement
from decomp_clarifier.evaluation.compile_eval import compile_candidate
from decomp_clarifier.evaluation.metrics import field_complete, placeholder_ratio
from decomp_clarifier.evaluation.naming_eval import normalized_name_similarity
from decomp_clarifier.evaluation.readability_eval import score_readability
from decomp_clarifier.schemas.dataset import FunctionDatasetSample
from decomp_clarifier.schemas.model_io import ClarifiedFunctionOutput


@dataclass(frozen=True)
class VerificationResult:
    json_valid: bool
    field_complete: bool
    compile_success: bool
    behavior_success: bool
    readability_score: float
    naming_score: float
    placeholder_ratio: float


def verify_output(
    sample: FunctionDatasetSample, output: ClarifiedFunctionOutput
) -> VerificationResult:
    compile_success = compile_candidate(
        output.cleaned_c, sample.compile_reference_source or sample.source_code
    )
    behavior_score = behavior_similarity(output.cleaned_c, sample.target_clean_code)
    behavior_success = behavior_score >= 0.35 and is_behavior_improvement(
        output.cleaned_c,
        sample.ghidra_decompiled_code,
        sample.target_clean_code,
    )
    return VerificationResult(
        json_valid=True,
        field_complete=field_complete(output),
        compile_success=compile_success,
        behavior_success=behavior_success,
        readability_score=score_readability(output.cleaned_c),
        naming_score=normalized_name_similarity(sample.rename_map_target, output.renamings),
        placeholder_ratio=placeholder_ratio(output.cleaned_c),
    )
