from __future__ import annotations

from dataclasses import dataclass

from decomp_clarifier.evaluation.behavior_eval import (
    behavior_similarity,
    evaluate_execution_behavior,
    is_behavior_improvement,
)
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
    sample: FunctionDatasetSample,
    output: ClarifiedFunctionOutput,
    *,
    json_valid: bool = True,
) -> VerificationResult:
    execution_result = evaluate_execution_behavior(
        output.cleaned_c,
        source_function_name=sample.source_function_name,
        compiler_executable=sample.compiler_executable,
        compiler_family=sample.compiler,
        tests_ref=sample.tests_ref or "",
    )
    if execution_result is not None:
        compile_success = execution_result.compile_success
        behavior_success = execution_result.pass_rate >= 1.0
    else:
        compile_success = compile_candidate(
            output.cleaned_c,
            sample.compile_reference_source or sample.source_code,
            function_name=sample.source_function_name,
        )
        behavior_score = behavior_similarity(output.cleaned_c, sample.target_clean_code)
        behavior_success = behavior_score >= 0.35 and is_behavior_improvement(
            output.cleaned_c,
            sample.ghidra_decompiled_code,
            sample.target_clean_code,
        )
    return VerificationResult(
        json_valid=json_valid,
        field_complete=field_complete(output) if json_valid else False,
        compile_success=compile_success,
        behavior_success=behavior_success if json_valid else False,
        readability_score=score_readability(output.cleaned_c),
        naming_score=(
            normalized_name_similarity(sample.rename_map_target, output.renamings)
            if json_valid
            else 0.0
        ),
        placeholder_ratio=placeholder_ratio(output.cleaned_c),
    )
