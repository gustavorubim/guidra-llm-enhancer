from __future__ import annotations

import re

from decomp_clarifier.evaluation.naming_eval import normalized_name_similarity
from decomp_clarifier.evaluation.readability_eval import readability_improvement
from decomp_clarifier.schemas.model_io import ClarifiedFunctionOutput


def format_reward(output: ClarifiedFunctionOutput) -> float:
    return 1.0 if output.summary.strip() and output.cleaned_c.strip() else 0.0


def cleanup_reward(output: ClarifiedFunctionOutput, raw_code: str) -> float:
    placeholders_before = len(
        re.findall(r"\b(?:param_\d+|local_[0-9A-Fa-f]+|iVar\d+|uVar\d+)\b", raw_code)
    )
    placeholders_after = len(
        re.findall(r"\b(?:param_\d+|local_[0-9A-Fa-f]+|iVar\d+|uVar\d+)\b", output.cleaned_c)
    )
    if placeholders_before == 0:
        return 0.5
    return max(0.0, min(1.0, (placeholders_before - placeholders_after) / placeholders_before))


def naming_reward(output: ClarifiedFunctionOutput, target_renamings: dict[str, str]) -> float:
    return normalized_name_similarity(target_renamings, output.renamings)


def compile_reward(compiles: bool) -> float:
    return 1.0 if compiles else 0.0


def behavior_reward(success: bool) -> float:
    return 1.0 if success else 0.0


def readability_reward(output: ClarifiedFunctionOutput, raw_code: str) -> float:
    return max(0.0, min(1.0, 0.5 + readability_improvement(output.cleaned_c, raw_code)))


def hallucination_penalty(
    output: ClarifiedFunctionOutput,
    allowed_imports: list[str],
    allowed_callees: list[str],
) -> float:
    allowed_calls = set(allowed_imports) | set(allowed_callees)
    observed_calls = set(re.findall(r"\b([A-Za-z_]\w*)\s*\(", output.cleaned_c))
    harmless = {"if", "for", "while", "switch", "return", "sizeof"}
    unsupported = [
        name for name in observed_calls if name not in allowed_calls and name not in harmless
    ]
    return float(len(unsupported))


def weighted_reward(
    *,
    output: ClarifiedFunctionOutput,
    raw_code: str,
    target_renamings: dict[str, str],
    compile_success: bool,
    behavior_success: bool,
    allowed_imports: list[str],
    allowed_callees: list[str],
    weights: dict[str, float],
) -> float:
    return (
        weights.get("format", 1.0) * format_reward(output)
        + weights.get("cleanup", 1.0) * cleanup_reward(output, raw_code)
        + weights.get("naming", 1.0) * naming_reward(output, target_renamings)
        + weights.get("compile", 1.0) * compile_reward(compile_success)
        + weights.get("behavior", 1.0) * behavior_reward(behavior_success)
        + weights.get("readability", 1.0) * readability_reward(output, raw_code)
        - weights.get("hallucination_penalty", 1.0)
        * hallucination_penalty(output, allowed_imports, allowed_callees)
    )
