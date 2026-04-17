from __future__ import annotations

import re

from decomp_clarifier.c_source import (
    extract_function_signature,
    function_name_from_signature_text,
    normalize_function_signature,
    parameter_count_from_signature,
    parameter_types_from_signature,
    return_type_from_signature,
)
from decomp_clarifier.evaluation.naming_eval import normalized_name_similarity
from decomp_clarifier.evaluation.readability_eval import readability_improvement
from decomp_clarifier.schemas.model_io import ClarifiedFunctionOutput

_DECOMPILER_TYPE_PATTERN = re.compile(
    r"\b(?:undefined\d*|ulong64|ulonglong|longlong|code|byte|word|dword|qword)\b"
)
_COMPILE_FAILURE_GATE = 0.0
_BEHAVIOR_FAILURE_GATE = 0.0
_COMPILE_FAILURE_PENALTY = 0.0
_BEHAVIOR_FAILURE_PENALTY = 0.0
_EXTRACTABLE_JSON_REWARD = 0.5
_MIN_COMPLETION_TOKEN_COUNT = 5


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _token_count(code: str) -> int:
    return len(re.findall(r"[A-Za-z_]\w*|\d+", code))


def _completion_ratio(candidate_code: str, raw_code: str) -> float:
    raw_tokens = _token_count(raw_code)
    if raw_tokens == 0:
        return 1.0 if _token_count(candidate_code) > 0 else 0.0
    return min(1.0, _token_count(candidate_code) / raw_tokens)


def _is_substantially_complete(
    candidate_code: str,
    raw_code: str,
    min_completion_ratio: float,
) -> tuple[float, float]:
    candidate_tokens = _token_count(candidate_code)
    raw_tokens = _token_count(raw_code)
    if raw_tokens == 0:
        ratio = 1.0 if candidate_tokens > 0 else 0.0
        completeness = 1.0 if candidate_tokens >= _MIN_COMPLETION_TOKEN_COUNT else 0.0
        return ratio, completeness
    ratio = min(1.0, candidate_tokens / raw_tokens)
    required_tokens = max(_MIN_COMPLETION_TOKEN_COUNT, int(raw_tokens * min_completion_ratio))
    completeness = 1.0 if candidate_tokens >= required_tokens else 0.0
    return ratio, completeness


def _task_style_scales(task_type: str | None) -> dict[str, float]:
    if task_type == "cleanup":
        return {"cleanup": 1.0, "naming": 0.0, "readability": 0.5}
    if task_type == "rename":
        return {"cleanup": 0.0, "naming": 1.0, "readability": 0.0}
    return {"cleanup": 1.0, "naming": 1.0, "readability": 1.0}


def empty_reward_breakdown(
    *,
    json_valid: float = 0.0,
    format_value: float = 0.0,
    exact_json: float = 0.0,
    completion_ratio: float = 0.0,
    completeness: float = 0.0,
    behavior_from_execution: float = 0.0,
) -> dict[str, float]:
    return {
        "json_valid": json_valid,
        "format": format_value,
        "cleanup": 0.0,
        "naming": 0.0,
        "compile": 0.0,
        "behavior": 0.0,
        "readability": 0.0,
        "signature": 0.0,
        "hallucination_penalty": 0.0,
        "decompiler_type_penalty": 0.0,
        "failure_penalty": 0.0,
        "gate_factor": 0.0,
        "exact_json": exact_json,
        "completion_ratio": completion_ratio,
        "completeness": completeness,
        "behavior_from_execution": behavior_from_execution,
        "total": 0.0,
    }


def format_reward(
    output: ClarifiedFunctionOutput,
    *,
    json_valid: bool = True,
    exact_json: bool = True,
) -> float:
    if not json_valid:
        return 0.0
    if not output.summary.strip() or not output.cleaned_c.strip():
        return 0.0
    return 1.0 if exact_json else _EXTRACTABLE_JSON_REWARD


def cleanup_reward(output: ClarifiedFunctionOutput, raw_code: str) -> float:
    placeholders_before = len(
        re.findall(r"\b(?:param_\d+|local_[0-9A-Fa-f]+|iVar\d+|uVar\d+)\b", raw_code)
    )
    placeholders_after = len(
        re.findall(r"\b(?:param_\d+|local_[0-9A-Fa-f]+|iVar\d+|uVar\d+)\b", output.cleaned_c)
    )
    if placeholders_before == 0:
        return 0.0
    return _clamp01((placeholders_before - placeholders_after) / placeholders_before)


def naming_reward(output: ClarifiedFunctionOutput, target_renamings: dict[str, str]) -> float:
    if not target_renamings:
        return 0.0
    return normalized_name_similarity(target_renamings, output.renamings)


def compile_reward(compiles: bool) -> float:
    return 1.0 if compiles else 0.0


def behavior_reward(score: float) -> float:
    return _clamp01(score)


def readability_reward(output: ClarifiedFunctionOutput, raw_code: str) -> float:
    return _clamp01(max(0.0, readability_improvement(output.cleaned_c, raw_code)))


def signature_reward(
    output: ClarifiedFunctionOutput,
    target_clean_code: str,
    source_function_name: str,
) -> float:
    candidate_signature = extract_function_signature(output.cleaned_c)
    target_signature = extract_function_signature(target_clean_code)
    if candidate_signature is None or target_signature is None:
        return 0.0
    if normalize_function_signature(candidate_signature) == normalize_function_signature(
        target_signature
    ):
        return 1.0

    candidate_name = function_name_from_signature_text(candidate_signature)
    target_name = function_name_from_signature_text(target_signature)
    candidate_params = parameter_count_from_signature(candidate_signature)
    target_params = parameter_count_from_signature(target_signature)
    candidate_param_types = parameter_types_from_signature(candidate_signature)
    target_param_types = parameter_types_from_signature(target_signature)
    candidate_return = return_type_from_signature(candidate_signature)
    target_return = return_type_from_signature(target_signature)

    score = 0.0
    if candidate_name is not None and candidate_name == target_name == source_function_name:
        score += 0.25
    elif candidate_name is not None and candidate_name == target_name:
        score += 0.15
    if candidate_params is not None and candidate_params == target_params:
        score += 0.2
    if (
        candidate_param_types is not None
        and target_param_types is not None
        and candidate_param_types == target_param_types
    ):
        score += 0.35
    if candidate_return is not None and candidate_return == target_return:
        score += 0.2
    return min(1.0, score)


def decompiler_type_penalty(output: ClarifiedFunctionOutput) -> float:
    types = {
        match.group(0).lower()
        for match in _DECOMPILER_TYPE_PATTERN.finditer(output.cleaned_c)
    }
    return float(min(3, len(types)) / 3)


def hallucination_penalty(
    output: ClarifiedFunctionOutput,
    allowed_imports: list[str],
    allowed_callees: list[str],
) -> float:
    allowed_calls = set(allowed_imports) | set(allowed_callees)
    if not allowed_calls:
        return 0.0
    observed_calls = set(re.findall(r"\b([A-Za-z_]\w*)\s*\(", output.cleaned_c))
    harmless = {"if", "for", "while", "switch", "return", "sizeof"}
    unsupported = [
        name for name in observed_calls if name not in allowed_calls and name not in harmless
    ]
    return float(min(3, len(unsupported)) / 3)


def safety_gate_factor(*, compile_success: bool, behavior_success: bool) -> float:
    factor = 1.0
    if not compile_success:
        factor *= _COMPILE_FAILURE_GATE
    if not behavior_success:
        factor *= _BEHAVIOR_FAILURE_GATE
    return factor


def reward_breakdown(
    *,
    output: ClarifiedFunctionOutput,
    json_valid: bool,
    exact_json: bool = True,
    raw_code: str,
    target_clean_code: str,
    source_function_name: str,
    target_renamings: dict[str, str],
    compile_success: bool,
    behavior_success: bool,
    behavior_score: float | None = None,
    behavior_improvement: bool = True,
    behavior_from_execution: bool = False,
    allowed_imports: list[str],
    allowed_callees: list[str],
    weights: dict[str, float],
    task_type: str = "full_clarify",
    min_completion_ratio: float = 0.3,
) -> dict[str, float]:
    if not json_valid or not output.summary.strip() or not output.cleaned_c.strip():
        return empty_reward_breakdown(
            json_valid=0.0 if not json_valid else 1.0,
            exact_json=1.0 if exact_json else 0.0,
            behavior_from_execution=1.0 if behavior_from_execution else 0.0,
        )
    format_value = format_reward(output, json_valid=True, exact_json=exact_json)
    completion_ratio, completeness = _is_substantially_complete(
        output.cleaned_c,
        raw_code,
        min_completion_ratio,
    )
    if not completeness:
        return empty_reward_breakdown(
            json_valid=1.0,
            format_value=format_value,
            exact_json=1.0 if exact_json else 0.0,
            completion_ratio=completion_ratio,
            completeness=0.0,
            behavior_from_execution=1.0 if behavior_from_execution else 0.0,
        )
    cleanup_value = cleanup_reward(output, raw_code)
    naming_value = naming_reward(output, target_renamings)
    compile_value = compile_reward(compile_success)
    effective_behavior_score = (
        behavior_score if behavior_score is not None else (1.0 if behavior_success else 0.0)
    )
    behavior_value = behavior_reward(
        effective_behavior_score if behavior_improvement else 0.0
    )
    readability_value = readability_reward(output, raw_code)
    signature_value = signature_reward(output, target_clean_code, source_function_name)
    hallucination_value = hallucination_penalty(output, allowed_imports, allowed_callees)
    decompiler_type_value = decompiler_type_penalty(output)
    gate_factor = safety_gate_factor(
        compile_success=compile_success,
        behavior_success=behavior_success,
    )
    style_scales = _task_style_scales(task_type)
    core_total = (
        weights.get("format", 1.0) * format_value
        + weights.get("compile", 1.0) * compile_value
        + weights.get("behavior", 1.0) * behavior_value
        + weights.get("signature", 0.0) * signature_value
    )
    style_total = gate_factor * (
        weights.get("cleanup", 1.0) * cleanup_value * style_scales["cleanup"]
        + weights.get("naming", 1.0) * naming_value * style_scales["naming"]
        + weights.get("readability", 1.0) * readability_value * style_scales["readability"]
    )
    penalty_total = (
        weights.get("hallucination_penalty", 1.0) * hallucination_value
        + weights.get("decompiler_type_penalty", 0.0) * decompiler_type_value
    )
    failure_penalty = 0.0
    if not compile_success:
        failure_penalty += weights.get("compile", 1.0) * _COMPILE_FAILURE_PENALTY
    if compile_success and not behavior_success:
        failure_penalty += weights.get("behavior", 1.0) * _BEHAVIOR_FAILURE_PENALTY
    total = core_total + style_total - penalty_total - failure_penalty
    if not compile_success:
        total = min(total, 0.0)
    return {
        "json_valid": 1.0,
        "format": format_value,
        "cleanup": cleanup_value,
        "naming": naming_value,
        "compile": compile_value,
        "behavior": behavior_value,
        "readability": readability_value,
        "signature": signature_value,
        "hallucination_penalty": hallucination_value,
        "decompiler_type_penalty": decompiler_type_value,
        "failure_penalty": failure_penalty,
        "gate_factor": gate_factor,
        "exact_json": 1.0 if exact_json else 0.0,
        "completion_ratio": completion_ratio,
        "completeness": completeness,
        "behavior_from_execution": 1.0 if behavior_from_execution else 0.0,
        "total": total,
    }


def weighted_reward(
    *,
    output: ClarifiedFunctionOutput,
    json_valid: bool,
    exact_json: bool = True,
    raw_code: str,
    target_clean_code: str,
    source_function_name: str,
    target_renamings: dict[str, str],
    compile_success: bool,
    behavior_success: bool,
    behavior_score: float | None = None,
    behavior_improvement: bool = True,
    behavior_from_execution: bool = False,
    allowed_imports: list[str],
    allowed_callees: list[str],
    weights: dict[str, float],
    task_type: str = "full_clarify",
    min_completion_ratio: float = 0.3,
) -> float:
    return reward_breakdown(
        output=output,
        json_valid=json_valid,
        exact_json=exact_json,
        raw_code=raw_code,
        target_clean_code=target_clean_code,
        source_function_name=source_function_name,
        target_renamings=target_renamings,
        compile_success=compile_success,
        behavior_success=behavior_success,
        behavior_score=behavior_score,
        behavior_improvement=behavior_improvement,
        behavior_from_execution=behavior_from_execution,
        allowed_imports=allowed_imports,
        allowed_callees=allowed_callees,
        weights=weights,
        task_type=task_type,
        min_completion_ratio=min_completion_ratio,
    )["total"]
