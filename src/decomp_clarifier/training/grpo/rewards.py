from __future__ import annotations

import re

from decomp_clarifier.c_source import (
    extract_function_signature,
    function_name_from_signature_text,
    iter_function_starts,
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
_INLINE_FUNCTION_START = re.compile(
    r"(?m)^\s*(?:[A-Za-z_][\w\s\*]*?\s+)?(?P<name>[A-Za-z_]\w*)\s*\([^;{}]*\)\s*\{"
)
_RAW_FUNCTION_START = re.compile(
    r"(?P<name>[A-Za-z_]\w*)\s*\([^;{}]{0,160}\)\s*\{"
)
_COMPILE_FAILURE_GATE = 0.4
_BEHAVIOR_FAILURE_GATE = 0.75
_COMPILE_FAILURE_PENALTY = 0.5
_BEHAVIOR_FAILURE_PENALTY = 0.25
_EXTRACTABLE_JSON_REWARD = 0.75
_MIN_COMPLETION_TOKEN_COUNT = 5
_MAX_COMPLETION_RATIO = 1.75
_MAX_FUNCTION_COUNT = 1
_MAX_INVALID_COMPLETION_RATIO = 0.9
_UPPERCASE_IDENTIFIER_PATTERN = re.compile(r"\b[A-Z][A-Z0-9_]{1,}\b")
_BOOL_TOKEN_PATTERN = re.compile(r"\b(?:bool|true|false)\b")
_COMMON_C_CONSTANTS = {
    "EOF",
    "NULL",
    "SIZE_MAX",
    "UINT_MAX",
    "ULONG_MAX",
    "INT_MAX",
    "INT_MIN",
    "CHAR_BIT",
    "EXIT_SUCCESS",
    "EXIT_FAILURE",
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _token_count(code: str) -> int:
    return len(re.findall(r"[A-Za-z_]\w*|\d+", code))


def _completion_ratio(candidate_code: str, raw_code: str) -> float:
    raw_tokens = _token_count(raw_code)
    if raw_tokens == 0:
        return 1.0 if _token_count(candidate_code) > 0 else 0.0
    return min(1.0, _token_count(candidate_code) / raw_tokens)


def _uncapped_completion_ratio(candidate_code: str, raw_code: str) -> float:
    raw_tokens = _token_count(raw_code)
    if raw_tokens == 0:
        return 1.0 if _token_count(candidate_code) > 0 else 0.0
    return _token_count(candidate_code) / raw_tokens


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


def _function_definition_count(candidate_code: str) -> int:
    starts = iter_function_starts(candidate_code)
    if starts:
        return len(starts)
    inline_names = [
        match.group("name")
        for match in _INLINE_FUNCTION_START.finditer(candidate_code)
        if match.group("name") not in {"if", "for", "while", "switch"}
    ]
    if inline_names:
        return len(inline_names)
    return 1 if extract_function_signature(candidate_code) is not None else 0


def overshoot_penalty(
    candidate_code: str,
    raw_code: str,
    *,
    max_completion_ratio: float = _MAX_COMPLETION_RATIO,
) -> float:
    if max_completion_ratio <= 0:
        return 0.0
    ratio = _uncapped_completion_ratio(candidate_code, raw_code)
    if ratio <= max_completion_ratio:
        return 0.0
    return _clamp01((ratio - max_completion_ratio) / max_completion_ratio)


def multi_function_penalty(
    candidate_code: str,
    *,
    max_function_count: int = _MAX_FUNCTION_COUNT,
) -> float:
    if max_function_count < 1:
        return 0.0
    function_count = _function_definition_count(candidate_code)
    if function_count <= max_function_count:
        return 0.0
    return _clamp01((function_count - max_function_count) / max_function_count)


def invalid_completion_length_penalty(
    raw_completion: str,
    raw_code: str,
    *,
    max_invalid_completion_ratio: float = _MAX_INVALID_COMPLETION_RATIO,
) -> float:
    if max_invalid_completion_ratio <= 0:
        return 0.0
    ratio = _uncapped_completion_ratio(raw_completion, raw_code)
    if ratio <= max_invalid_completion_ratio:
        return 0.0
    return _clamp01((ratio - max_invalid_completion_ratio) / max_invalid_completion_ratio)


def invalid_scope_penalty(
    raw_completion: str,
    source_function_name: str,
    *,
    max_function_count: int = _MAX_FUNCTION_COUNT,
) -> float:
    target_name = source_function_name.strip()
    matched_names = [
        match.group("name")
        for match in _RAW_FUNCTION_START.finditer(raw_completion)
        if match.group("name") not in {"if", "for", "while", "switch"}
    ]
    if not matched_names:
        return 0.0
    unique_names = list(dict.fromkeys(matched_names))
    if target_name:
        non_target_names = [name for name in unique_names if name != target_name]
        if not non_target_names:
            return 0.0
        if target_name != "main" and "main" in non_target_names:
            return 1.0
        return _clamp01(len(non_target_names) / max(1, max_function_count))
    if len(unique_names) <= max_function_count:
        return 0.0
    return _clamp01((len(unique_names) - max_function_count) / max(1, max_function_count))


def _has_unbalanced_quotes(text: str) -> bool:
    in_string = False
    escaped = False
    for character in text:
        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if character == '"':
            in_string = not in_string
    return in_string


def truncation_penalty(raw_completion: str) -> float:
    stripped = raw_completion.strip()
    if not stripped:
        return 0.0
    score = 0.0
    if stripped.startswith("{"):
        score += 0.25
    if stripped.count("{") > stripped.count("}"):
        score += 0.35
    if _has_unbalanced_quotes(stripped):
        score += 0.25
    if stripped.endswith((",", ":", "\\", "{", "[")):
        score += 0.25
    if stripped.endswith('"') and not stripped.endswith('"}'):
        score += 0.15
    return _clamp01(score)


def invalid_json_penalty(
    raw_completion: str,
    raw_code: str,
    source_function_name: str,
    *,
    max_invalid_completion_ratio: float = _MAX_INVALID_COMPLETION_RATIO,
    max_function_count: int = _MAX_FUNCTION_COUNT,
    weights: dict[str, float],
) -> tuple[float, dict[str, float]]:
    length_penalty = invalid_completion_length_penalty(
        raw_completion,
        raw_code,
        max_invalid_completion_ratio=max_invalid_completion_ratio,
    )
    truncation_value = truncation_penalty(raw_completion)
    scope_penalty = invalid_scope_penalty(
        raw_completion,
        source_function_name,
        max_function_count=max_function_count,
    )
    total_penalty = (
        weights.get("invalid_json_penalty", 0.0)
        + weights.get("invalid_length_penalty", 0.0) * length_penalty
        + weights.get("truncation_penalty", 0.0) * truncation_value
        + weights.get("invalid_scope_penalty", 0.0) * scope_penalty
    )
    return total_penalty, {
        "invalid_length_penalty": length_penalty,
        "truncation_penalty": truncation_value,
        "invalid_scope_penalty": scope_penalty,
        "raw_completion_ratio": _uncapped_completion_ratio(raw_completion, raw_code),
    }


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
    overshoot_penalty_value: float = 0.0,
    multi_function_penalty_value: float = 0.0,
    function_count: float = 0.0,
    invalid_length_penalty_value: float = 0.0,
    truncation_penalty_value: float = 0.0,
    invalid_scope_penalty_value: float = 0.0,
    unknown_constant_penalty_value: float = 0.0,
    unsupported_bool_penalty_value: float = 0.0,
    raw_completion_ratio: float = 0.0,
    core_total: float = 0.0,
    style_total: float = 0.0,
    constraint_total: float = 0.0,
    total: float = 0.0,
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
        "overshoot_penalty": overshoot_penalty_value,
        "multi_function_penalty": multi_function_penalty_value,
        "function_count": function_count,
        "invalid_length_penalty": invalid_length_penalty_value,
        "truncation_penalty": truncation_penalty_value,
        "invalid_scope_penalty": invalid_scope_penalty_value,
        "unknown_constant_penalty": unknown_constant_penalty_value,
        "unsupported_bool_penalty": unsupported_bool_penalty_value,
        "raw_completion_ratio": raw_completion_ratio,
        "core_total": core_total,
        "style_total": style_total,
        "constraint_total": constraint_total,
        "total": total,
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


def _uppercase_identifiers(text: str) -> set[str]:
    return {match.group(0) for match in _UPPERCASE_IDENTIFIER_PATTERN.finditer(text)}


def unknown_constant_penalty(
    output: ClarifiedFunctionOutput,
    raw_code: str,
    target_clean_code: str,
    compile_reference_source: str = "",
) -> float:
    allowed = (
        _uppercase_identifiers(raw_code)
        | _uppercase_identifiers(target_clean_code)
        | _uppercase_identifiers(compile_reference_source)
        | _COMMON_C_CONSTANTS
    )
    observed = _uppercase_identifiers(output.cleaned_c)
    unsupported = observed - allowed
    return float(min(3, len(unsupported)) / 3)


def unsupported_bool_penalty(
    output: ClarifiedFunctionOutput,
    target_clean_code: str,
    compile_reference_source: str = "",
) -> float:
    if "<stdbool.h>" in compile_reference_source:
        return 0.0
    if _BOOL_TOKEN_PATTERN.search(target_clean_code) or _BOOL_TOKEN_PATTERN.search(
        compile_reference_source
    ):
        return 0.0
    return 1.0 if _BOOL_TOKEN_PATTERN.search(output.cleaned_c) else 0.0


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
    max_completion_ratio: float = _MAX_COMPLETION_RATIO,
    max_function_count: int = _MAX_FUNCTION_COUNT,
    compile_reference_source: str = "",
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
    overshoot_value = overshoot_penalty(
        output.cleaned_c,
        raw_code,
        max_completion_ratio=max_completion_ratio,
    )
    multi_function_value = multi_function_penalty(
        output.cleaned_c,
        max_function_count=max_function_count,
    )
    function_count = float(_function_definition_count(output.cleaned_c))
    cleanup_value = cleanup_reward(output, raw_code)
    naming_value = naming_reward(output, target_renamings)
    compile_value = compile_reward(compile_success)
    effective_behavior_score = (
        behavior_score if behavior_score is not None else (1.0 if behavior_success else 0.0)
    )
    if not compile_success:
        effective_behavior_score = 0.0
    behavior_value = behavior_reward(
        effective_behavior_score if behavior_improvement else 0.0
    )
    readability_value = readability_reward(output, raw_code)
    signature_value = signature_reward(output, target_clean_code, source_function_name)
    hallucination_value = hallucination_penalty(output, allowed_imports, allowed_callees)
    unknown_constant_value = unknown_constant_penalty(
        output,
        raw_code,
        target_clean_code,
        compile_reference_source,
    )
    unsupported_bool_value = unsupported_bool_penalty(
        output,
        target_clean_code,
        compile_reference_source,
    )
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
    style_total = 0.0
    if compile_success:
        style_total = gate_factor * (
            weights.get("cleanup", 1.0) * cleanup_value * style_scales["cleanup"]
            + weights.get("naming", 1.0) * naming_value * style_scales["naming"]
            + weights.get("readability", 1.0) * readability_value * style_scales["readability"]
        )
    penalty_total = (
        weights.get("hallucination_penalty", 1.0) * hallucination_value
        + weights.get("unknown_constant_penalty", 0.0) * unknown_constant_value
        + weights.get("unsupported_bool_penalty", 0.0) * unsupported_bool_value
        + weights.get("decompiler_type_penalty", 0.0) * decompiler_type_value
        + weights.get("overshoot_penalty", 0.0) * overshoot_value
        + weights.get("multi_function_penalty", 0.0) * multi_function_value
    )
    failure_penalty = 0.0
    if not compile_success:
        failure_penalty += weights.get("compile", 1.0) * _COMPILE_FAILURE_PENALTY
    if compile_success and not behavior_success:
        failure_penalty += weights.get("behavior", 1.0) * _BEHAVIOR_FAILURE_PENALTY
    core_total -= failure_penalty
    if not compile_success:
        core_total = min(core_total, 0.0)
    constraint_total = -penalty_total
    total = core_total + style_total + constraint_total
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
        "unknown_constant_penalty": unknown_constant_value,
        "unsupported_bool_penalty": unsupported_bool_value,
        "decompiler_type_penalty": decompiler_type_value,
        "failure_penalty": failure_penalty,
        "gate_factor": gate_factor,
        "exact_json": 1.0 if exact_json else 0.0,
        "completion_ratio": completion_ratio,
        "completeness": completeness,
        "behavior_from_execution": 1.0 if behavior_from_execution else 0.0,
        "overshoot_penalty": overshoot_value,
        "multi_function_penalty": multi_function_value,
        "function_count": function_count,
        "invalid_length_penalty": 0.0,
        "truncation_penalty": 0.0,
        "invalid_scope_penalty": 0.0,
        "raw_completion_ratio": 0.0,
        "core_total": core_total,
        "style_total": style_total,
        "constraint_total": constraint_total,
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
    max_completion_ratio: float = _MAX_COMPLETION_RATIO,
    max_function_count: int = _MAX_FUNCTION_COUNT,
    compile_reference_source: str = "",
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
        max_completion_ratio=max_completion_ratio,
        max_function_count=max_function_count,
        compile_reference_source=compile_reference_source,
    )["total"]
