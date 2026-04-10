from __future__ import annotations

import re


def _tokenize(code: str) -> set[str]:
    return set(re.findall(r"[A-Za-z_]\w*", code))


def behavior_similarity(candidate_code: str, reference_code: str) -> float:
    candidate_tokens = _tokenize(candidate_code)
    reference_tokens = _tokenize(reference_code)
    if not reference_tokens:
        return 0.0
    overlap = len(candidate_tokens & reference_tokens)
    union = len(candidate_tokens | reference_tokens)
    return overlap / union if union else 0.0


def is_behavior_improvement(
    candidate_code: str,
    raw_code: str,
    reference_code: str,
) -> bool:
    sim_to_ref = behavior_similarity(candidate_code, reference_code)
    sim_to_raw = behavior_similarity(candidate_code, raw_code)
    # Treat ties as non-regressions because the token-overlap proxy is weak.
    return sim_to_ref >= sim_to_raw
