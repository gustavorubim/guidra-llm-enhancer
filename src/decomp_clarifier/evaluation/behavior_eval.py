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
