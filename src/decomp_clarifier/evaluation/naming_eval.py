from __future__ import annotations

from difflib import SequenceMatcher


def normalized_name_similarity(expected: dict[str, str], predicted: dict[str, str]) -> float:
    if not expected:
        return 1.0
    scores: list[float] = []
    for key, target in expected.items():
        candidate = predicted.get(key, "")
        scores.append(SequenceMatcher(a=target, b=candidate).ratio())
    return sum(scores) / len(scores)
