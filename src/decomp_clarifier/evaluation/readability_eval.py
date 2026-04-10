from __future__ import annotations

import re


def score_readability(code: str) -> float:
    stripped = code.strip()
    if not stripped:
        return 0.0
    lines = [line for line in stripped.splitlines() if line.strip()]
    avg_line_length = sum(len(line) for line in lines) / len(lines)
    placeholders = len(
        re.findall(r"\b(?:param_\d+|local_[0-9A-Fa-f]+|iVar\d+|uVar\d+)\b", stripped)
    )
    gotos = stripped.count("goto ")
    raw_score = 1.0
    raw_score -= min(avg_line_length / 240.0, 0.3)
    raw_score -= min(placeholders * 0.05, 0.4)
    raw_score -= min(gotos * 0.1, 0.2)
    return max(0.0, min(raw_score, 1.0))


def readability_improvement(candidate: str, baseline: str) -> float:
    return score_readability(candidate) - score_readability(baseline)
