from __future__ import annotations

from decomp_clarifier.schemas.evaluation import SampleEvaluation
from decomp_clarifier.schemas.model_io import ClarifiedFunctionOutput

PLACEHOLDER_TOKENS = ("param_", "local_", "iVar", "uVar", "puVar", "FUN_")


def field_complete(output: ClarifiedFunctionOutput) -> bool:
    return bool(output.summary.strip() and output.cleaned_c.strip())


def placeholder_ratio(code: str) -> float:
    tokens = [token for token in code.replace("(", " ").replace(")", " ").split() if token]
    if not tokens:
        return 0.0
    placeholder_count = sum(1 for token in tokens if token.startswith(PLACEHOLDER_TOKENS))
    return placeholder_count / len(tokens)


def aggregate_metric(evaluations: list[SampleEvaluation], attribute: str) -> float:
    if not evaluations:
        return 0.0
    values = [float(getattr(item, attribute)) for item in evaluations]
    return sum(values) / len(values)
