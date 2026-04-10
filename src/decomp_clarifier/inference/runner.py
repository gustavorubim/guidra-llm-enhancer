from __future__ import annotations

from collections.abc import Callable

from decomp_clarifier.schemas.dataset import FunctionDatasetSample
from decomp_clarifier.schemas.model_io import ClarifiedFunctionOutput


class InferenceRunner:
    def __init__(
        self, predictor: Callable[[FunctionDatasetSample], ClarifiedFunctionOutput]
    ) -> None:
        self.predictor = predictor

    def run(self, samples: list[FunctionDatasetSample]) -> list[ClarifiedFunctionOutput]:
        return [self.predictor(sample) for sample in samples]
