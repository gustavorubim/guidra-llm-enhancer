from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from decomp_clarifier.baselines import raw_ghidra
from decomp_clarifier.schemas.dataset import FunctionDatasetSample
from decomp_clarifier.schemas.model_io import ClarifiedFunctionOutput, PredictionRecord


class OpenRouterStructuredBaselinePredictor:
    def __init__(
        self,
        *,
        client: Any,
        model: str,
        prompt_formatter: Callable[[FunctionDatasetSample], str],
        fallback_predictor: Callable[[FunctionDatasetSample], ClarifiedFunctionOutput] = (
            raw_ghidra.predict
        ),
        schema_version: str = "structured-baseline",
    ) -> None:
        self.client = client
        self.model = model
        self.prompt_formatter = prompt_formatter
        self.fallback_predictor = fallback_predictor
        self.schema_version = schema_version

    def predict(
        self,
        sample: FunctionDatasetSample,
        *,
        system: str,
        max_new_tokens: int,
        temperature: float,
    ) -> PredictionRecord:
        prompt = self.prompt_formatter(sample)
        try:
            payload = self.client.generate_json(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=temperature,
                response_schema=ClarifiedFunctionOutput.model_json_schema(),
                schema_version=f"{self.schema_version}-{system}",
            )
            output = ClarifiedFunctionOutput.model_validate(json.loads(json.dumps(payload)))
            return PredictionRecord(
                sample_id=sample.sample_id,
                system=system,
                output=output,
                raw_text=json.dumps(payload, sort_keys=True),
                json_valid=True,
            )
        except Exception as exc:  # noqa: BLE001 - baseline runs should degrade to inspectable failures
            return PredictionRecord(
                sample_id=sample.sample_id,
                system=system,
                output=self.fallback_predictor(sample),
                raw_text=f"OPENROUTER_ERROR: {exc}",
                json_valid=False,
            )
