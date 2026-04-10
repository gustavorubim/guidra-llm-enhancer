from __future__ import annotations

import json
from typing import Any

from decomp_clarifier.generation.prompt_builder import build_cleanup_prompt
from decomp_clarifier.schemas.dataset import FunctionDatasetSample
from decomp_clarifier.schemas.model_io import ClarifiedFunctionOutput, PromptInput


def heuristic_cleanup(sample: FunctionDatasetSample) -> ClarifiedFunctionOutput:
    cleaned = sample.ghidra_decompiled_code.replace("undefined4", "int").replace(
        "undefined8", "long long"
    )
    return ClarifiedFunctionOutput(
        summary=f"Prompt-free heuristic cleanup for {sample.ghidra_function_name}.",
        confidence=0.35,
        renamings={},
        cleaned_c=cleaned,
    )


class PromptOnlyCleanupBaseline:
    def __init__(self, client: Any | None, prompt_template: str, model: str) -> None:
        self.client = client
        self.prompt_template = prompt_template
        self.model = model

    def predict(self, sample: FunctionDatasetSample) -> ClarifiedFunctionOutput:
        if self.client is None:
            return heuristic_cleanup(sample)
        prompt = build_cleanup_prompt(
            self.prompt_template,
            PromptInput(
                task_type=sample.task_type,
                decompiled_code=sample.ghidra_decompiled_code,
                assembly=sample.assembly,
                strings=sample.strings,
                imports=sample.imports,
                callers=sample.callers,
                callees=sample.callees,
                semantic_summary=sample.semantic_summary,
            ),
        )
        payload = self.client.generate_json(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1200,
            temperature=0.2,
            response_schema=ClarifiedFunctionOutput.model_json_schema(),
            schema_version="prompt-only-baseline",
        )
        return ClarifiedFunctionOutput.model_validate(json.loads(json.dumps(payload)))
