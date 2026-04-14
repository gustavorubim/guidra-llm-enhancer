from __future__ import annotations

from decomp_clarifier.baselines.openrouter_structured import OpenRouterStructuredBaselinePredictor
from decomp_clarifier.dataset.prompt_formatter import format_rl_prompt


def test_openrouter_structured_baseline_returns_valid_prediction(sample_dataset_samples) -> None:
    sample = sample_dataset_samples[0]

    class DummyClient:
        def generate_json(self, **kwargs):
            assert kwargs["model"] == "Qwen/Qwen3.5-2B"
            assert kwargs["max_tokens"] == 384
            assert kwargs["temperature"] == 0.0
            assert kwargs["schema_version"] == "base-qwen-openrouter-baseline-base_qwen_openrouter"
            assert kwargs["messages"] == [{"role": "user", "content": format_rl_prompt(sample)}]
            return {
                "summary": "remote",
                "confidence": 0.8,
                "renamings": {"param_1": "count"},
                "cleaned_c": "int demo(int count) { return count + 1; }",
            }

    predictor = OpenRouterStructuredBaselinePredictor(
        client=DummyClient(),
        model="Qwen/Qwen3.5-2B",
        prompt_formatter=format_rl_prompt,
        schema_version="base-qwen-openrouter-baseline",
    )

    record = predictor.predict(
        sample,
        system="base_qwen_openrouter",
        max_new_tokens=384,
        temperature=0.0,
    )

    assert record.system == "base_qwen_openrouter"
    assert record.json_valid is True
    assert record.output.summary == "remote"
    assert record.output.renamings == {"param_1": "count"}
    assert record.raw_text is not None
    assert '"summary": "remote"' in record.raw_text


def test_openrouter_structured_baseline_falls_back_to_raw_ghidra(sample_dataset_samples) -> None:
    sample = sample_dataset_samples[0]

    class BrokenClient:
        def generate_json(self, **kwargs):
            raise RuntimeError("boom")

    predictor = OpenRouterStructuredBaselinePredictor(
        client=BrokenClient(),
        model="Qwen/Qwen3.5-2B",
        prompt_formatter=format_rl_prompt,
    )

    record = predictor.predict(
        sample,
        system="base_qwen_openrouter",
        max_new_tokens=384,
        temperature=0.0,
    )

    assert record.system == "base_qwen_openrouter"
    assert record.json_valid is False
    assert record.output.cleaned_c == sample.ghidra_decompiled_code
    assert record.output.summary == "Raw Ghidra decompiler output without clarification."
    assert record.raw_text == "OPENROUTER_ERROR: boom"
