from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from decomp_clarifier.dataset.prompt_formatter import format_prompt
from decomp_clarifier.inference.formatter import normalize_output_with_status
from decomp_clarifier.schemas.dataset import FunctionDatasetSample
from decomp_clarifier.schemas.model_io import PredictionRecord
from decomp_clarifier.settings import TrainingConfig
from decomp_clarifier.training.windows_guard import (
    ensure_windows_cuda,
    prepare_model_runtime_environment,
)


def _checkpoint_has_model_artifacts(checkpoint_dir: Path) -> bool:
    return any(
        (checkpoint_dir / filename).exists()
        for filename in (
            "adapter_config.json",
            "config.json",
            "adapter_model.safetensors",
            "model.safetensors",
            "pytorch_model.bin",
        )
    )


def validate_checkpoint_dir(checkpoint_dir: Path) -> None:
    if not checkpoint_dir.exists():
        raise FileNotFoundError(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        raise NotADirectoryError(checkpoint_dir)
    if not _checkpoint_has_model_artifacts(checkpoint_dir):
        raise FileNotFoundError(
            f"Checkpoint directory {checkpoint_dir} does not contain adapter or model files yet."
        )


def _local_model_dir(model_source: str | Path) -> Path | None:
    if isinstance(model_source, Path):
        return model_source
    candidate = Path(model_source).expanduser()
    if candidate.exists():
        return candidate
    if model_source.startswith((".", "~")) or candidate.is_absolute():
        return candidate
    return None


def _text_tokenizer(tokenizer_or_processor: Any) -> Any:
    return getattr(tokenizer_or_processor, "tokenizer", tokenizer_or_processor)


def _encode_prompt(tokenizer_or_processor: Any, prompt: str) -> Any:
    if hasattr(tokenizer_or_processor, "tokenizer"):
        return tokenizer_or_processor(text=prompt, return_tensors="pt")
    return tokenizer_or_processor(prompt, return_tensors="pt")


def _prepare_generation_prompt(
    tokenizer_or_processor: Any,
    text_tokenizer: Any,
    prompt: str,
    *,
    enable_thinking: bool = False,
) -> Any:
    if hasattr(text_tokenizer, "apply_chat_template"):
        try:
            kwargs: dict[str, Any] = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if enable_thinking:
                kwargs["enable_thinking"] = True
            rendered_prompt = text_tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                **kwargs,
            )
            return _encode_prompt(text_tokenizer, rendered_prompt)
        except Exception:  # noqa: BLE001 - fall back to raw prompt on tokenizer/template mismatch
            pass
    return _encode_prompt(text_tokenizer, prompt)


class CheckpointPredictor:
    def __init__(
        self,
        checkpoint_dir: Path | str,
        config: TrainingConfig,
        *,
        prompt_formatter: Callable[[FunctionDatasetSample], str] = format_prompt,
        enable_thinking: bool = False,
    ) -> None:
        ensure_windows_cuda()
        prepare_model_runtime_environment()
        local_dir = _local_model_dir(checkpoint_dir)
        if local_dir is not None:
            validate_checkpoint_dir(local_dir)

        import torch  # type: ignore[import-not-found]
        import unsloth  # noqa: F401 - import before transformers-backed loader  # type: ignore[import-not-found]
        from unsloth import FastLanguageModel  # type: ignore[import-not-found]

        self._torch = torch
        self.checkpoint_dir = local_dir if local_dir is not None else Path(str(checkpoint_dir))
        self.model_source = str(local_dir if local_dir is not None else checkpoint_dir)
        self.prompt_formatter = prompt_formatter
        self.enable_thinking = enable_thinking
        max_seq_length = config.training.max_seq_length or 512
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_source,
            max_seq_length=max_seq_length,
            load_in_4bit=bool(config.training.load_in_4bit),
            device_map="cuda:0",
        )
        self.text_tokenizer = _text_tokenizer(self.tokenizer)
        if self.text_tokenizer.pad_token is None:
            self.text_tokenizer.pad_token = self.text_tokenizer.eos_token
        self.model.eval()

    def generate_text(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        prompt = f"{prompt}\n\n"
        inputs = _prepare_generation_prompt(
            self.tokenizer,
            self.text_tokenizer,
            prompt,
            enable_thinking=self.enable_thinking,
        )
        inputs = {name: value.to("cuda:0") for name, value in inputs.items()}

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.text_tokenizer.pad_token_id,
            "eos_token_id": self.text_tokenizer.eos_token_id,
            "use_cache": True,
        }
        if temperature > 0.0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = 0.95
        else:
            generation_kwargs["do_sample"] = False

        with self._torch.inference_mode():
            output_ids = self.model.generate(**inputs, **generation_kwargs)

        prompt_length = int(inputs["input_ids"].shape[1])
        generated_ids = output_ids[0][prompt_length:]
        return self.text_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def predict(
        self,
        sample: FunctionDatasetSample,
        *,
        system: str,
        max_new_tokens: int,
        temperature: float,
    ) -> PredictionRecord:
        raw_text = self.generate_text(
            self.prompt_formatter(sample),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        output, json_valid = normalize_output_with_status(
            raw_text,
            strip_thinking=self.enable_thinking,
        )
        return PredictionRecord(
            sample_id=sample.sample_id,
            system=system,
            output=output,
            raw_text=raw_text,
            json_valid=json_valid,
        )
