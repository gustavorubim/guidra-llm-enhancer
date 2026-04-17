from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Any

from decomp_clarifier.settings import TrainingConfig


def _checkpoint_has_lora_adapters(model_name: str | None) -> bool:
    if not model_name:
        return False
    path = Path(model_name)
    return path.exists() and (path / "adapter_config.json").exists()


def _is_local_model_reference(model_name: str) -> bool:
    return Path(model_name).exists()


def _cached_remote_snapshot_dir(model_name: str) -> Path | None:
    try:
        from huggingface_hub import snapshot_download  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        return Path(snapshot_download(repo_id=model_name, local_files_only=True))
    except Exception:  # noqa: BLE001 - training-only probe against optional dependency
        return None


def _can_resolve_huggingface() -> bool:
    try:
        socket.getaddrinfo("huggingface.co", 443)
    except OSError:
        return False
    return True


def _resolve_model_source(model_name: str | None) -> str:
    if not model_name:
        raise RuntimeError("training config missing model.base_model_id")
    if _is_local_model_reference(model_name):
        return model_name
    snapshot_dir = _cached_remote_snapshot_dir(model_name)
    if snapshot_dir is not None:
        if not _can_resolve_huggingface():
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            return str(snapshot_dir)
        return model_name
    if _can_resolve_huggingface():
        return model_name
    raise RuntimeError(
        "Could not resolve huggingface.co while loading "
        f"{model_name}. The model is not available in the local Hugging Face cache. "
        "Restore DNS/internet access to Hugging Face or change model.base_model_id "
        "to a local checkpoint directory."
    )


def load_model_and_tokenizer(config: TrainingConfig) -> tuple[Any, Any]:
    from unsloth import FastLanguageModel  # type: ignore[import-not-found]

    model_source = _resolve_model_source(config.model.base_model_id)
    max_seq_length = config.training.max_seq_length or 512
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_source,
            max_seq_length=max_seq_length,
            load_in_4bit=bool(config.training.load_in_4bit),
            device_map="cuda:0",
        )
    except RuntimeError as exc:
        if "No config file found" in str(exc):
            raise RuntimeError(
                "Failed to load the configured training model "
                f"{config.model.base_model_id}. If this host is offline, ensure the "
                "model is already cached locally or point model.base_model_id to a "
                "local checkpoint directory."
            ) from exc
        raise
    if _checkpoint_has_lora_adapters(config.model.base_model_id) or (
        "PeftModel" in type(model).__name__
    ):
        return model, tokenizer
    lora_r = config.training.lora_rank or 16
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=lora_r,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )
    return model, tokenizer
