from __future__ import annotations

from typing import Any

from decomp_clarifier.settings import TrainingConfig


def load_model_and_tokenizer(config: TrainingConfig) -> tuple[Any, Any]:
    from unsloth import FastLanguageModel  # type: ignore[import-not-found]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model.base_model_id,
        max_seq_length=config.training.max_seq_length,
        load_in_4bit=bool(config.training.load_in_4bit),
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.training.lora_rank or 16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=(config.training.lora_rank or 16) * 2,
        use_gradient_checkpointing=True,
    )
    return model, tokenizer
