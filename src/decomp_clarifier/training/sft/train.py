from __future__ import annotations

import json
from pathlib import Path

from decomp_clarifier.settings import TrainingConfig
from decomp_clarifier.training.sft.data import combine_prompt_and_response
from decomp_clarifier.training.sft.model import load_model_and_tokenizer
from decomp_clarifier.training.utils.hardware import detect_hardware
from decomp_clarifier.training.utils.version_lock import validate_version_lock
from decomp_clarifier.training.windows_guard import ensure_windows_cuda


def run_sft_training(dataset_path: Path, output_dir: Path, config: TrainingConfig) -> Path:
    ensure_windows_cuda()
    versions = validate_version_lock()
    hardware = detect_hardware()

    from datasets import load_dataset  # type: ignore[import-not-found]
    from trl import SFTConfig, SFTTrainer  # type: ignore[import-not-found]

    model, tokenizer = load_model_and_tokenizer(config)
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    dataset = dataset.map(lambda row: {"text": combine_prompt_and_response(row)})

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=str(output_dir),
            max_seq_length=config.training.max_seq_length or 4096,
            per_device_train_batch_size=config.training.batch_size or 1,
            gradient_accumulation_steps=config.training.grad_accum_steps or 1,
            num_train_epochs=config.training.epochs or 1,
            learning_rate=2e-4,
            logging_steps=1,
        ),
        dataset_text_field="text",
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    manifest_path = output_dir / "sft_training_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {"versions": versions, "hardware": hardware, "dataset": str(dataset_path)},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return manifest_path
