from __future__ import annotations

import json
from pathlib import Path

from decomp_clarifier.settings import TrainingConfig
from decomp_clarifier.training.sft.data import combine_prompt_and_response
from decomp_clarifier.training.sft.model import load_model_and_tokenizer
from decomp_clarifier.training.utils.hardware import detect_hardware
from decomp_clarifier.training.utils.telemetry import (
    TrainingTelemetry,
    TrainingTelemetryCallback,
)
from decomp_clarifier.training.utils.version_lock import validate_version_lock
from decomp_clarifier.training.windows_guard import ensure_windows_cuda


def run_sft_training(dataset_path: Path, output_dir: Path, config: TrainingConfig) -> Path:
    ensure_windows_cuda()
    versions = validate_version_lock()
    hardware = detect_hardware()
    telemetry = TrainingTelemetry("sft", output_dir)

    import unsloth  # noqa: F401 - must be imported before trl/transformers  # type: ignore[import-not-found]
    from datasets import load_dataset  # type: ignore[import-not-found]
    from trl import SFTConfig, SFTTrainer  # type: ignore[import-not-found]

    model, tokenizer = load_model_and_tokenizer(config)
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    if config.training.min_train_samples is not None and len(dataset) < config.training.min_train_samples:
        raise ValueError(
            f"training dataset has only {len(dataset)} records; "
            f"min_train_samples requires at least {config.training.min_train_samples}"
        )
    dataset = dataset.map(lambda row: {"text": combine_prompt_and_response(row)})

    max_length = config.training.max_seq_length or 512
    max_steps = config.training.max_steps if getattr(config.training, "max_steps", None) is not None else -1
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=str(output_dir),
            logging_dir=str(telemetry.tensorboard_dir),
            logging_first_step=True,
            max_length=max_length,
            per_device_train_batch_size=config.training.batch_size or 1,
            gradient_accumulation_steps=config.training.grad_accum_steps or 1,
            num_train_epochs=config.training.epochs or 1,
            max_steps=max_steps,
            learning_rate=2e-4,
            logging_steps=1,
            logging_strategy="steps",
            report_to=["tensorboard"],
        ),
        dataset_text_field="text",
        callbacks=[TrainingTelemetryCallback(telemetry)],
    )
    train_result = trainer.train()
    trainer.save_model(str(output_dir))
    telemetry_summary = telemetry.finalize(
        trainer=trainer,
        final_metrics=getattr(train_result, "metrics", None),
    )
    manifest_path = output_dir / "sft_training_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "versions": versions,
                "hardware": hardware,
                "dataset": str(dataset_path),
                "telemetry": telemetry_summary,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return manifest_path
