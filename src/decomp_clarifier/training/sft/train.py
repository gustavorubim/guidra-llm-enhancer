from __future__ import annotations

import json
import logging
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


def _dataset_size(dataset: object) -> int | None:
    try:
        return len(dataset)  # type: ignore[arg-type]
    except TypeError:
        return None


def run_sft_training(dataset_path: Path, output_dir: Path, config: TrainingConfig) -> Path:
    logger = logging.getLogger("decomp_clarifier")
    ensure_windows_cuda()
    versions = validate_version_lock()
    hardware = detect_hardware()
    telemetry = TrainingTelemetry("sft", output_dir)
    logger.info(
        "starting sft training dataset=%s output_dir=%s model=%s",
        dataset_path,
        output_dir,
        config.model.base_model_id,
    )

    import unsloth  # noqa: F401 - must be imported before trl/transformers  # type: ignore[import-not-found]
    from datasets import load_dataset  # type: ignore[import-not-found]
    from trl import SFTConfig, SFTTrainer  # type: ignore[import-not-found]

    model, tokenizer = load_model_and_tokenizer(config)
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    dataset_size = _dataset_size(dataset)
    logger.info(
        "loaded sft dataset rows=%s",
        dataset_size if dataset_size is not None else "unknown",
    )
    if (
        config.training.min_train_samples is not None
        and dataset_size is not None
        and dataset_size < config.training.min_train_samples
    ):
        raise ValueError(
            f"training dataset has only {dataset_size} records; "
            f"min_train_samples requires at least {config.training.min_train_samples}"
        )
    dataset = dataset.map(lambda row: {"text": combine_prompt_and_response(row)})

    max_length = config.training.max_seq_length or 512
    max_steps = (
        config.training.max_steps
        if getattr(config.training, "max_steps", None) is not None
        else -1
    )
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
    logger.info(
        "configured sft trainer max_length=%s batch_size=%s grad_accum=%s epochs=%s max_steps=%s",
        max_length,
        config.training.batch_size or 1,
        config.training.grad_accum_steps or 1,
        config.training.epochs or 1,
        max_steps,
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
    logger.info(
        "finished sft training manifest=%s telemetry_jsonl=%s tensorboard_dir=%s",
        manifest_path,
        telemetry_summary["metrics_jsonl"],
        telemetry_summary["tensorboard_dir"],
    )
    return manifest_path
