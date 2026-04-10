from __future__ import annotations

import json
from pathlib import Path

from decomp_clarifier.settings import TrainingConfig
from decomp_clarifier.training.grpo.data import prompt_from_record
from decomp_clarifier.training.sft.model import load_model_and_tokenizer
from decomp_clarifier.training.utils.hardware import detect_hardware
from decomp_clarifier.training.utils.version_lock import validate_version_lock
from decomp_clarifier.training.windows_guard import ensure_windows_cuda


def run_grpo_training(dataset_path: Path, output_dir: Path, config: TrainingConfig) -> Path:
    ensure_windows_cuda()
    versions = validate_version_lock()
    hardware = detect_hardware()

    from datasets import load_dataset  # type: ignore[import-not-found]
    from trl import GRPOConfig, GRPOTrainer  # type: ignore[import-not-found]

    model, tokenizer = load_model_and_tokenizer(config)
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    dataset = dataset.map(lambda row: {"prompt": prompt_from_record(row)})

    def reward_func(completions: list[str], **_: object) -> list[float]:
        return [0.0 for _completion in completions]

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=[reward_func],
        train_dataset=dataset,
        args=GRPOConfig(
            output_dir=str(output_dir),
            max_prompt_length=config.training.max_prompt_length or 3072,
            max_completion_length=config.training.max_completion_length or 1024,
            num_generations=config.training.generations_per_prompt or 4,
        ),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    manifest_path = output_dir / "grpo_training_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {"versions": versions, "hardware": hardware, "dataset": str(dataset_path)},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return manifest_path
