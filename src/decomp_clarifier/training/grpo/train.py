from __future__ import annotations

import json
from pathlib import Path

from decomp_clarifier.evaluation.behavior_eval import behavior_similarity, is_behavior_improvement
from decomp_clarifier.evaluation.compile_eval import compile_candidate
from decomp_clarifier.settings import TrainingConfig
from decomp_clarifier.training.grpo.data import prompt_from_record, reward_fields_from_record
from decomp_clarifier.training.grpo.rewards import weighted_reward
from decomp_clarifier.training.grpo.rollout import normalize_completion
from decomp_clarifier.training.sft.model import load_model_and_tokenizer
from decomp_clarifier.training.utils.hardware import detect_hardware
from decomp_clarifier.training.utils.telemetry import (
    TrainingTelemetry,
    TrainingTelemetryCallback,
    reward_log_row,
)
from decomp_clarifier.training.utils.version_lock import validate_version_lock
from decomp_clarifier.training.windows_guard import ensure_windows_cuda

_BEHAVIOR_THRESHOLD = 0.35


def compute_completion_reward(
    completion: str,
    raw_code: str,
    compile_reference_source: str,
    target_clean_code: str,
    target_renamings_json: str,
    allowed_imports_json: str,
    allowed_callees_json: str,
    weights: dict[str, float],
    behavior_threshold: float = _BEHAVIOR_THRESHOLD,
) -> float:
    try:
        output = normalize_completion(completion)
        renamings: dict[str, str] = json.loads(target_renamings_json)
        imports: list[str] = json.loads(allowed_imports_json)
        callees: list[str] = json.loads(allowed_callees_json)
        compile_success = compile_candidate(
            output.cleaned_c, compile_reference_source or target_clean_code
        )
        behavior_score = behavior_similarity(output.cleaned_c, target_clean_code)
        behavior_success = behavior_score >= behavior_threshold and is_behavior_improvement(
            output.cleaned_c,
            raw_code,
            target_clean_code,
        )
        return weighted_reward(
            output=output,
            raw_code=raw_code,
            target_renamings=renamings,
            compile_success=compile_success,
            behavior_success=behavior_success,
            allowed_imports=imports,
            allowed_callees=callees,
            weights=weights,
        )
    except Exception:  # noqa: BLE001
        return 0.0


def run_grpo_training(dataset_path: Path, output_dir: Path, config: TrainingConfig) -> Path:
    ensure_windows_cuda()
    versions = validate_version_lock()
    hardware = detect_hardware()
    telemetry = TrainingTelemetry("grpo", output_dir)

    import unsloth  # noqa: F401 - must be imported before trl/transformers  # type: ignore[import-not-found]
    from datasets import load_dataset  # type: ignore[import-not-found]
    from trl import GRPOConfig, GRPOTrainer  # type: ignore[import-not-found]

    model, tokenizer = load_model_and_tokenizer(config)
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    dataset = dataset.map(
        lambda row: {"prompt": prompt_from_record(row), **reward_fields_from_record(row)}
    )

    weights = config.training.reward_weights
    behavior_threshold = (
        config.training.behavior_similarity_threshold
        if config.training.behavior_similarity_threshold is not None
        else _BEHAVIOR_THRESHOLD
    )
    reward_step = 0

    def reward_func(
        completions: list[str],
        *,
        raw_code: list[str] | None = None,
        compile_reference_source: list[str] | None = None,
        target_clean_code: list[str] | None = None,
        target_renamings: list[str] | None = None,
        allowed_imports: list[str] | None = None,
        allowed_callees: list[str] | None = None,
        **_: object,
    ) -> list[float]:
        nonlocal reward_step
        n = len(completions)
        raw_codes = raw_code or [""] * n
        compile_sources = compile_reference_source or [""] * n
        target_codes = target_clean_code or [""] * n
        renaming_maps = target_renamings or ["{}"] * n
        import_lists = allowed_imports or ["[]"] * n
        callee_lists = allowed_callees or ["[]"] * n
        rewards = [
            compute_completion_reward(
                completion=completion,
                raw_code=raw_codes[index % len(raw_codes)],
                compile_reference_source=compile_sources[index % len(compile_sources)],
                target_clean_code=target_codes[index % len(target_codes)],
                target_renamings_json=renaming_maps[index % len(renaming_maps)],
                allowed_imports_json=import_lists[index % len(import_lists)],
                allowed_callees_json=callee_lists[index % len(callee_lists)],
                weights=weights,
                behavior_threshold=behavior_threshold,
            )
            for index, completion in enumerate(completions)
        ]
        reward_step += 1
        reward_metrics = reward_log_row(rewards, step=reward_step)
        telemetry.record_metrics(
            {key: value for key, value in reward_metrics.items() if key != "step"},
            step=reward_metrics["step"],
            source="reward_func",
        )
        return rewards

    max_steps = config.training.max_steps if config.training.max_steps is not None else -1
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=[reward_func],
        train_dataset=dataset,
        args=GRPOConfig(
            output_dir=str(output_dir),
            logging_dir=str(telemetry.tensorboard_dir),
            logging_first_step=True,
            logging_steps=1,
            logging_strategy="steps",
            max_prompt_length=config.training.max_prompt_length or 512,
            max_completion_length=config.training.max_completion_length or 256,
            num_generations=config.training.generations_per_prompt or 4,
            max_steps=max_steps,
            report_to=["tensorboard"],
        ),
        callbacks=[TrainingTelemetryCallback(telemetry)],
    )
    train_result = trainer.train()
    trainer.save_model(str(output_dir))
    telemetry_summary = telemetry.finalize(
        trainer=trainer,
        final_metrics=getattr(train_result, "metrics", None),
    )
    manifest_path = output_dir / "grpo_training_manifest.json"
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
