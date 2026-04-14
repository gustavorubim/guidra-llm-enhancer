from __future__ import annotations

import json
import logging
from pathlib import Path

from decomp_clarifier.evaluation.behavior_eval import behavior_similarity, is_behavior_improvement
from decomp_clarifier.evaluation.compile_eval import compile_candidate
from decomp_clarifier.inference.formatter import normalize_output_with_status
from decomp_clarifier.settings import TrainingConfig
from decomp_clarifier.training.grpo.data import prompt_from_record, reward_fields_from_record
from decomp_clarifier.training.grpo.rewards import reward_breakdown
from decomp_clarifier.training.utils.hardware import detect_hardware
from decomp_clarifier.training.utils.telemetry import (
    TrainingTelemetry,
    create_training_telemetry_callback,
    reward_log_row,
)
from decomp_clarifier.training.utils.trl_compat import (
    ensure_model_warnings_issued,
    patch_trl_optional_availability,
)
from decomp_clarifier.training.utils.version_lock import validate_version_lock
from decomp_clarifier.training.windows_guard import (
    ensure_windows_cuda,
    prepare_model_runtime_environment,
)

_BEHAVIOR_THRESHOLD = 0.35


def _dataset_size(dataset: object) -> int | None:
    try:
        return len(dataset)  # type: ignore[arg-type]
    except TypeError:
        return None


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def compute_completion_reward(
    completion: str,
    source_function_name: str,
    raw_code: str,
    compile_reference_source: str,
    target_clean_code: str,
    target_renamings_json: str,
    allowed_imports_json: str,
    allowed_callees_json: str,
    weights: dict[str, float],
    behavior_threshold: float = _BEHAVIOR_THRESHOLD,
) -> float:
    return compute_completion_reward_details(
        completion=completion,
        source_function_name=source_function_name,
        raw_code=raw_code,
        compile_reference_source=compile_reference_source,
        target_clean_code=target_clean_code,
        target_renamings_json=target_renamings_json,
        allowed_imports_json=allowed_imports_json,
        allowed_callees_json=allowed_callees_json,
        weights=weights,
        behavior_threshold=behavior_threshold,
    )["total"]


def compute_completion_reward_details(
    completion: str,
    source_function_name: str,
    raw_code: str,
    compile_reference_source: str,
    target_clean_code: str,
    target_renamings_json: str,
    allowed_imports_json: str,
    allowed_callees_json: str,
    weights: dict[str, float],
    behavior_threshold: float = _BEHAVIOR_THRESHOLD,
) -> dict[str, float]:
    try:
        output, json_valid = normalize_output_with_status(completion)
        if not json_valid:
            return {
                "json_valid": 0.0,
                "format": 0.0,
                "cleanup": 0.0,
                "naming": 0.0,
                "compile": 0.0,
                "behavior": 0.0,
                "readability": 0.0,
                "signature": 0.0,
                "hallucination_penalty": 0.0,
                "decompiler_type_penalty": 0.0,
                "gate_factor": 0.0,
                "total": 0.0,
            }
        renamings: dict[str, str] = json.loads(target_renamings_json)
        imports: list[str] = json.loads(allowed_imports_json)
        callees: list[str] = json.loads(allowed_callees_json)
        compile_success = compile_candidate(
            output.cleaned_c,
            compile_reference_source or target_clean_code,
            function_name=source_function_name or None,
        )
        behavior_score = behavior_similarity(output.cleaned_c, target_clean_code)
        behavior_success = behavior_score >= behavior_threshold and is_behavior_improvement(
            output.cleaned_c,
            raw_code,
            target_clean_code,
        )
        return reward_breakdown(
            output=output,
            json_valid=json_valid,
            raw_code=raw_code,
            target_clean_code=target_clean_code,
            source_function_name=source_function_name,
            target_renamings=renamings,
            compile_success=compile_success,
            behavior_success=behavior_success,
            allowed_imports=imports,
            allowed_callees=callees,
            weights=weights,
        )
    except Exception:  # noqa: BLE001
        return {
            "json_valid": 0.0,
            "format": 0.0,
            "cleanup": 0.0,
            "naming": 0.0,
            "compile": 0.0,
            "behavior": 0.0,
            "readability": 0.0,
            "signature": 0.0,
            "hallucination_penalty": 0.0,
            "decompiler_type_penalty": 0.0,
            "gate_factor": 0.0,
            "total": 0.0,
        }


def run_grpo_training(dataset_path: Path, output_dir: Path, config: TrainingConfig) -> Path:
    logger = logging.getLogger("decomp_clarifier")
    ensure_windows_cuda()
    prepare_model_runtime_environment()
    versions = validate_version_lock()
    hardware = detect_hardware()
    telemetry = TrainingTelemetry("grpo", output_dir)
    logger.info(
        "starting grpo training dataset=%s output_dir=%s model=%s",
        dataset_path,
        output_dir,
        config.model.base_model_id,
    )

    import unsloth  # noqa: F401 - must be imported before trl/transformers  # type: ignore[import-not-found]
    from datasets import load_dataset  # type: ignore[import-not-found]

    patch_trl_optional_availability()
    from trl import GRPOConfig, GRPOTrainer  # type: ignore[import-not-found]

    from decomp_clarifier.training.sft.model import load_model_and_tokenizer

    model, tokenizer = load_model_and_tokenizer(config)
    patched_model_count = ensure_model_warnings_issued(model)
    logger.info("patched warnings_issued on grpo model_chain_nodes=%s", patched_model_count)
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    dataset = dataset.map(
        lambda row: {"prompt": prompt_from_record(row), **reward_fields_from_record(row)}
    )
    dataset_size = _dataset_size(dataset)
    logger.info(
        "loaded grpo dataset rows=%s",
        dataset_size if dataset_size is not None else "unknown",
    )

    weights = config.training.reward_weights
    behavior_threshold = (
        config.training.behavior_similarity_threshold
        if config.training.behavior_similarity_threshold is not None
        else _BEHAVIOR_THRESHOLD
    )
    batch_size = config.training.batch_size or config.hardware.batch_size or 1
    grad_accum_steps = config.training.grad_accum_steps or 1
    epochs = config.training.epochs or 1
    num_generations = config.training.generations_per_prompt or 4
    generation_batch_size = batch_size * max(grad_accum_steps, num_generations)
    learning_rate = (
        config.training.learning_rate if config.training.learning_rate is not None else 5e-6
    )
    adam_beta1 = config.training.adam_beta1 if config.training.adam_beta1 is not None else 0.9
    adam_beta2 = (
        config.training.adam_beta2 if config.training.adam_beta2 is not None else 0.99
    )
    weight_decay = (
        config.training.weight_decay if config.training.weight_decay is not None else 0.1
    )
    warmup_ratio = (
        config.training.warmup_ratio if config.training.warmup_ratio is not None else 0.1
    )
    lr_scheduler_type = config.training.lr_scheduler_type or "cosine"
    optim = config.training.optim or "adamw_8bit"
    max_grad_norm = (
        config.training.max_grad_norm if config.training.max_grad_norm is not None else 0.1
    )
    reward_step = 0

    def reward_func(
        completions: list[str],
        source_function_name: list[str] | None = None,
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
        function_names = source_function_name or [""] * n
        raw_codes = raw_code or [""] * n
        compile_sources = compile_reference_source or [""] * n
        target_codes = target_clean_code or [""] * n
        renaming_maps = target_renamings or ["{}"] * n
        import_lists = allowed_imports or ["[]"] * n
        callee_lists = allowed_callees or ["[]"] * n
        details = [
            compute_completion_reward_details(
                completion=completion,
                source_function_name=function_names[index % len(function_names)],
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
        rewards = [item["total"] for item in details]
        reward_step += 1
        reward_metrics = reward_log_row(rewards, step=reward_step)
        component_names = sorted({key for item in details for key in item if key != "total"})
        reward_metrics |= {
            f"components/{name}_mean": _mean([item[name] for item in details])
            for name in component_names
        }
        telemetry.record_metrics(
            {key: value for key, value in reward_metrics.items() if key != "step"},
            step=reward_metrics["step"],
            source="reward_func",
        )
        return rewards

    max_steps = config.training.max_steps if config.training.max_steps is not None else -1
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_func],
        train_dataset=dataset,
        args=GRPOConfig(
            output_dir=str(output_dir),
            logging_dir=str(telemetry.tensorboard_dir),
            logging_first_step=True,
            logging_steps=1,
            logging_strategy="steps",
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum_steps,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type=lr_scheduler_type,
            optim=optim,
            max_prompt_length=config.training.max_prompt_length or 512,
            max_completion_length=config.training.max_completion_length or 256,
            num_generations=num_generations,
            generation_batch_size=generation_batch_size,
            max_steps=max_steps,
            max_grad_norm=max_grad_norm,
            save_steps=(
                config.training.save_steps
                or (max_steps if max_steps > 0 else 100)
            ),
            report_to=["tensorboard"],
        ),
        callbacks=[create_training_telemetry_callback(telemetry)],
    )
    logger.info(
        "configured grpo trainer batch_size=%s grad_accum=%s epochs=%s "
        "generation_batch_size=%s learning_rate=%s scheduler=%s "
        "max_prompt_length=%s max_completion_length=%s generations=%s max_steps=%s "
        "save_steps=%s",
        batch_size,
        grad_accum_steps,
        epochs,
        generation_batch_size,
        learning_rate,
        lr_scheduler_type,
        config.training.max_prompt_length or 512,
        config.training.max_completion_length or 256,
        num_generations,
        max_steps,
        config.training.save_steps or (max_steps if max_steps > 0 else 100),
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
    logger.info(
        "finished grpo training manifest=%s telemetry_jsonl=%s tensorboard_dir=%s",
        manifest_path,
        telemetry_summary["metrics_jsonl"],
        telemetry_summary["tensorboard_dir"],
    )
    return manifest_path
