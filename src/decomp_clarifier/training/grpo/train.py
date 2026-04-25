from __future__ import annotations

import hashlib
import inspect
import json
import logging
from pathlib import Path

from decomp_clarifier.evaluation.behavior_eval import (
    behavior_similarity,
    evaluate_execution_behavior,
    is_behavior_improvement,
)
from decomp_clarifier.evaluation.compile_eval import compile_candidate
from decomp_clarifier.inference.formatter import normalize_output_with_schema_status
from decomp_clarifier.settings import TrainingConfig
from decomp_clarifier.training.grpo.data import prompt_from_record, reward_fields_from_record
from decomp_clarifier.training.grpo.rewards import (
    empty_reward_breakdown,
    invalid_json_penalty,
    reward_breakdown,
)
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

_BEHAVIOR_SIMILARITY_THRESHOLD = 0.35
_EXECUTION_PASS_RATE_THRESHOLD = 1.0
_MIN_COMPLETION_RATIO = 0.3
_MAX_COMPLETION_RATIO = 1.75
_MAX_INVALID_COMPLETION_RATIO = 0.9
_MAX_FUNCTION_COUNT = 1
_OBJECTIVE_FIELDS: tuple[tuple[str, str], ...] = (
    ("correctness", "core_total"),
    ("style", "style_total"),
    ("constraints", "constraint_total"),
)


def _dataset_size(dataset: object) -> int | None:
    try:
        return len(dataset)  # type: ignore[arg-type]
    except TypeError:
        return None


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _default_multi_reward_weights() -> list[float]:
    return [1.0] * len(_OBJECTIVE_FIELDS)


def _resolve_multi_reward_weights(configured: list[float] | None) -> list[float]:
    resolved = list(configured) if configured is not None else _default_multi_reward_weights()
    if len(resolved) != len(_OBJECTIVE_FIELDS):
        raise ValueError(
            "training.multi_reward_weights must match the number of reward functions: "
            f"expected {len(_OBJECTIVE_FIELDS)}, got {len(resolved)}"
        )
    return resolved


def _trainer_provenance(trainer_cls: type[object]) -> dict[str, object]:
    source_path: str | None = None
    source_sha256: str | None = None
    try:
        raw_source_path = inspect.getsourcefile(trainer_cls) or inspect.getfile(trainer_cls)
    except TypeError:
        raw_source_path = None
    if raw_source_path:
        path = Path(raw_source_path)
        source_path = str(path)
        if path.exists() and path.is_file():
            source_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "class": f"{trainer_cls.__module__}.{trainer_cls.__qualname__}",
        "module": trainer_cls.__module__,
        "source_path": source_path,
        "source_sha256": source_sha256,
    }


def _completion_text(completion: object) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        content = completion.get("content")
        if isinstance(content, str):
            return content
        text = completion.get("text")
        if isinstance(text, str):
            return text
        return json.dumps(completion, sort_keys=True)
    if isinstance(completion, list):
        for item in reversed(completion):
            if (
                isinstance(item, dict)
                and item.get("role") == "assistant"
                and isinstance(item.get("content"), str)
            ):
                return item["content"]
        parts = [_completion_text(item).strip() for item in completion]
        return "\n".join(part for part in parts if part)
    return str(completion)


def compute_completion_reward(
    completion: str,
    task_type: str,
    source_function_name: str,
    raw_code: str,
    compile_reference_source: str,
    target_clean_code: str,
    target_renamings_json: str,
    allowed_imports_json: str,
    allowed_callees_json: str,
    compiler_executable: str | None,
    tests_ref: str,
    weights: dict[str, float],
    behavior_threshold: float = _BEHAVIOR_SIMILARITY_THRESHOLD,
    execution_pass_rate_threshold: float = _EXECUTION_PASS_RATE_THRESHOLD,
    min_completion_ratio: float = _MIN_COMPLETION_RATIO,
    max_completion_ratio: float = _MAX_COMPLETION_RATIO,
    max_invalid_completion_ratio: float = _MAX_INVALID_COMPLETION_RATIO,
    max_function_count: int = _MAX_FUNCTION_COUNT,
) -> float:
    return compute_completion_reward_details(
        completion=completion,
        task_type=task_type,
        source_function_name=source_function_name,
        raw_code=raw_code,
        compile_reference_source=compile_reference_source,
        target_clean_code=target_clean_code,
        target_renamings_json=target_renamings_json,
        allowed_imports_json=allowed_imports_json,
        allowed_callees_json=allowed_callees_json,
        compiler_executable=compiler_executable,
        tests_ref=tests_ref,
        weights=weights,
        behavior_threshold=behavior_threshold,
        execution_pass_rate_threshold=execution_pass_rate_threshold,
        min_completion_ratio=min_completion_ratio,
        max_completion_ratio=max_completion_ratio,
        max_invalid_completion_ratio=max_invalid_completion_ratio,
        max_function_count=max_function_count,
    )["total"]


def compute_completion_reward_details(
    completion: str,
    task_type: str,
    source_function_name: str,
    raw_code: str,
    compile_reference_source: str,
    target_clean_code: str,
    target_renamings_json: str,
    allowed_imports_json: str,
    allowed_callees_json: str,
    compiler_executable: str | None,
    tests_ref: str,
    weights: dict[str, float],
    behavior_threshold: float = _BEHAVIOR_SIMILARITY_THRESHOLD,
    execution_pass_rate_threshold: float = _EXECUTION_PASS_RATE_THRESHOLD,
    min_completion_ratio: float = _MIN_COMPLETION_RATIO,
    max_completion_ratio: float = _MAX_COMPLETION_RATIO,
    max_invalid_completion_ratio: float = _MAX_INVALID_COMPLETION_RATIO,
    max_function_count: int = _MAX_FUNCTION_COUNT,
) -> dict[str, float]:
    try:
        output, schema_status = normalize_output_with_schema_status(completion)
        json_valid = schema_status != "invalid"
        if not json_valid:
            total_penalty, invalid_details = invalid_json_penalty(
                completion,
                raw_code,
                source_function_name,
                max_invalid_completion_ratio=max_invalid_completion_ratio,
                max_function_count=max_function_count,
                weights=weights,
            )
            constraint_penalty_total = (
                weights.get("invalid_length_penalty", 0.0)
                * invalid_details["invalid_length_penalty"]
                + weights.get("truncation_penalty", 0.0)
                * invalid_details["truncation_penalty"]
                + weights.get("invalid_scope_penalty", 0.0)
                * invalid_details["invalid_scope_penalty"]
            )
            constraint_total = -constraint_penalty_total
            core_total = -(total_penalty - constraint_penalty_total)
            return empty_reward_breakdown(
                invalid_length_penalty_value=invalid_details["invalid_length_penalty"],
                truncation_penalty_value=invalid_details["truncation_penalty"],
                invalid_scope_penalty_value=invalid_details["invalid_scope_penalty"],
                raw_completion_ratio=invalid_details["raw_completion_ratio"],
                core_total=core_total,
                constraint_total=constraint_total,
                total=-total_penalty,
            )
        renamings: dict[str, str] = json.loads(target_renamings_json)
        imports: list[str] = json.loads(allowed_imports_json)
        callees: list[str] = json.loads(allowed_callees_json)
        behavior_from_execution = False
        execution_result = evaluate_execution_behavior(
            output.cleaned_c,
            source_function_name=source_function_name,
            compiler_executable=compiler_executable,
            tests_ref=tests_ref,
        )
        if execution_result is not None:
            behavior_from_execution = True
            compile_success = execution_result.compile_success
            behavior_score = execution_result.pass_rate
            behavior_improvement = True
            behavior_success = behavior_score >= execution_pass_rate_threshold
        else:
            compile_success = compile_candidate(
                output.cleaned_c,
                compile_reference_source or target_clean_code,
                function_name=source_function_name or None,
            )
            behavior_score = behavior_similarity(output.cleaned_c, target_clean_code)
            behavior_improvement = is_behavior_improvement(
                output.cleaned_c,
                raw_code,
                target_clean_code,
            )
            behavior_success = behavior_score >= behavior_threshold and behavior_improvement
        return reward_breakdown(
            output=output,
            json_valid=json_valid,
            exact_json=schema_status == "strict",
            raw_code=raw_code,
            target_clean_code=target_clean_code,
            source_function_name=source_function_name,
            target_renamings=renamings,
            compile_success=compile_success,
            behavior_success=behavior_success,
            behavior_score=behavior_score,
            behavior_improvement=behavior_improvement,
            behavior_from_execution=behavior_from_execution,
            allowed_imports=imports,
            allowed_callees=callees,
            weights=weights,
            task_type=task_type,
            min_completion_ratio=min_completion_ratio,
            max_completion_ratio=max_completion_ratio,
            max_function_count=max_function_count,
            compile_reference_source=compile_reference_source,
        )
    except Exception:  # noqa: BLE001
        return empty_reward_breakdown()


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

    trainer_provenance = _trainer_provenance(GRPOTrainer)
    model, tokenizer = load_model_and_tokenizer(config)
    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    if getattr(text_tokenizer, "pad_token", None) is None:
        eos_token = getattr(text_tokenizer, "eos_token", None)
        if eos_token is not None:
            try:
                text_tokenizer.pad_token = eos_token
            except (AttributeError, TypeError):
                logger.debug("tokenizer does not expose a mutable pad_token attribute")
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

    component_weights = config.training.reward_weights
    behavior_threshold = (
        config.training.behavior_similarity_threshold
        if config.training.behavior_similarity_threshold is not None
        else _BEHAVIOR_SIMILARITY_THRESHOLD
    )
    execution_pass_rate_threshold = (
        config.training.execution_pass_rate_threshold
        if config.training.execution_pass_rate_threshold is not None
        else _EXECUTION_PASS_RATE_THRESHOLD
    )
    min_completion_ratio = (
        config.training.min_completion_ratio
        if config.training.min_completion_ratio is not None
        else _MIN_COMPLETION_RATIO
    )
    max_completion_ratio = (
        config.training.max_completion_ratio
        if config.training.max_completion_ratio is not None
        else _MAX_COMPLETION_RATIO
    )
    max_invalid_completion_ratio = (
        config.training.max_invalid_completion_ratio
        if config.training.max_invalid_completion_ratio is not None
        else _MAX_INVALID_COMPLETION_RATIO
    )
    max_function_count = (
        config.training.max_function_count
        if config.training.max_function_count is not None
        else _MAX_FUNCTION_COUNT
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
    rollout_temperature = (
        config.training.rollout_temperature
        if config.training.rollout_temperature is not None
        else 1.0
    )
    rollout_top_p = (
        config.training.rollout_top_p if config.training.rollout_top_p is not None else 1.0
    )
    rollout_top_k = config.training.rollout_top_k
    rollout_min_p = config.training.rollout_min_p
    rollout_repetition_penalty = (
        config.training.rollout_repetition_penalty
        if config.training.rollout_repetition_penalty is not None
        else 1.0
    )
    loss_type = config.training.loss_type or "dr_grpo"
    multi_reward_weights = _resolve_multi_reward_weights(config.training.multi_reward_weights)
    scale_rewards = (
        config.training.scale_rewards if config.training.scale_rewards is not None else "group"
    )
    beta = config.training.beta if config.training.beta is not None else 0.0
    mask_truncated_completions = (
        config.training.mask_truncated_completions
        if config.training.mask_truncated_completions is not None
        else True
    )
    trainer_provenance |= {
        "loss_type": loss_type,
        "scale_rewards": scale_rewards,
        "beta": beta,
        "mask_truncated_completions": mask_truncated_completions,
        "reward_objectives": [
            {"name": objective_name, "field": field_name}
            for objective_name, field_name in _OBJECTIVE_FIELDS
        ],
        "reward_weights": multi_reward_weights,
        "component_reward_weights": component_weights,
    }
    reward_step = 0
    cached_reward_key: tuple[object, ...] | None = None
    cached_reward_details: list[dict[str, float]] | None = None
    cached_reward_uses_remaining = 0

    def _reward_details_batch(
        completions: list[object],
        source_function_name: list[str] | None = None,
        *,
        task_type: list[str] | None = None,
        raw_code: list[str] | None = None,
        compile_reference_source: list[str] | None = None,
        target_clean_code: list[str] | None = None,
        target_renamings: list[str] | None = None,
        allowed_imports: list[str] | None = None,
        allowed_callees: list[str] | None = None,
        compiler_executable: list[str] | None = None,
        tests_ref: list[str] | None = None,
        **_: object,
    ) -> list[dict[str, float]]:
        nonlocal reward_step
        nonlocal cached_reward_key
        nonlocal cached_reward_details
        nonlocal cached_reward_uses_remaining
        n = len(completions)
        completion_texts = [_completion_text(completion) for completion in completions]
        task_types = task_type or ["full_clarify"] * n
        function_names = source_function_name or [""] * n
        raw_codes = raw_code or [""] * n
        compile_sources = compile_reference_source or [""] * n
        target_codes = target_clean_code or [""] * n
        renaming_maps = target_renamings or ["{}"] * n
        import_lists = allowed_imports or ["[]"] * n
        callee_lists = allowed_callees or ["[]"] * n
        compiler_commands = compiler_executable or [None] * n
        test_refs = tests_ref or [""] * n
        reward_key = (
            tuple(completion_texts),
            tuple(function_names),
            tuple(task_types),
            tuple(raw_codes),
            tuple(compile_sources),
            tuple(target_codes),
            tuple(renaming_maps),
            tuple(import_lists),
            tuple(callee_lists),
            tuple(compiler_commands),
            tuple(test_refs),
        )
        if (
            cached_reward_key == reward_key
            and cached_reward_details is not None
            and cached_reward_uses_remaining > 0
        ):
            cached_reward_uses_remaining -= 1
            details = cached_reward_details
            if cached_reward_uses_remaining == 0:
                cached_reward_key = None
                cached_reward_details = None
            return details
        details = [
            compute_completion_reward_details(
                completion=completion_text,
                task_type=task_types[index % len(task_types)],
                source_function_name=function_names[index % len(function_names)],
                raw_code=raw_codes[index % len(raw_codes)],
                compile_reference_source=compile_sources[index % len(compile_sources)],
                target_clean_code=target_codes[index % len(target_codes)],
                target_renamings_json=renaming_maps[index % len(renaming_maps)],
                allowed_imports_json=import_lists[index % len(import_lists)],
                allowed_callees_json=callee_lists[index % len(callee_lists)],
                compiler_executable=compiler_commands[index % len(compiler_commands)],
                tests_ref=test_refs[index % len(test_refs)],
                weights=component_weights,
                behavior_threshold=behavior_threshold,
                execution_pass_rate_threshold=execution_pass_rate_threshold,
                min_completion_ratio=min_completion_ratio,
                max_completion_ratio=max_completion_ratio,
                max_invalid_completion_ratio=max_invalid_completion_ratio,
                max_function_count=max_function_count,
            )
            for index, completion_text in enumerate(completion_texts)
        ]
        reward_step += 1
        reward_metrics = reward_log_row([item["total"] for item in details], step=reward_step)
        component_names = sorted({key for item in details for key in item if key != "total"})
        reward_metrics |= {
            f"components/{name}_mean": _mean([item[name] for item in details])
            for name in component_names
        }
        reward_metrics |= {
            f"objectives/{name}_mean": _mean(
                [float(item.get(field_name, 0.0)) for item in details]
            )
            for name, field_name in _OBJECTIVE_FIELDS
        }
        telemetry.record_metrics(
            {key: value for key, value in reward_metrics.items() if key != "step"},
            step=reward_metrics["step"],
            source="reward_func",
        )
        cached_reward_key = reward_key
        cached_reward_details = details
        cached_reward_uses_remaining = max(len(_OBJECTIVE_FIELDS) - 1, 0)
        return details

    def _make_reward_func(objective_name: str, field_name: str):
        def reward_func(
            completions: list[object],
            source_function_name: list[str] | None = None,
            **kwargs: object,
        ) -> list[float]:
            details = _reward_details_batch(
                completions,
                source_function_name=source_function_name,
                **kwargs,
            )
            return [float(item.get(field_name, 0.0)) for item in details]

        reward_func.__name__ = f"{objective_name}_reward_func"
        return reward_func

    max_steps = config.training.max_steps if config.training.max_steps is not None else -1
    save_steps = config.training.save_steps or (max_steps if max_steps > 0 else 100)
    reward_funcs = [
        _make_reward_func(objective_name, field_name)
        for objective_name, field_name in _OBJECTIVE_FIELDS
    ]
    trainer = GRPOTrainer(
        model=model,
        processing_class=text_tokenizer,
        reward_funcs=reward_funcs,
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
            temperature=rollout_temperature,
            top_p=rollout_top_p,
            top_k=rollout_top_k,
            min_p=rollout_min_p,
            repetition_penalty=rollout_repetition_penalty,
            reward_weights=multi_reward_weights,
            scale_rewards=scale_rewards,
            loss_type=loss_type,
            beta=beta,
            mask_truncated_completions=mask_truncated_completions,
            max_steps=max_steps,
            max_grad_norm=max_grad_norm,
            save_steps=save_steps,
            report_to=["tensorboard"],
        ),
        callbacks=[create_training_telemetry_callback(telemetry)],
    )
    logger.info(
        "configured grpo trainer batch_size=%s grad_accum=%s epochs=%s "
        "generation_batch_size=%s learning_rate=%s scheduler=%s "
        "rollout_temperature=%s rollout_top_p=%s rollout_top_k=%s rollout_min_p=%s "
        "rollout_repetition_penalty=%s "
        "max_prompt_length=%s max_completion_length=%s generations=%s max_steps=%s "
        "loss_type=%s reward_aggregation=weighted_sum reward_weights=%s "
        "scale_rewards=%s beta=%s mask_truncated=%s save_steps=%s",
        batch_size,
        grad_accum_steps,
        epochs,
        generation_batch_size,
        learning_rate,
        lr_scheduler_type,
        rollout_temperature,
        rollout_top_p,
        rollout_top_k,
        rollout_min_p,
        rollout_repetition_penalty,
        config.training.max_prompt_length or 512,
        config.training.max_completion_length or 256,
        num_generations,
        max_steps,
        loss_type,
        multi_reward_weights,
        scale_rewards,
        beta,
        mask_truncated_completions,
        save_steps,
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
                "model": {
                    "base_model_id": config.model.base_model_id,
                    "source_training_profile": config.model.source_training_profile,
                    "loader_variant": config.model.loader_variant,
                },
                "trainer": trainer_provenance,
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
