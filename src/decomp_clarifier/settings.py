from __future__ import annotations

import os
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, Field


class RunConfig(BaseModel):
    seed: int = 7
    log_level: str = "INFO"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    log_to_console: bool = True
    platform_name: str | None = None


class PathsConfig(BaseModel):
    generated_projects_dir: str = "data/raw/generated_projects"
    manifests_dir: str = "data/raw/manifests"
    binaries_dir: str = "data/raw/binaries"
    ghidra_exports_dir: str = "data/raw/ghidra_exports"
    aligned_projects_dir: str = "data/interim/aligned_projects"
    aligned_functions_dir: str = "data/interim/aligned_functions"
    processed_sft_dir: str = "data/processed/sft"
    processed_rl_dir: str = "data/processed/rl"
    processed_eval_dir: str = "data/processed/eval"
    reports_dir: str = "artifacts/reports"
    runs_dir: str = "artifacts/runs"
    logs_dir: str = "artifacts/logs"


class GhidraConfig(BaseModel):
    project_dir: str = "ghidra/project"
    script_dir: str = "ghidra/scripts"
    install_dir: str | None = None
    analyze_headless_path: str | None = None
    script_name: str = "ExportFunctions.java"
    timeout_seconds: int = 300
    output_debug_text: bool = True


class AppConfig(BaseModel):
    run: RunConfig = Field(default_factory=RunConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    ghidra: GhidraConfig = Field(default_factory=GhidraConfig)


class GenerationModelConfig(BaseModel):
    model_id: str
    fallback_models: list[str] = Field(default_factory=list)
    repair_model_id: str | None = None
    repair_fallback_models: list[str] = Field(default_factory=list)
    temperature: float = 0.3
    repair_temperature: float = 0.1
    max_tokens: int = 4000
    repair_max_tokens: int | None = None


class GenerationOptions(BaseModel):
    project_count: int = 10
    max_repair_attempts: int = 1
    difficulty_weights: dict[str, float] = Field(default_factory=dict)
    topic_weights: dict[str, float] = Field(default_factory=dict)


class GenerationValidationConfig(BaseModel):
    min_source_files: int = 1
    min_function_count: int = 3
    max_source_files: int = 8
    banned_includes: list[str] = Field(default_factory=list)
    banned_calls: list[str] = Field(default_factory=list)


class GenerationConfig(BaseModel):
    model: GenerationModelConfig
    generation: GenerationOptions = Field(default_factory=GenerationOptions)
    validation: GenerationValidationConfig = Field(default_factory=GenerationValidationConfig)


class CompilerProfile(BaseModel):
    family: str = "clang"
    executable: str = "clang"
    c_standard: str = "c11"
    opt_level: str = "O0"
    warnings_as_errors: bool = False
    extra_flags: list[str] = Field(default_factory=list)


class CompileConfig(BaseModel):
    compiler: CompilerProfile = Field(default_factory=CompilerProfile)


class DatasetConfigData(BaseModel):
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 7
    task_mix: dict[str, float] = Field(default_factory=dict)
    prompt_limit: int | None = None
    include_task_types: list[str] = Field(default_factory=list)


class DatasetConfig(BaseModel):
    dataset: DatasetConfigData = Field(default_factory=DatasetConfigData)


class TrainingModelConfig(BaseModel):
    base_model_id: str | None = None
    source_training_profile: str | None = None
    loader_variant: str | None = None


class TrainingRunConfig(BaseModel):
    max_seq_length: int | None = None
    load_in_4bit: bool | None = None
    precision: str | None = None
    lora_rank: int | None = None
    batch_size: int | None = None
    grad_accum_steps: int | None = None
    epochs: int | None = None
    max_steps: int | None = None
    max_prompt_length: int | None = None
    max_completion_length: int | None = None
    generations_per_prompt: int | None = None
    learning_rate: float | None = None
    adam_beta1: float | None = None
    adam_beta2: float | None = None
    weight_decay: float | None = None
    warmup_ratio: float | None = None
    lr_scheduler_type: str | None = None
    optim: str | None = None
    max_grad_norm: float | None = None
    save_steps: int | None = None
    reward_weights: dict[str, float] = Field(default_factory=dict)
    min_train_samples: int | None = None
    behavior_similarity_threshold: float | None = None
    execution_pass_rate_threshold: float | None = None
    min_completion_ratio: float | None = None
    max_completion_ratio: float | None = None
    max_invalid_completion_ratio: float | None = None
    max_function_count: int | None = None


class TrainingHardwareConfig(BaseModel):
    profile: str | None = None
    max_seq_length: int | None = None
    batch_size: int | None = None


class TrainingConfig(BaseModel):
    model: TrainingModelConfig = Field(default_factory=TrainingModelConfig)
    training: TrainingRunConfig = Field(default_factory=TrainingRunConfig)
    hardware: TrainingHardwareConfig = Field(default_factory=TrainingHardwareConfig)


ModelT = TypeVar("ModelT", bound=BaseModel)


def _normalize_env_key(key: str) -> str:
    alias_map = {
        "open_router": "OPENROUTER_API_KEY",
        "OPEN_ROUTER": "OPENROUTER_API_KEY",
        "openrouter_api_key": "OPENROUTER_API_KEY",
    }
    return alias_map.get(key, key)


def _strip_env_value(value: str) -> str:
    trimmed = value.strip()
    if len(trimmed) >= 2 and trimmed[0] == trimmed[-1] and trimmed[0] in {'"', "'"}:
        return trimmed[1:-1]
    return trimmed


def load_dotenv(root: Path) -> None:
    dotenv_path = root / ".env"
    if not dotenv_path.exists():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        normalized_key = _normalize_env_key(key.strip())
        if normalized_key not in os.environ:
            os.environ[normalized_key] = _strip_env_value(value)


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _read_env_overrides() -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    ghidra_dir = os.getenv("DECOMP_CLARIFIER_GHIDRA_DIR")
    analyze_headless = os.getenv("DECOMP_CLARIFIER_GHIDRA_ANALYZE_HEADLESS")
    compiler_executable = os.getenv("DECOMP_CLARIFIER_COMPILER_EXECUTABLE") or os.getenv(
        "DECOMP_CLARIFIER_CLANG"
    )
    if ghidra_dir or analyze_headless:
        overrides["ghidra"] = {}
        if ghidra_dir:
            overrides["ghidra"]["install_dir"] = ghidra_dir
        if analyze_headless:
            overrides["ghidra"]["analyze_headless_path"] = analyze_headless
    if compiler_executable:
        overrides.setdefault("compiler", {})
        overrides["compiler"]["executable"] = compiler_executable
    return overrides


class ConfigLoader:
    def __init__(self, root: Path) -> None:
        self.root = root

    def load(
        self, model_type: type[ModelT], path: Path, cli_overrides: dict[str, Any] | None = None
    ) -> ModelT:
        raw = load_yaml(path)
        merged = deep_merge(raw, _read_env_overrides())
        if cli_overrides:
            merged = deep_merge(merged, cli_overrides)
        return model_type.model_validate(merged)


def load_app_config(
    root: Path, name: str = "default", cli_overrides: dict[str, Any] | None = None
) -> AppConfig:
    default_path = root / "configs" / "app" / "default.yaml"
    data = load_yaml(default_path)
    if name != "default":
        data = deep_merge(data, load_yaml(root / "configs" / "app" / f"{name}.yaml"))
    data = deep_merge(data, _read_env_overrides())
    if cli_overrides:
        data = deep_merge(data, cli_overrides)
    return AppConfig.model_validate(data)


def load_generation_config(
    root: Path, name: str = "default", cli_overrides: dict[str, Any] | None = None
) -> GenerationConfig:
    return ConfigLoader(root).load(
        GenerationConfig,
        root / "configs" / "generation" / f"{name}.yaml",
        cli_overrides=cli_overrides,
    )


def load_compile_config(
    root: Path, name: str = "clang_o0", cli_overrides: dict[str, Any] | None = None
) -> CompileConfig:
    return ConfigLoader(root).load(
        CompileConfig,
        root / "configs" / "compile" / f"{name}.yaml",
        cli_overrides=cli_overrides,
    )


def load_ghidra_config(
    root: Path, name: str = "default", cli_overrides: dict[str, Any] | None = None
) -> GhidraConfig:
    base = load_app_config(root).ghidra.model_dump(mode="python")
    data = deep_merge(
        base, load_yaml(root / "configs" / "ghidra" / f"{name}.yaml").get("ghidra", {})
    )
    data = deep_merge(data, _read_env_overrides().get("ghidra", {}))
    if cli_overrides:
        data = deep_merge(data, cli_overrides)
    return GhidraConfig.model_validate(data)


def load_dataset_config(
    root: Path, name: str = "sft", cli_overrides: dict[str, Any] | None = None
) -> DatasetConfig:
    return ConfigLoader(root).load(
        DatasetConfig,
        root / "configs" / "dataset" / f"{name}.yaml",
        cli_overrides=cli_overrides,
    )


def load_training_config(
    root: Path, name: str, cli_overrides: dict[str, Any] | None = None
) -> TrainingConfig:
    candidate = Path(name)
    if candidate.is_absolute() or candidate.suffix in {".yaml", ".yml"} or any(
        separator in name for separator in ("/", "\\")
    ):
        path = candidate if candidate.is_absolute() else root / candidate
        return ConfigLoader(root).load(
            TrainingConfig,
            path,
            cli_overrides=cli_overrides,
        )
    return ConfigLoader(root).load(
        TrainingConfig,
        root / "configs" / "training" / f"{name}.yaml",
        cli_overrides=cli_overrides,
    )


def dump_yaml(path: Path, model: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(model.model_dump(mode="python"), sort_keys=False), encoding="utf-8"
    )
