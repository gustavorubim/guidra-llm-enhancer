from __future__ import annotations

import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import typer
import yaml

from decomp_clarifier.adapters.compiler_clang import ClangCompiler, resolve_clang_executable
from decomp_clarifier.adapters.filesystem_cache import FilesystemCache
from decomp_clarifier.adapters.ghidra_headless import GhidraHeadlessAdapter
from decomp_clarifier.adapters.openrouter_client import OpenRouterClient
from decomp_clarifier.baselines import naming_only, raw_ghidra
from decomp_clarifier.baselines.openrouter_structured import OpenRouterStructuredBaselinePredictor
from decomp_clarifier.baselines.simple_llm_cleanup import PromptOnlyCleanupBaseline
from decomp_clarifier.compilation.build_runner import BuildRunner
from decomp_clarifier.dataset.builders import build_function_dataset
from decomp_clarifier.dataset.packers import (
    pack_rl_records,
    pack_sft_records,
    select_training_samples,
    write_jsonl_records,
)
from decomp_clarifier.doctor import build_doctor_report, doctor_exit_code, render_doctor_report
from decomp_clarifier.evaluation.metrics import placeholder_ratio
from decomp_clarifier.evaluation.readability_eval import score_readability
from decomp_clarifier.evaluation.report_builder import build_report, write_report
from decomp_clarifier.generation.project_generator import ProjectGenerator
from decomp_clarifier.generation.prompt_builder import load_template
from decomp_clarifier.ghidra_export.export_runner import GhidraExportRunner
from decomp_clarifier.ghidra_export.parse_exports import (
    ParsedGhidraProject,
    parse_ghidra_export_dir,
)
from decomp_clarifier.inference.explain import summarize_improvements
from decomp_clarifier.logging import configure_logging
from decomp_clarifier.paths import ProjectPaths
from decomp_clarifier.schemas.compiler import CompileManifest
from decomp_clarifier.schemas.dataset import FunctionDatasetSample
from decomp_clarifier.schemas.evaluation import ReportExample, SampleEvaluation
from decomp_clarifier.schemas.generation import GeneratedProject
from decomp_clarifier.schemas.model_io import ClarifiedFunctionOutput, PredictionRecord
from decomp_clarifier.settings import (
    AppConfig,
    CompileConfig,
    DatasetConfig,
    GhidraConfig,
    TrainingConfig,
    dump_yaml,
    load_app_config,
    load_compile_config,
    load_dataset_config,
    load_dotenv,
    load_generation_config,
    load_ghidra_config,
    load_training_config,
)

app = typer.Typer(help="Binary-grounded decompiler clarification pipeline.")


def _run_id(prefix: str) -> str:
    return f"{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def _bootstrap(
    prefix: str, app_profile: str = "default"
) -> tuple[Path, ProjectPaths, str, Path, Any, AppConfig]:
    root = ProjectPaths.discover()
    load_dotenv(root)
    app_config = load_app_config(root, name=app_profile)
    paths = ProjectPaths.from_config(root, app_config)
    paths.ensure()
    run_id = _run_id(prefix)
    run_dir = paths.run_dir(run_id)
    logger = configure_logging(
        app_config.run.log_level, paths.log_file(run_id), app_config.run.log_to_console
    )
    dump_yaml(run_dir / "app_config.yaml", app_config)
    return root, paths, run_id, run_dir, logger, app_config


def _write_resolved(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _ensure_compiler_available(compiler: ClangCompiler) -> None:
    try:
        _ = compiler.executable
    except FileNotFoundError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


def _warn_if_clang_missing(logger: Any, command_name: str) -> None:
    if resolve_clang_executable("clang") is not None:
        return
    logger.warning(
        "WARNING: clang not found on PATH - compile_success_rate may be 0.0 for all "
        "samples in %s. Install clang or add it to PATH to get real compile metrics.",
        command_name,
    )


def _make_openrouter_client(*, api_key: str, base_url: str, cache_root: Path) -> OpenRouterClient:
    return OpenRouterClient(
        api_key=api_key,
        base_url=base_url,
        cache=FilesystemCache(cache_root),
    )


def _baseline_record(
    *,
    sample: FunctionDatasetSample,
    system: str,
    output: ClarifiedFunctionOutput,
    raw_text: str | None = None,
    json_valid: bool = True,
) -> PredictionRecord:
    return PredictionRecord(
        sample_id=sample.sample_id,
        system=system,
        output=output,
        raw_text=raw_text,
        json_valid=json_valid,
    )


def _progress_interval(sample_count: int) -> int:
    if sample_count <= 10:
        return 1
    return max(1, sample_count // 10)


def _log_every_completion(*, system: str, max_workers: int) -> bool:
    return system == "base_qwen" and max_workers == 1


def _resolve_openrouter_model_id(
    *, base_model_id: str | None, base_model_openrouter_id: str | None
) -> str | None:
    if base_model_openrouter_id is not None:
        return base_model_openrouter_id
    return base_model_id


def _run_output_baseline_system(
    samples: list[FunctionDatasetSample],
    *,
    system: str,
    predictor: Any,
    logger: Any | None = None,
    max_workers: int = 1,
) -> list[PredictionRecord]:
    sample_count = len(samples)
    interval = _progress_interval(sample_count)
    log_each_completion = _log_every_completion(system=system, max_workers=max_workers)
    started_at = time.perf_counter()
    if logger is not None:
        logger.info(
            "starting baseline system=%s samples=%s workers=%s",
            system,
            sample_count,
            max_workers,
        )
    records: list[PredictionRecord] = []

    def predict_one(sample: FunctionDatasetSample) -> PredictionRecord:
        return _baseline_record(sample=sample, system=system, output=predictor(sample))

    if max_workers > 1 and sample_count > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for index, record in enumerate(executor.map(predict_one, samples), start=1):
                records.append(record)
                if logger is not None and (
                    log_each_completion
                    or index == 1
                    or index == sample_count
                    or index % interval == 0
                ):
                    elapsed = max(time.perf_counter() - started_at, 1e-9)
                    logger.info(
                        "baseline progress system=%s completed=%s/%s "
                        "sample_id=%s elapsed=%.1fs rate=%.2f samples/s",
                        system,
                        index,
                        sample_count,
                        samples[index - 1].sample_id,
                        elapsed,
                        index / elapsed,
                    )
    else:
        for index, sample in enumerate(samples, start=1):
            records.append(predict_one(sample))
            if logger is not None and (
                log_each_completion
                or index == 1
                or index == sample_count
                or index % interval == 0
            ):
                elapsed = max(time.perf_counter() - started_at, 1e-9)
                logger.info(
                    "baseline progress system=%s completed=%s/%s "
                    "sample_id=%s elapsed=%.1fs rate=%.2f samples/s",
                    system,
                    index,
                    sample_count,
                    sample.sample_id,
                    elapsed,
                    index / elapsed,
                )
    if logger is not None:
        elapsed = max(time.perf_counter() - started_at, 1e-9)
        logger.info(
            "finished baseline system=%s samples=%s elapsed=%.1fs rate=%.2f samples/s",
            system,
            sample_count,
            elapsed,
            sample_count / elapsed,
        )
    return records


def _run_checkpoint_baseline_system(
    samples: list[FunctionDatasetSample],
    *,
    system: str,
    predictor: Any,
    max_new_tokens: int,
    temperature: float,
    logger: Any | None = None,
    max_workers: int = 1,
) -> list[PredictionRecord]:
    sample_count = len(samples)
    interval = _progress_interval(sample_count)
    log_each_completion = _log_every_completion(system=system, max_workers=max_workers)
    started_at = time.perf_counter()
    if logger is not None:
        logger.info(
            "starting baseline system=%s samples=%s workers=%s",
            system,
            sample_count,
            max_workers,
        )
    records: list[PredictionRecord] = []

    def predict_one(sample: FunctionDatasetSample) -> PredictionRecord:
        return predictor.predict(
            sample,
            system=system,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    if max_workers > 1 and sample_count > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for index, record in enumerate(executor.map(predict_one, samples), start=1):
                records.append(record)
                if logger is not None and (
                    log_each_completion
                    or index == 1
                    or index == sample_count
                    or index % interval == 0
                ):
                    elapsed = max(time.perf_counter() - started_at, 1e-9)
                    logger.info(
                        "baseline progress system=%s completed=%s/%s "
                        "sample_id=%s json_valid=%s elapsed=%.1fs rate=%.2f samples/s",
                        system,
                        index,
                        sample_count,
                        samples[index - 1].sample_id,
                        record.json_valid,
                        elapsed,
                        index / elapsed,
                    )
    else:
        for index, sample in enumerate(samples, start=1):
            record = predict_one(sample)
            records.append(record)
            if logger is not None and (
                log_each_completion
                or index == 1
                or index == sample_count
                or index % interval == 0
            ):
                elapsed = max(time.perf_counter() - started_at, 1e-9)
                logger.info(
                    "baseline progress system=%s completed=%s/%s "
                    "sample_id=%s json_valid=%s elapsed=%.1fs rate=%.2f samples/s",
                    system,
                    index,
                    sample_count,
                    sample.sample_id,
                    record.json_valid,
                    elapsed,
                    index / elapsed,
                )
    if logger is not None:
        elapsed = max(time.perf_counter() - started_at, 1e-9)
        logger.info(
            "finished baseline system=%s samples=%s elapsed=%.1fs rate=%.2f samples/s",
            system,
            sample_count,
            elapsed,
            sample_count / elapsed,
        )
    return records


def _ordered_baseline_predictions(
    samples: list[FunctionDatasetSample],
    system_predictions: dict[str, list[PredictionRecord]],
) -> list[PredictionRecord]:
    system_order = [
        "raw_ghidra",
        "naming_only",
        "prompt_only_cleanup",
        "generation_model",
        "strong_model",
        "base_qwen",
        "base_qwen_openrouter",
    ]
    sample_count = len(samples)
    for system, records in system_predictions.items():
        if len(records) != sample_count:
            raise ValueError(
                f"system {system} produced {len(records)} predictions for {sample_count} samples"
            )
    ordered: list[PredictionRecord] = []
    for index in range(sample_count):
        for system in system_order:
            records = system_predictions.get(system)
            if records is not None:
                ordered.append(records[index])
    return ordered


def _compile_manifest_is_valid(compile_manifest: CompileManifest) -> bool:
    tests_ok = not compile_manifest.test_results or all(
        result.passed for result in compile_manifest.test_results
    )
    return bool(compile_manifest.binaries) and tests_ok


def _load_generated_projects(paths: ProjectPaths) -> list[GeneratedProject]:
    manifests = sorted(paths.generated_projects_dir.glob("*/project_manifest.json"))
    return [
        GeneratedProject.model_validate_json(path.read_text(encoding="utf-8")) for path in manifests
    ]


def _load_compile_manifests(paths: ProjectPaths) -> list[CompileManifest]:
    manifests = sorted(paths.binaries_dir.glob("*/compile_manifest.json"))
    return [
        CompileManifest.model_validate_json(path.read_text(encoding="utf-8")) for path in manifests
    ]


def _load_dataset_samples(path: Path) -> list[FunctionDatasetSample]:
    samples: list[FunctionDatasetSample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            samples.append(FunctionDatasetSample.model_validate_json(line))
    return samples


def _quarantine_project(paths: ProjectPaths, project_id: str) -> None:
    source_dir = paths.generated_projects_dir / project_id
    source_quarantine = paths.generated_projects_dir / "_quarantine" / project_id
    manifest_path = paths.manifests_dir / f"{project_id}.json"
    manifest_quarantine = paths.manifests_dir / "quarantine" / f"{project_id}.json"
    binary_dir = paths.binaries_dir / project_id
    binary_quarantine = paths.binaries_dir / "_quarantine" / project_id

    if source_dir.exists():
        source_quarantine.parent.mkdir(parents=True, exist_ok=True)
        if source_quarantine.exists():
            shutil.rmtree(source_quarantine)
        shutil.move(str(source_dir), str(source_quarantine))
    if manifest_path.exists():
        manifest_quarantine.parent.mkdir(parents=True, exist_ok=True)
        manifest_quarantine.write_text(manifest_path.read_text(encoding="utf-8"), encoding="utf-8")
        manifest_path.unlink()
    if binary_dir.exists():
        binary_quarantine.parent.mkdir(parents=True, exist_ok=True)
        if binary_quarantine.exists():
            shutil.rmtree(binary_quarantine)
        shutil.move(str(binary_dir), str(binary_quarantine))


def _run_sft_training(*args: Any, **kwargs: Any) -> Path:
    from decomp_clarifier.training.sft.train import run_sft_training

    return run_sft_training(*args, **kwargs)


def _run_grpo_training(*args: Any, **kwargs: Any) -> Path:
    from decomp_clarifier.training.grpo.train import run_grpo_training

    return run_grpo_training(*args, **kwargs)


def _run_checkpoint_evaluation(*args: Any, **kwargs: Any) -> Any:
    from decomp_clarifier.evaluation.checkpoint_eval import run_checkpoint_evaluation

    return run_checkpoint_evaluation(*args, **kwargs)


def _is_completed_sft_checkpoint(value: str, paths: ProjectPaths) -> bool:
    candidate = Path(value)
    checkpoint_path = candidate if candidate.is_absolute() else paths.root / candidate
    paths_to_check = [checkpoint_path, checkpoint_path / "model", *checkpoint_path.parents[:3]]
    return any((path / "sft_training_manifest.json").exists() for path in paths_to_check)


def _ensure_grpo_base_model_is_sft_checkpoint(
    paths: ProjectPaths,
    training_config: TrainingConfig,
    *,
    training_profile: str,
    allow_raw_base: bool,
) -> None:
    source_profile = training_config.model.source_training_profile
    base_model_id = training_config.model.base_model_id
    if not source_profile or not base_model_id or allow_raw_base:
        return
    if _is_completed_sft_checkpoint(base_model_id, paths):
        return
    raise typer.BadParameter(
        f"training_profile={training_profile!r} is configured with source_training_profile="
        f"{source_profile!r} must start from a completed SFT checkpoint. "
        f"{base_model_id!r} is not a local SFT checkpoint; rerun without "
        "--base-model-id to auto-select SFT, pass an SFT checkpoint path, or use "
        "--allow-raw-base for an intentional raw-model ablation.",
        param_hint="--base-model-id",
    )


def _resolve_grpo_base_model(
    paths: ProjectPaths,
    training_config: TrainingConfig,
    *,
    training_profile: str,
    allow_raw_base: bool = False,
) -> str:
    if training_config.model.base_model_id:
        _ensure_grpo_base_model_is_sft_checkpoint(
            paths,
            training_config,
            training_profile=training_profile,
            allow_raw_base=allow_raw_base,
        )
        return training_config.model.base_model_id
    from decomp_clarifier.evaluation.checkpoint_eval import find_latest_completed_checkpoint

    checkpoint_dir = find_latest_completed_checkpoint(
        paths,
        "sft",
        training_profile=training_config.model.source_training_profile,
    )
    training_config.model.base_model_id = str(checkpoint_dir)
    return training_config.model.base_model_id


@app.command("doctor")
def doctor(
    training: bool = typer.Option(
        False, "--training", help="Include Windows CUDA training dependency checks."
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON output."),
    app_profile: str = typer.Option("default"),
) -> None:
    root = ProjectPaths.discover()
    load_dotenv(root)
    app_config = load_app_config(root, name=app_profile)
    paths = ProjectPaths.from_config(root, app_config)
    report = build_doctor_report(paths, include_training=training)
    if json_output:
        typer.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        typer.echo(render_doctor_report(report, include_training=training))
    raise typer.Exit(code=doctor_exit_code(report, include_training=training))


@app.command("generate-projects")
def generate_projects(
    count: int | None = typer.Option(None, help="Override the configured project count."),
    generation_profile: str = typer.Option("default"),
    compile_profile: str = typer.Option("clang_o0"),
    app_profile: str = typer.Option("default"),
) -> None:
    root, paths, run_id, run_dir, logger, app_config = _bootstrap(
        "generate", app_profile=app_profile
    )
    generation_config = load_generation_config(
        root,
        name=generation_profile,
        cli_overrides={"generation": {"project_count": count}} if count is not None else None,
    )
    compile_config = load_compile_config(root, name=compile_profile)
    _write_resolved(
        run_dir / "resolved_config.yaml",
        {
            "app": app_config.model_dump(mode="python"),
            "generation": generation_config.model_dump(mode="python"),
            "compile": compile_config.model_dump(mode="python"),
        },
    )
    client = OpenRouterClient(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=app_config.run.openrouter_base_url,
        cache=FilesystemCache(paths.root / "data" / "cache" / "openrouter"),
    )
    generator = ProjectGenerator(
        client=client,
        config=generation_config,
        prompt_template=load_template(root / "configs" / "prompts" / "project_generation.md"),
        repair_prompt_template=load_template(root / "configs" / "prompts" / "project_repair.md"),
        project_root=paths.generated_projects_dir,
        manifest_root=paths.manifests_dir,
    )
    build_runner = BuildRunner(compile_config)
    _ensure_compiler_available(build_runner.compiler)
    target_count = count or generation_config.generation.project_count
    max_attempts = max(target_count * 5, target_count)
    projects: list[GeneratedProject] = []
    quarantined = 0
    repaired = 0
    for attempt in range(max_attempts):
        if len(projects) >= target_count:
            break
        project = generator.generate_one(index=attempt)
        compile_manifest = build_runner.compile_project(
            project,
            paths.generated_projects_dir,
            paths.binaries_dir,
        )
        if _compile_manifest_is_valid(compile_manifest):
            projects.append(project)
            logger.info("validated generated project %s", project.project_id)
            continue
        repaired_project = project
        repair_succeeded = False
        for repair_attempt in range(generation_config.generation.max_repair_attempts):
            logger.info(
                "repairing generated project %s after failed validation attempt=%s",
                repaired_project.project_id,
                repair_attempt + 1,
            )
            try:
                repaired_project = generator.repair_project(
                    repaired_project, compile_manifest, attempt=repair_attempt + 1
                )
            except Exception as exc:
                logger.warning(
                    "repair failed for generated project %s: %s",
                    repaired_project.project_id,
                    exc,
                )
                break
            compile_manifest = build_runner.compile_project(
                repaired_project,
                paths.generated_projects_dir,
                paths.binaries_dir,
            )
            if _compile_manifest_is_valid(compile_manifest):
                projects.append(repaired_project)
                repaired += 1
                repair_succeeded = True
                logger.info("validated repaired generated project %s", repaired_project.project_id)
                break
        if repair_succeeded:
            continue
        _quarantine_project(paths, repaired_project.project_id)
        quarantined += 1
        logger.warning("quarantined invalid generated project %s", repaired_project.project_id)
    if len(projects) < target_count:
        raise RuntimeError(
            f"only generated {len(projects)} valid projects after {max_attempts} attempts"
        )
    metrics = {
        "run_id": run_id,
        "generated_count": len(projects),
        "repaired_count": repaired,
        "quarantined_count": quarantined,
        "attempt_count": len(projects) + quarantined,
        "project_ids": [project.project_id for project in projects],
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    logger.info("generated %s projects", len(projects))
    typer.echo(str(run_dir / "metrics.json"))


@app.command("compile-projects")
def compile_projects(
    compile_profile: str = typer.Option("clang_o0"),
    app_profile: str = typer.Option("default"),
) -> None:
    root, paths, run_id, run_dir, logger, app_config = _bootstrap(
        "compile", app_profile=app_profile
    )
    compile_config: CompileConfig = load_compile_config(root, name=compile_profile)
    _write_resolved(
        run_dir / "resolved_config.yaml",
        {
            "app": app_config.model_dump(mode="python"),
            "compile": compile_config.model_dump(mode="python"),
        },
    )
    projects = _load_generated_projects(paths)
    runner = BuildRunner(compile_config)
    _ensure_compiler_available(runner.compiler)
    manifests = [
        runner.compile_project(project, paths.generated_projects_dir, paths.binaries_dir)
        for project in projects
    ]
    metrics = {
        "run_id": run_id,
        "projects": len(projects),
        "compiled": sum(1 for manifest in manifests if manifest.binaries),
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    logger.info("compiled %s/%s projects", metrics["compiled"], metrics["projects"])
    typer.echo(str(run_dir / "metrics.json"))


@app.command("export-ghidra")
def export_ghidra(
    ghidra_profile: str = typer.Option("default"),
    app_profile: str = typer.Option("default"),
) -> None:
    root, paths, run_id, run_dir, logger, app_config = _bootstrap("ghidra", app_profile=app_profile)
    ghidra_config: GhidraConfig = load_ghidra_config(root, name=ghidra_profile)
    _write_resolved(
        run_dir / "resolved_config.yaml",
        {
            "app": app_config.model_dump(mode="python"),
            "ghidra": ghidra_config.model_dump(mode="python"),
        },
    )
    adapter = GhidraHeadlessAdapter(ghidra_config, root=root)
    runner = GhidraExportRunner(adapter)
    compile_manifests = _load_compile_manifests(paths)
    exported_dirs = [
        runner.export_manifest(manifest, paths.ghidra_exports_dir)
        for manifest in compile_manifests
        if manifest.binaries
    ]
    metrics = {"run_id": run_id, "exported": len(exported_dirs)}
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    logger.info("exported %s ghidra projects", len(exported_dirs))
    typer.echo(str(run_dir / "metrics.json"))


@app.command("build-dataset")
def build_dataset(
    dataset_profile: str = typer.Option("sft"),
    app_profile: str = typer.Option("default"),
) -> None:
    root, paths, run_id, run_dir, logger, app_config = _bootstrap(
        "dataset", app_profile=app_profile
    )
    dataset_config: DatasetConfig = load_dataset_config(root, name=dataset_profile)
    rl_dataset_config: DatasetConfig = load_dataset_config(root, name="rl")
    _write_resolved(
        run_dir / "resolved_config.yaml",
        {
            "app": app_config.model_dump(mode="python"),
            "dataset": dataset_config.model_dump(mode="python"),
            "rl_dataset": rl_dataset_config.model_dump(mode="python"),
        },
    )
    projects = _load_generated_projects(paths)
    manifests = _load_compile_manifests(paths)
    exports: list[ParsedGhidraProject] = [
        parse_ghidra_export_dir(path)
        for path in sorted(paths.ghidra_exports_dir.iterdir())
        if path.is_dir()
        and (path / "project_manifest.json").exists()
        and (path / "functions.jsonl").exists()
    ]
    samples = build_function_dataset(
        projects=projects,
        compile_manifests=manifests,
        parsed_exports=exports,
        config=dataset_config,
        output_dir=paths.processed_sft_dir,
    )
    sft_samples = select_training_samples(samples, split="train")
    records = pack_sft_records(sft_samples)
    manifest = write_jsonl_records(
        paths.processed_sft_dir / "sft_records.jsonl",
        records,
        split_counts={"train": len(sft_samples)},
    )
    rl_samples = select_training_samples(
        samples,
        split="train",
        include_task_types=rl_dataset_config.dataset.include_task_types,
        prompt_limit=rl_dataset_config.dataset.prompt_limit,
    )
    rl_records = pack_rl_records(rl_samples)
    rl_path = paths.processed_rl_dir / "rl_records.jsonl"
    rl_path.parent.mkdir(parents=True, exist_ok=True)
    rl_path.write_text(
        "\n".join(record.model_dump_json() for record in rl_records) + ("\n" if rl_records else ""),
        encoding="utf-8",
    )
    split_counts = {
        split: sum(1 for sample in samples if sample.split == split)
        for split in {"train", "val", "test"}
    }
    rl_task_counts: dict[str, int] = {}
    for record in rl_records:
        rl_task_counts[record.task_type] = rl_task_counts.get(record.task_type, 0) + 1
    metrics = {
        "run_id": run_id,
        "samples": len(samples),
        "split_counts": split_counts,
        "packed": manifest.record_count,
        "sft_packed": manifest.record_count,
        "rl_packed": len(rl_records),
        "rl_task_counts": rl_task_counts,
        "rl_include_task_types": rl_dataset_config.dataset.include_task_types,
        "rl_prompt_limit": rl_dataset_config.dataset.prompt_limit,
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    logger.info("built %s dataset samples", len(samples))
    typer.echo(str(paths.processed_sft_dir / "sft_records.jsonl"))


@app.command("run-baselines")
def run_baselines(
    app_profile: str = typer.Option("default"),
    model_id: str = typer.Option("openai/gpt-4.1-mini"),
    generation_model_id: str = typer.Option("openai/gpt-5.4-mini"),
    strong_model_id: str = typer.Option("openai/gpt-5.4-xhigh"),
    base_model_id: str | None = typer.Option(
        None,
        help="OpenRouter model id for the remote base_qwen comparison baseline.",
    ),
    base_model_openrouter_id: str | None = typer.Option(
        None,
        help=(
            "Explicit override for the OpenRouter base_qwen model id. "
            "Defaults to --base-model-id."
        ),
    ),
    base_model_local_id: str | None = typer.Option(
        None,
        help=(
            "Optional local Windows CUDA model/checkpoint for the local base_qwen baseline. "
            "If omitted, the local base_qwen run is skipped."
        ),
    ),
    sample_limit: int | None = typer.Option(None),
    remote_workers: int = typer.Option(2, min=1),
) -> None:
    root, paths, run_id, run_dir, logger, app_config = _bootstrap(
        "baseline", app_profile=app_profile
    )
    _warn_if_clang_missing(logger, "run-baselines")
    dataset_path = paths.processed_sft_dir / "function_dataset.jsonl"
    samples = _load_dataset_samples(dataset_path)
    if sample_limit is not None:
        samples = samples[:sample_limit]
    logger.info(
        "loaded baseline dataset samples=%s path=%s sample_limit=%s remote_workers=%s",
        len(samples),
        dataset_path,
        sample_limit,
        remote_workers,
    )
    api_key = os.getenv("OPENROUTER_API_KEY")
    cache_root = paths.root / "data" / "cache" / "openrouter"
    baseline = PromptOnlyCleanupBaseline(
        client=(
            _make_openrouter_client(
                api_key=api_key,
                base_url=app_config.run.openrouter_base_url,
                cache_root=cache_root,
            )
            if api_key
            else None
        ),
        prompt_template=load_template(root / "configs" / "prompts" / "function_cleanup.md"),
        model=model_id,
    )
    generation_baseline = (
        PromptOnlyCleanupBaseline(
            client=_make_openrouter_client(
                api_key=api_key,
                base_url=app_config.run.openrouter_base_url,
                cache_root=cache_root,
            ),
            prompt_template=load_template(root / "configs" / "prompts" / "function_cleanup.md"),
            model=generation_model_id,
        )
        if api_key
        else None
    )
    strong_baseline = (
        PromptOnlyCleanupBaseline(
            client=_make_openrouter_client(
                api_key=api_key,
                base_url=app_config.run.openrouter_base_url,
                cache_root=cache_root,
            ),
            prompt_template=load_template(root / "configs" / "prompts" / "function_cleanup.md"),
            model=strong_model_id,
        )
        if api_key
        else None
    )
    if api_key is None:
        logger.warning(
            "OPENROUTER_API_KEY not set; skipping generation_model, strong_model, "
            "and base_qwen_openrouter baselines."
        )

    base_qwen_predictor = None
    if base_model_local_id:
        try:
            import unsloth  # noqa: F401  # type: ignore[import-not-found]

            from decomp_clarifier.dataset.prompt_formatter import format_rl_prompt
            from decomp_clarifier.inference.checkpoint_predictor import CheckpointPredictor

            base_config = load_training_config(root, "grpo_qwen35_2b")
            base_config.model.base_model_id = base_model_local_id
            base_qwen_predictor = CheckpointPredictor(
                base_model_local_id,
                base_config,
                prompt_formatter=format_rl_prompt,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("skipping base_qwen baseline: %s", exc)

    base_qwen_openrouter_predictor = None
    remote_base_model_id = _resolve_openrouter_model_id(
        base_model_id=base_model_id,
        base_model_openrouter_id=base_model_openrouter_id,
    )
    if api_key is not None and remote_base_model_id is not None:
        from decomp_clarifier.dataset.prompt_formatter import format_rl_prompt

        base_qwen_openrouter_predictor = OpenRouterStructuredBaselinePredictor(
            client=_make_openrouter_client(
                api_key=api_key,
                base_url=app_config.run.openrouter_base_url,
                cache_root=cache_root,
            ),
            model=remote_base_model_id,
            prompt_formatter=format_rl_prompt,
            schema_version="base-qwen-openrouter-baseline",
        )

    system_predictions: dict[str, list[PredictionRecord]] = {
        "raw_ghidra": _run_output_baseline_system(
            samples,
            system="raw_ghidra",
            predictor=raw_ghidra.predict,
            logger=logger,
        ),
        "naming_only": _run_output_baseline_system(
            samples,
            system="naming_only",
            predictor=naming_only.predict,
            logger=logger,
        ),
    }

    pending_jobs: list[tuple[str, Any, dict[str, Any]]] = []
    if api_key is None:
        system_predictions["prompt_only_cleanup"] = _run_output_baseline_system(
            samples,
            system="prompt_only_cleanup",
            predictor=baseline.predict,
            logger=logger,
        )
    else:
        pending_jobs.append(
            (
                "prompt_only_cleanup",
                _run_output_baseline_system,
                {
                    "samples": samples,
                    "system": "prompt_only_cleanup",
                    "predictor": baseline.predict,
                    "logger": logger,
                    "max_workers": remote_workers,
                },
            )
        )
        if generation_baseline is not None:
            pending_jobs.append(
                (
                    "generation_model",
                    _run_output_baseline_system,
                    {
                        "samples": samples,
                        "system": "generation_model",
                        "predictor": generation_baseline.predict,
                        "logger": logger,
                        "max_workers": remote_workers,
                    },
                )
            )
        if strong_baseline is not None:
            pending_jobs.append(
                (
                    "strong_model",
                    _run_output_baseline_system,
                    {
                        "samples": samples,
                        "system": "strong_model",
                        "predictor": strong_baseline.predict,
                        "logger": logger,
                        "max_workers": remote_workers,
                    },
                )
            )
    if base_qwen_predictor is not None:
        pending_jobs.append(
            (
                "base_qwen",
                _run_checkpoint_baseline_system,
                {
                    "samples": samples,
                    "system": "base_qwen",
                    "predictor": base_qwen_predictor,
                    "max_new_tokens": 384,
                    "temperature": 0.0,
                    "logger": logger,
                    "max_workers": 1,
                },
            )
        )
    if base_qwen_openrouter_predictor is not None:
        pending_jobs.append(
            (
                "base_qwen_openrouter",
                _run_checkpoint_baseline_system,
                {
                    "samples": samples,
                    "system": "base_qwen_openrouter",
                    "predictor": base_qwen_openrouter_predictor,
                    "max_new_tokens": 384,
                    "temperature": 0.0,
                    "logger": logger,
                    "max_workers": remote_workers,
                },
            )
        )

    if pending_jobs:
        logger.info(
            "running %s expensive baseline systems in parallel: %s",
            len(pending_jobs),
            ", ".join(system for system, _fn, _kwargs in pending_jobs),
        )
        with ThreadPoolExecutor(max_workers=len(pending_jobs)) as executor:
            futures = {
                system: executor.submit(fn, **kwargs) for system, fn, kwargs in pending_jobs
            }
            for system, future in futures.items():
                system_predictions[system] = future.result()

    predictions = _ordered_baseline_predictions(samples, system_predictions)
    output_path = run_dir / "baseline_predictions.jsonl"
    output_path.write_text(
        "\n".join(record.model_dump_json() for record in predictions) + "\n",
        encoding="utf-8",
    )
    logger.info("wrote %s baseline predictions", len(predictions))
    typer.echo(str(output_path))


@app.command("eval")
def evaluate(
    app_profile: str = typer.Option("default"),
) -> None:
    _root, paths, run_id, run_dir, logger, _app_config = _bootstrap("eval", app_profile=app_profile)
    dataset_path = paths.processed_sft_dir / "function_dataset.jsonl"
    samples = {sample.sample_id: sample for sample in _load_dataset_samples(dataset_path)}
    baseline_path = sorted(paths.runs_dir.glob("baseline-*/baseline_predictions.jsonl"))[-1]
    evaluations: list[SampleEvaluation] = []
    examples: list[ReportExample] = []
    for line in baseline_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        sample = samples[payload["sample_id"]]
        cleaned_c = payload["output"]["cleaned_c"]
        evaluation = SampleEvaluation(
            sample_id=sample.sample_id,
            system=payload["system"],
            json_valid=True,
            field_complete=bool(payload["output"]["summary"] and cleaned_c),
            placeholder_ratio=placeholder_ratio(cleaned_c),
            readability_score=score_readability(cleaned_c),
            naming_score=1.0 if payload["output"]["renamings"] == sample.rename_map_target else 0.0,
            compile_success=False,
            behavior_success=cleaned_c.strip() == sample.target_clean_code.strip(),
            notes=[],
        )
        evaluations.append(evaluation)
        if len(examples) < 5:
            examples.append(
                ReportExample(
                    sample_id=sample.sample_id,
                    project_id=sample.project_id,
                    source_function_name=sample.source_function_name,
                    raw_ghidra=sample.ghidra_decompiled_code,
                    candidate=cleaned_c,
                    original_source=sample.target_clean_code,
                    note="; ".join(summarize_improvements(sample, raw_ghidra.predict(sample))),
                )
            )
    report = build_report(run_id, evaluations)
    report.examples = examples
    write_report(report, paths.reports_dir)
    (run_dir / "metrics.json").write_text(
        json.dumps(report.metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    logger.info("evaluated %s predictions", len(evaluations))
    typer.echo(str(paths.reports_dir / f"{run_id}.md"))


@app.command("report")
def report(
    app_profile: str = typer.Option("default"),
) -> None:
    _root, paths, _run_id, _run_dir, _logger, _app_config = _bootstrap(
        "report", app_profile=app_profile
    )
    latest = sorted(paths.reports_dir.glob("*.md"))[-1]
    typer.echo(str(latest))


@app.command("demo")
def demo(
    app_profile: str = typer.Option("default"),
) -> None:
    _root, paths, _run_id, _run_dir, _logger, _app_config = _bootstrap(
        "demo", app_profile=app_profile
    )
    samples = _load_dataset_samples(paths.processed_sft_dir / "function_dataset.jsonl")
    sample = samples[0]
    output = naming_only.predict(sample)
    typer.echo(
        json.dumps(
            {"sample_id": sample.sample_id, "output": output.model_dump(mode="python")}, indent=2
        )
    )


@app.command("train-sft")
def train_sft(
    training_profile: str = typer.Option("sft_qwen35_2b"),
    app_profile: str = typer.Option("default"),
) -> None:
    _root, paths, run_id, run_dir, logger, app_config = _bootstrap(
        "train-sft", app_profile=app_profile
    )
    training_config: TrainingConfig = load_training_config(paths.root, training_profile)
    _write_resolved(
        run_dir / "resolved_config.yaml",
        {
            "app": app_config.model_dump(mode="python"),
            "training_profile": training_profile,
            "training": training_config.model_dump(mode="python"),
        },
    )
    logger.info(
        "starting train-sft run_id=%s profile=%s dataset=%s output_dir=%s",
        run_id,
        training_profile,
        paths.processed_sft_dir / "sft_records.jsonl",
        run_dir / "model",
    )
    manifest = _run_sft_training(
        paths.processed_sft_dir / "sft_records.jsonl", run_dir / "model", training_config
    )
    logger.info("completed train-sft manifest=%s", manifest)
    typer.echo(str(manifest))


@app.command("train-grpo")
def train_grpo(
    training_profile: str = typer.Option("grpo_qwen35_2b"),
    base_model_id: str | None = typer.Option(
        None,
        help=(
            "Optional local checkpoint/model path or Hugging Face model id to use as the "
            "GRPO starting point. Overrides the profile's base_model_id/source_training_profile."
        ),
    ),
    dataset_path: Path | None = typer.Option(  # noqa: B008
        None,
        help=(
            "Optional packed RL JSONL dataset path. Defaults to "
            "data/processed/rl/rl_records.jsonl."
        ),
    ),
    allow_raw_base: bool = typer.Option(
        False,
        "--allow-raw-base",
        help=(
            "Permit overriding an SFT-sourced GRPO/GDPO profile with a raw Hugging Face "
            "model id. Intended only for ablations."
        ),
    ),
    app_profile: str = typer.Option("default"),
) -> None:
    _root, paths, run_id, run_dir, logger, app_config = _bootstrap(
        "train-grpo", app_profile=app_profile
    )
    training_config: TrainingConfig = load_training_config(paths.root, training_profile)
    if base_model_id:
        training_config.model.base_model_id = base_model_id
        _ensure_grpo_base_model_is_sft_checkpoint(
            paths,
            training_config,
            training_profile=training_profile,
            allow_raw_base=allow_raw_base,
        )
        if not _is_completed_sft_checkpoint(base_model_id, paths):
            training_config.model.source_training_profile = None
    resolved_base_model = _resolve_grpo_base_model(
        paths,
        training_config,
        training_profile=training_profile,
        allow_raw_base=allow_raw_base,
    )
    resolved_dataset_path = (
        paths.resolve(dataset_path)
        if dataset_path is not None
        else paths.processed_rl_dir / "rl_records.jsonl"
    )
    _write_resolved(
        run_dir / "resolved_config.yaml",
        {
            "app": app_config.model_dump(mode="python"),
            "training_profile": training_profile,
            "training": training_config.model_dump(mode="python"),
            "dataset_path": str(resolved_dataset_path),
        },
    )
    logger.info(
        "starting train-grpo run_id=%s profile=%s dataset=%s output_dir=%s base_model=%s",
        run_id,
        training_profile,
        resolved_dataset_path,
        run_dir / "model",
        resolved_base_model,
    )
    manifest = _run_grpo_training(
        resolved_dataset_path, run_dir / "model", training_config
    )
    logger.info("completed train-grpo manifest=%s", manifest)
    typer.echo(str(manifest))


@app.command("eval-sft-checkpoint")
def eval_sft_checkpoint(
    checkpoint_dir: Path | None = typer.Option(None),  # noqa: B008
    training_profile: str = typer.Option("sft_qwen35_2b"),
    split: str = typer.Option("val"),
    sample_limit: int | None = typer.Option(None),
    inspection_sample_count: int = typer.Option(8),
    max_new_tokens: int = typer.Option(384),
    temperature: float = typer.Option(0.0),
    prompt_profile: str = typer.Option(
        "stage",
        help=(
            "Prompt formatter for checkpoint eval: stage, compact, full, "
            "context_plus, or context_plus_strict."
        ),
    ),
    thinking: bool = typer.Option(
        False,
        "--thinking/--no-thinking",
        help="Enable the checkpoint chat template thinking path during generation.",
    ),
    app_profile: str = typer.Option("default"),
) -> None:
    from decomp_clarifier.evaluation.checkpoint_eval import find_latest_completed_checkpoint

    root, paths, run_id, run_dir, logger, app_config = _bootstrap(
        "eval-sft-checkpoint", app_profile=app_profile
    )
    _warn_if_clang_missing(logger, "eval-sft-checkpoint")
    resolved_checkpoint = (
        paths.resolve(checkpoint_dir)
        if checkpoint_dir is not None
        else find_latest_completed_checkpoint(paths, "sft", training_profile=training_profile)
    )
    logger.info(
        "starting eval-sft-checkpoint run_id=%s checkpoint=%s split=%s",
        run_id,
        resolved_checkpoint,
        split,
    )
    artifacts = _run_checkpoint_evaluation(
        root=root,
        paths=paths,
        run_id=run_id,
        run_dir=run_dir,
        logger=logger,
        stage="sft",
        checkpoint_dir=resolved_checkpoint,
        training_profile=training_profile,
        split=split,
        sample_limit=sample_limit,
        inspection_sample_count=inspection_sample_count,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        prompt_profile=prompt_profile,
        enable_thinking=thinking,
    )
    _write_resolved(
        run_dir / "resolved_config.yaml",
        {
            "app": app_config.model_dump(mode="python"),
            "stage": "sft",
            "checkpoint_dir": str(resolved_checkpoint),
            "training_profile": training_profile,
            "split": split,
            "sample_limit": sample_limit,
            "inspection_sample_count": inspection_sample_count,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "prompt_profile": prompt_profile,
            "enable_thinking": thinking,
        },
    )
    logger.info("completed eval-sft-checkpoint manifest=%s", artifacts.manifest_path)
    typer.echo(str(artifacts.manifest_path))


@app.command("eval-grpo-checkpoint")
def eval_grpo_checkpoint(
    checkpoint_dir: Path | None = typer.Option(None),  # noqa: B008
    training_profile: str = typer.Option("grpo_qwen35_2b"),
    split: str = typer.Option("val"),
    sample_limit: int | None = typer.Option(None),
    inspection_sample_count: int = typer.Option(8),
    max_new_tokens: int = typer.Option(384),
    temperature: float = typer.Option(0.0),
    prompt_profile: str = typer.Option(
        "stage",
        help=(
            "Prompt formatter for checkpoint eval: stage, compact, full, "
            "context_plus, or context_plus_strict."
        ),
    ),
    thinking: bool = typer.Option(
        False,
        "--thinking/--no-thinking",
        help="Enable the checkpoint chat template thinking path during generation.",
    ),
    app_profile: str = typer.Option("default"),
) -> None:
    from decomp_clarifier.evaluation.checkpoint_eval import find_latest_completed_checkpoint

    root, paths, run_id, run_dir, logger, app_config = _bootstrap(
        "eval-grpo-checkpoint", app_profile=app_profile
    )
    _warn_if_clang_missing(logger, "eval-grpo-checkpoint")
    resolved_checkpoint = (
        paths.resolve(checkpoint_dir)
        if checkpoint_dir is not None
        else find_latest_completed_checkpoint(paths, "grpo", training_profile=training_profile)
    )
    logger.info(
        "starting eval-grpo-checkpoint run_id=%s checkpoint=%s split=%s",
        run_id,
        resolved_checkpoint,
        split,
    )
    artifacts = _run_checkpoint_evaluation(
        root=root,
        paths=paths,
        run_id=run_id,
        run_dir=run_dir,
        logger=logger,
        stage="grpo",
        checkpoint_dir=resolved_checkpoint,
        training_profile=training_profile,
        split=split,
        sample_limit=sample_limit,
        inspection_sample_count=inspection_sample_count,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        prompt_profile=prompt_profile,
        enable_thinking=thinking,
    )
    _write_resolved(
        run_dir / "resolved_config.yaml",
        {
            "app": app_config.model_dump(mode="python"),
            "stage": "grpo",
            "checkpoint_dir": str(resolved_checkpoint),
            "training_profile": training_profile,
            "split": split,
            "sample_limit": sample_limit,
            "inspection_sample_count": inspection_sample_count,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "prompt_profile": prompt_profile,
            "enable_thinking": thinking,
        },
    )
    logger.info("completed eval-grpo-checkpoint manifest=%s", artifacts.manifest_path)
    typer.echo(str(artifacts.manifest_path))


if __name__ == "__main__":
    app()
