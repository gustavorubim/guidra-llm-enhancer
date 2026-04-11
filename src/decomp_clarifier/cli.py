from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import typer
import yaml

from decomp_clarifier.adapters.compiler_clang import ClangCompiler
from decomp_clarifier.adapters.filesystem_cache import FilesystemCache
from decomp_clarifier.adapters.ghidra_headless import GhidraHeadlessAdapter
from decomp_clarifier.adapters.openrouter_client import OpenRouterClient
from decomp_clarifier.baselines import naming_only, raw_ghidra
from decomp_clarifier.baselines.simple_llm_cleanup import PromptOnlyCleanupBaseline
from decomp_clarifier.compilation.build_runner import BuildRunner
from decomp_clarifier.dataset.builders import build_function_dataset
from decomp_clarifier.dataset.packers import pack_rl_records, pack_sft_records, write_jsonl_records
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
        project_root=paths.generated_projects_dir,
        manifest_root=paths.manifests_dir,
    )
    build_runner = BuildRunner(compile_config)
    _ensure_compiler_available(build_runner.compiler)
    target_count = count or generation_config.generation.project_count
    max_attempts = max(target_count * 5, target_count)
    projects: list[GeneratedProject] = []
    quarantined = 0
    for attempt in range(max_attempts):
        if len(projects) >= target_count:
            break
        project = generator.generate_one(index=attempt)
        compile_manifest = build_runner.compile_project(
            project,
            paths.generated_projects_dir,
            paths.binaries_dir,
        )
        tests_ok = not compile_manifest.test_results or all(
            result.passed for result in compile_manifest.test_results
        )
        if compile_manifest.binaries and tests_ok:
            projects.append(project)
            logger.info("validated generated project %s", project.project_id)
            continue
        _quarantine_project(paths, project.project_id)
        quarantined += 1
        logger.warning("quarantined invalid generated project %s", project.project_id)
    if len(projects) < target_count:
        raise RuntimeError(
            f"only generated {len(projects)} valid projects after {max_attempts} attempts"
        )
    metrics = {
        "run_id": run_id,
        "generated_count": len(projects),
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
    _write_resolved(
        run_dir / "resolved_config.yaml",
        {
            "app": app_config.model_dump(mode="python"),
            "dataset": dataset_config.model_dump(mode="python"),
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
    records = pack_sft_records(samples)
    manifest = write_jsonl_records(paths.processed_sft_dir / "sft_records.jsonl", records)
    rl_records = pack_rl_records(samples)
    rl_path = paths.processed_rl_dir / "rl_records.jsonl"
    rl_path.parent.mkdir(parents=True, exist_ok=True)
    rl_path.write_text(
        "\n".join(record.model_dump_json() for record in rl_records) + ("\n" if rl_records else ""),
        encoding="utf-8",
    )
    metrics = {
        "run_id": run_id,
        "samples": len(samples),
        "packed": manifest.record_count,
        "rl_packed": len(rl_records),
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
) -> None:
    root, paths, run_id, run_dir, logger, app_config = _bootstrap(
        "baseline", app_profile=app_profile
    )
    dataset_path = paths.processed_sft_dir / "function_dataset.jsonl"
    samples = _load_dataset_samples(dataset_path)
    baseline = PromptOnlyCleanupBaseline(
        client=OpenRouterClient(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=app_config.run.openrouter_base_url,
            cache=FilesystemCache(paths.root / "data" / "cache" / "openrouter"),
        )
        if os.getenv("OPENROUTER_API_KEY")
        else None,
        prompt_template=load_template(root / "configs" / "prompts" / "function_cleanup.md"),
        model=model_id,
    )
    predictions: list[dict[str, Any]] = []
    for sample in samples:
        predictions.append(
            {
                "sample_id": sample.sample_id,
                "system": "raw_ghidra",
                "output": raw_ghidra.predict(sample).model_dump(mode="python"),
            }
        )
        predictions.append(
            {
                "sample_id": sample.sample_id,
                "system": "naming_only",
                "output": naming_only.predict(sample).model_dump(mode="python"),
            }
        )
        predictions.append(
            {
                "sample_id": sample.sample_id,
                "system": "prompt_only_cleanup",
                "output": baseline.predict(sample).model_dump(mode="python"),
            }
        )
    output_path = run_dir / "baseline_predictions.jsonl"
    output_path.write_text(
        "\n".join(json.dumps(item) for item in predictions) + "\n", encoding="utf-8"
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
    training_profile: str = typer.Option("sft_qwen35_4b"),
    app_profile: str = typer.Option("default"),
) -> None:
    _root, paths, run_id, run_dir, _logger, app_config = _bootstrap(
        "train-sft", app_profile=app_profile
    )
    training_config: TrainingConfig = load_training_config(paths.root, training_profile)
    _write_resolved(
        run_dir / "resolved_config.yaml",
        {
            "app": app_config.model_dump(mode="python"),
            "training": training_config.model_dump(mode="python"),
        },
    )
    manifest = _run_sft_training(
        paths.processed_sft_dir / "sft_records.jsonl", run_dir / "model", training_config
    )
    typer.echo(str(manifest))


@app.command("train-grpo")
def train_grpo(
    training_profile: str = typer.Option("grpo_qwen35_4b"),
    app_profile: str = typer.Option("default"),
) -> None:
    _root, paths, run_id, run_dir, _logger, app_config = _bootstrap(
        "train-grpo", app_profile=app_profile
    )
    training_config: TrainingConfig = load_training_config(paths.root, training_profile)
    _write_resolved(
        run_dir / "resolved_config.yaml",
        {
            "app": app_config.model_dump(mode="python"),
            "training": training_config.model_dump(mode="python"),
        },
    )
    manifest = _run_grpo_training(
        paths.processed_rl_dir / "rl_records.jsonl", run_dir / "model", training_config
    )
    typer.echo(str(manifest))


if __name__ == "__main__":
    app()
