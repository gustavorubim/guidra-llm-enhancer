from __future__ import annotations

import json
import logging
import types
from pathlib import Path

import pytest
import typer

from decomp_clarifier import cli as cli_module
from decomp_clarifier.schemas.compiler import (
    BinaryArtifact,
    CompileManifest,
)
from decomp_clarifier.schemas.compiler import (
    TestExecutionResult as CompilerTestExecutionResult,
)
from decomp_clarifier.schemas.model_io import ClarifiedFunctionOutput, PredictionRecord
from decomp_clarifier.settings import GenerationConfig, TrainingConfig


def test_bootstrap_writes_app_config(tmp_path: Path, temp_app_config, monkeypatch) -> None:
    logger = logging.getLogger("bootstrap-test")
    logger.handlers.clear()

    monkeypatch.setattr(cli_module.ProjectPaths, "discover", lambda: tmp_path)
    monkeypatch.setattr(cli_module, "load_dotenv", lambda root: None)
    monkeypatch.setattr(cli_module, "load_app_config", lambda root, name="default": temp_app_config)
    monkeypatch.setattr(
        cli_module,
        "configure_logging",
        lambda level, log_file, log_to_console: logger,
    )

    root, paths, run_id, run_dir, configured_logger, app_config = cli_module._bootstrap("unit")

    assert root == tmp_path
    assert paths.root == tmp_path
    assert run_id.startswith("unit-")
    assert run_dir.exists()
    assert configured_logger is logger
    assert app_config == temp_app_config
    assert (run_dir / "app_config.yaml").exists()


def test_cli_loader_helpers_round_trip(
    temp_paths, sample_project, sample_compile_manifest, sample_dataset_samples
) -> None:
    project_dir = temp_paths.generated_projects_dir / sample_project.project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "project_manifest.json").write_text(
        sample_project.model_dump_json(),
        encoding="utf-8",
    )

    binary_dir = temp_paths.binaries_dir / sample_project.project_id
    binary_dir.mkdir(parents=True, exist_ok=True)
    (binary_dir / "compile_manifest.json").write_text(
        sample_compile_manifest.model_dump_json(),
        encoding="utf-8",
    )

    dataset_path = temp_paths.processed_sft_dir / "function_dataset.jsonl"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text(
        "\n".join(sample.model_dump_json() for sample in sample_dataset_samples[:2]) + "\n",
        encoding="utf-8",
    )

    loaded_project = cli_module._load_generated_projects(temp_paths)[0]
    assert loaded_project.project_id == sample_project.project_id
    assert (
        cli_module._load_compile_manifests(temp_paths)[0].project_id
        == sample_compile_manifest.project_id
    )
    assert len(cli_module._load_dataset_samples(dataset_path)) == 2


def test_quarantine_project_moves_artifacts(temp_paths, sample_project) -> None:
    project_dir = temp_paths.generated_projects_dir / sample_project.project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "main.c").write_text("int main(void) { return 0; }\n", encoding="utf-8")

    manifest_path = temp_paths.manifests_dir / f"{sample_project.project_id}.json"
    manifest_path.write_text('{"project_id":"sample_project"}', encoding="utf-8")

    binary_dir = temp_paths.binaries_dir / sample_project.project_id
    binary_dir.mkdir(parents=True, exist_ok=True)
    (binary_dir / "sample_project.exe").write_text("", encoding="utf-8")

    cli_module._quarantine_project(temp_paths, sample_project.project_id)

    assert not project_dir.exists()
    assert not manifest_path.exists()
    assert not binary_dir.exists()
    assert (
        temp_paths.generated_projects_dir / "_quarantine" / sample_project.project_id / "main.c"
    ).exists()
    assert (
        temp_paths.manifests_dir / "quarantine" / f"{sample_project.project_id}.json"
    ).exists()
    assert (temp_paths.binaries_dir / "_quarantine" / sample_project.project_id).exists()


def test_ensure_compiler_available_exits_on_missing_compiler() -> None:
    class BrokenCompiler:
        @property
        def executable(self) -> str:
            raise FileNotFoundError("missing compiler")

    with pytest.raises(typer.Exit):
        cli_module._ensure_compiler_available(BrokenCompiler())  # type: ignore[arg-type]


def test_generate_projects_repairs_before_quarantine(
    monkeypatch,
    repo_root: Path,
    sample_project,
    temp_app_config,
    temp_paths,
    test_logger,
) -> None:
    from typer.testing import CliRunner

    def fake_bootstrap(prefix: str, app_profile: str = "default"):
        run_id = f"{prefix}-test"
        run_dir = temp_paths.run_dir(run_id)
        return repo_root, temp_paths, run_id, run_dir, test_logger, temp_app_config

    monkeypatch.setattr(cli_module, "_bootstrap", fake_bootstrap)
    monkeypatch.setattr(
        cli_module,
        "load_generation_config",
        lambda root, name="default", cli_overrides=None: GenerationConfig.model_validate(
            {
                "model": {
                    "model_id": "model",
                    "repair_model_id": "repair-model",
                    "fallback_models": [],
                    "repair_fallback_models": [],
                    "temperature": 0.2,
                    "repair_temperature": 0.1,
                    "max_tokens": 10,
                },
                "generation": {
                    "project_count": 1,
                    "max_repair_attempts": 1,
                    "difficulty_weights": {"easy": 1.0},
                    "topic_weights": {"string parsing": 1.0},
                },
                "validation": {
                    "min_source_files": 1,
                    "min_function_count": 3,
                    "max_source_files": 8,
                    "banned_includes": [],
                    "banned_calls": [],
                },
            }
        ),
    )
    monkeypatch.setattr(
        cli_module,
        "load_template",
        lambda path: "Repair." if "repair" in path.name else "Generate.",
    )
    monkeypatch.setattr(cli_module, "OpenRouterClient", lambda **kwargs: object())

    class DummyProjectGenerator:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def generate_one(self, index: int):
            project_dir = temp_paths.generated_projects_dir / sample_project.project_id
            project_dir.mkdir(parents=True, exist_ok=True)
            (project_dir / "project_manifest.json").write_text(
                sample_project.model_dump_json(), encoding="utf-8"
            )
            return sample_project

        def repair_project(self, project, compile_manifest, attempt: int = 1):
            return project

    monkeypatch.setattr(cli_module, "ProjectGenerator", DummyProjectGenerator)

    class DummyBuildRunner:
        def __init__(self, _config):
            self.compiler = type("Compiler", (), {"executable": "clang"})()
            self.calls = 0

        def compile_project(self, project, project_root, output_root):
            self.calls += 1
            build_dir = output_root / project.project_id
            build_dir.mkdir(parents=True, exist_ok=True)
            tests = [
                CompilerTestExecutionResult(
                    name="counts_letters",
                    passed=self.calls > 1,
                    returncode=0,
                    stdout="4\n",
                    stderr="",
                )
            ]
            manifest = CompileManifest(
                project_id=project.project_id,
                build_id=f"build-{self.calls}",
                compiler_family="clang",
                compiler_version="clang 20",
                host_os="windows",
                binary_format="pe",
                arch="amd64",
                opt_level="O0",
                source_root=str(project_root / project.project_id),
                output_root=str(build_dir),
                binaries=[
                    BinaryArtifact(
                        path=str(build_dir / f"{project.project_id}.exe"),
                        binary_format="pe",
                        arch="amd64",
                        stripped=False,
                    )
                ],
                test_results=tests,
            )
            (build_dir / "compile_manifest.json").write_text(
                manifest.model_dump_json(), encoding="utf-8"
            )
            return manifest

    monkeypatch.setattr(cli_module, "BuildRunner", DummyBuildRunner)

    runner = CliRunner()
    result = runner.invoke(cli_module.app, ["generate-projects"])
    assert result.exit_code == 0
    metrics = temp_paths.run_dir("generate-test") / "metrics.json"
    payload = metrics.read_text(encoding="utf-8")
    assert '"generated_count": 1' in payload
    assert '"repaired_count": 1' in payload
    assert '"quarantined_count": 0' in payload


def test_base_qwen_logs_every_completion(caplog) -> None:
    samples = [
        types.SimpleNamespace(sample_id="sample-1"),
        types.SimpleNamespace(sample_id="sample-2"),
        types.SimpleNamespace(sample_id="sample-3"),
    ]

    class DummyPredictor:
        def predict(
            self,
            sample,
            *,
            system: str,
            max_new_tokens: int,
            temperature: float,
        ) -> PredictionRecord:
            assert system == "base_qwen"
            assert max_new_tokens == 128
            assert temperature == 0.0
            return PredictionRecord(
                sample_id=sample.sample_id,
                system=system,
                output=ClarifiedFunctionOutput(
                    summary="ok",
                    confidence=1.0,
                    renamings={},
                    cleaned_c="int helper(void) { return 0; }",
                ),
                raw_text=json.dumps(
                    {
                        "summary": "ok",
                        "confidence": 1.0,
                        "renamings": {},
                        "cleaned_c": "int helper(void) { return 0; }",
                    }
                ),
                json_valid=True,
            )

    with caplog.at_level(logging.INFO):
        records = cli_module._run_checkpoint_baseline_system(
            samples,  # type: ignore[arg-type]
            system="base_qwen",
            predictor=DummyPredictor(),
            max_new_tokens=128,
            temperature=0.0,
            logger=logging.getLogger("base-qwen-test"),
            max_workers=1,
        )

    assert len(records) == 3
    progress_messages = [
        record.message
        for record in caplog.records
        if "baseline progress system=base_qwen" in record.message
    ]
    assert len(progress_messages) == 3
    assert "completed=1/3 sample_id=sample-1 json_valid=True" in progress_messages[0]
    assert "completed=2/3 sample_id=sample-2 json_valid=True" in progress_messages[1]
    assert "completed=3/3 sample_id=sample-3 json_valid=True" in progress_messages[2]


def test_resolve_openrouter_model_id_prefers_explicit_override() -> None:
    assert (
        cli_module._resolve_openrouter_model_id(
            base_model_id="Qwen/Qwen3.5-2B",
            base_model_openrouter_id="openrouter/qwen",
        )
        == "openrouter/qwen"
    )
    assert (
        cli_module._resolve_openrouter_model_id(
            base_model_id="Qwen/Qwen3.5-2B",
            base_model_openrouter_id=None,
        )
        == "Qwen/Qwen3.5-2B"
    )
    assert (
        cli_module._resolve_openrouter_model_id(
            base_model_id=None,
            base_model_openrouter_id=None,
        )
        is None
    )


def test_resolve_grpo_base_model_blocks_raw_override_for_sft_profile(temp_paths) -> None:
    config = TrainingConfig.model_validate(
        {
            "model": {
                "base_model_id": "Qwen/Qwen3.5-2B",
                "source_training_profile": "sft_qwen35_2b",
            }
        }
    )

    with pytest.raises(typer.BadParameter, match="completed SFT checkpoint"):
        cli_module._resolve_grpo_base_model(
            temp_paths,
            config,
            training_profile="grpo_qwen35_2b_gdpo_300",
        )

    assert (
        cli_module._resolve_grpo_base_model(
            temp_paths,
            config,
            training_profile="grpo_qwen35_2b_gdpo_300",
            allow_raw_base=True,
        )
        == "Qwen/Qwen3.5-2B"
    )


def test_resolve_grpo_base_model_accepts_sft_checkpoint_override(temp_paths) -> None:
    checkpoint_dir = temp_paths.run_dir("train-sft-20260423-010000") / "model"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "sft_training_manifest.json").write_text("{}", encoding="utf-8")
    config = TrainingConfig.model_validate(
        {
            "model": {
                "base_model_id": str(checkpoint_dir),
                "source_training_profile": "sft_qwen35_2b",
            }
        }
    )

    assert (
        cli_module._resolve_grpo_base_model(
            temp_paths,
            config,
            training_profile="grpo_qwen35_2b_gdpo_300",
        )
        == str(checkpoint_dir)
    )
