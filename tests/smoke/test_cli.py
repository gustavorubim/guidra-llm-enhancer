from __future__ import annotations

import sys
import types
from pathlib import Path

from typer.testing import CliRunner

from decomp_clarifier.cli import app


def test_cli_help() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "generate-projects" in result.stdout
    assert "doctor" in result.stdout


def test_cli_workflow_smoke(
    monkeypatch,
    tmp_path: Path,
    sample_project,
    sample_compile_manifest,
    sample_parsed_ghidra_project,
    sample_dataset_samples,
    temp_app_config,
    temp_paths,
    test_logger,
) -> None:
    from decomp_clarifier import cli as cli_module
    from decomp_clarifier.schemas.compiler import BinaryArtifact, CompileManifest
    from decomp_clarifier.schemas.dataset import DatasetManifest
    from decomp_clarifier.settings import DatasetConfig, GenerationConfig, TrainingConfig

    root = Path(__file__).resolve().parents[2]

    def fake_bootstrap(prefix: str, app_profile: str = "default"):
        run_id = f"{prefix}-test"
        run_dir = temp_paths.run_dir(run_id)
        return root, temp_paths, run_id, run_dir, test_logger, temp_app_config

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(cli_module, "_bootstrap", fake_bootstrap)
    monkeypatch.setattr(
        cli_module,
        "load_generation_config",
        lambda root, name="default", cli_overrides=None: GenerationConfig.model_validate(
            {
                "model": {
                    "model_id": "model",
                    "fallback_models": [],
                    "temperature": 0.2,
                    "max_tokens": 10,
                },
                "generation": {
                    "project_count": 1,
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
        lambda path: (
            "Topics: {topics}\nWeights: {difficulty_weights}\nValidation: {validation_rules}"
        ),
    )

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

    monkeypatch.setattr(cli_module, "ProjectGenerator", DummyProjectGenerator)
    monkeypatch.setattr(cli_module, "OpenRouterClient", lambda **kwargs: object())

    class DummyBuildRunner:
        def __init__(self, _config):
            self.compiler = type("Compiler", (), {"executable": "clang"})()

        def compile_project(self, project, project_root, output_root):
            manifest = CompileManifest.model_validate(
                sample_compile_manifest.model_dump(mode="python")
                | {
                    "binaries": [
                        BinaryArtifact(
                            path=str(
                                temp_paths.binaries_dir / project.project_id / project.project_id
                            ),
                            binary_format="macho",
                            arch="arm64",
                            stripped=False,
                        )
                    ]
                }
            )
            build_dir = temp_paths.binaries_dir / project.project_id
            build_dir.mkdir(parents=True, exist_ok=True)
            (build_dir / "compile_manifest.json").write_text(
                manifest.model_dump_json(), encoding="utf-8"
            )
            return CompileManifest.model_validate(manifest)

    monkeypatch.setattr(cli_module, "BuildRunner", DummyBuildRunner)

    class DummyGhidraRunner:
        def __init__(self, _adapter):
            pass

        def export_manifest(self, manifest, output_root):
            export_dir = output_root / manifest.project_id
            export_dir.mkdir(parents=True, exist_ok=True)
            (export_dir / "project_manifest.json").write_text(
                sample_parsed_ghidra_project.manifest.model_dump_json(), encoding="utf-8"
            )
            (export_dir / "functions.jsonl").write_text(
                "\n".join(item.model_dump_json() for item in sample_parsed_ghidra_project.functions)
                + "\n",
                encoding="utf-8",
            )
            return export_dir

    monkeypatch.setattr(cli_module, "GhidraExportRunner", DummyGhidraRunner)
    monkeypatch.setattr(cli_module, "_load_generated_projects", lambda paths: [sample_project])
    monkeypatch.setattr(
        cli_module,
        "_load_compile_manifests",
        lambda paths: [
            CompileManifest.model_validate(
                sample_compile_manifest.model_dump(mode="python")
                | {
                    "binaries": [
                        BinaryArtifact(
                            path=str(
                                temp_paths.binaries_dir
                                / sample_project.project_id
                                / sample_project.project_id
                            ),
                            binary_format="macho",
                            arch="arm64",
                            stripped=False,
                        )
                    ]
                }
            )
        ],
    )
    monkeypatch.setattr(
        cli_module,
        "load_dataset_config",
        lambda root, name="sft": DatasetConfig.model_validate(
            {"dataset": {"task_mix": {"full_clarify": 0.5, "cleanup": 0.3, "rename": 0.2}}}
        ),
    )
    monkeypatch.setattr(
        cli_module, "parse_ghidra_export_dir", lambda path: sample_parsed_ghidra_project
    )
    monkeypatch.setattr(
        cli_module,
        "write_jsonl_records",
        lambda path, records: DatasetManifest(
            record_count=len(records),
            split_counts={},
            task_counts={"full_clarify": 3, "cleanup": 3, "rename": 3},
            output_path=str(path),
        ),
    )
    monkeypatch.setattr(
        cli_module,
        "load_training_config",
        lambda root, name: TrainingConfig.model_validate(
            {
                "model": {"base_model_id": "Qwen/Qwen3.5-4B", "loader_variant": "unsloth"},
                "training": {"max_seq_length": 1024},
            }
        ),
    )

    from decomp_clarifier.schemas.model_io import ClarifiedFunctionOutput, PredictionRecord

    class DummyPromptBaseline:
        def __init__(self, client, prompt_template, model):
            self.model = model

        def predict(self, sample):
            return ClarifiedFunctionOutput(
                summary=f"baseline:{self.model}",
                confidence=0.5,
                renamings={},
                cleaned_c=sample.target_clean_code,
            )

    class DummyCheckpointPredictor:
        def __init__(self, checkpoint_dir, config, prompt_formatter=None):
            self.checkpoint_dir = checkpoint_dir

        def predict(self, sample, *, system, max_new_tokens, temperature):
            return PredictionRecord(
                sample_id=sample.sample_id,
                system=system,
                output=ClarifiedFunctionOutput(
                    summary="base model",
                    confidence=0.4,
                    renamings={},
                    cleaned_c=sample.target_clean_code,
                ),
                raw_text='{"summary":"base model"}',
                json_valid=True,
            )

    monkeypatch.setattr(cli_module, "PromptOnlyCleanupBaseline", DummyPromptBaseline)
    monkeypatch.setitem(sys.modules, "unsloth", types.SimpleNamespace())
    import decomp_clarifier.inference.checkpoint_predictor as checkpoint_predictor_module

    monkeypatch.setattr(
        checkpoint_predictor_module,
        "CheckpointPredictor",
        DummyCheckpointPredictor,
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    monkeypatch.setattr(
        cli_module,
        "_run_sft_training",
        lambda dataset_path, output_dir, config: output_dir / "sft_manifest.json",
    )
    monkeypatch.setattr(
        cli_module,
        "_run_grpo_training",
        lambda dataset_path, output_dir, config: output_dir / "grpo_manifest.json",
    )
    monkeypatch.setattr(
        cli_module,
        "_run_checkpoint_evaluation",
        lambda **kwargs: type(
            "Artifacts",
            (),
            {
                "manifest_path": kwargs["run_dir"] / "checkpoint_eval_manifest.json",
            },
        )(),
    )
    monkeypatch.setattr(
        cli_module,
        "build_doctor_report",
        lambda paths, include_training=False: {
            "python": {
                "ok": True,
                "version": "3.13.0",
                "executable": "python.exe",
                "venv_active": True,
            },
            "compiler": {"ok": True, "requested": "clang", "resolved": "clang.exe"},
            "ghidra": {"ok": True, "path": "ghidra/support/analyzeHeadless.bat"},
            "openrouter": {"ok": False, "api_key_present": False},
            **(
                {
                    "training": {
                        "ok": True,
                        "windows_cuda_guard": {"ok": True},
                        "version_lock": {"ok": True},
                        "hardware": {
                            "cuda_available": True,
                            "torch_version": "2.10.0+cu128",
                            "cuda_version": "12.8",
                            "gpu_name": "RTX 4070",
                        },
                        "unsloth_import": {
                            "ok": True,
                            "stdout_tail": "FastLanguageModel",
                            "stderr_tail": None,
                        },
                        "xformers_import": {
                            "ok": True,
                            "stdout_tail": "0.0.35",
                            "stderr_tail": None,
                        },
                        "bitsandbytes_import": {
                            "ok": True,
                            "stdout_tail": "0.49.2",
                            "stderr_tail": None,
                        },
                        "tensorboard_import": {
                            "ok": True,
                            "stdout_tail": "2.20.0",
                            "stderr_tail": None,
                        },
                    }
                }
                if include_training
                else {}
            ),
        },
    )
    monkeypatch.setattr(
        cli_module,
        "render_doctor_report",
        lambda report, include_training=False: "doctor ok",
    )
    monkeypatch.setattr(
        cli_module,
        "doctor_exit_code",
        lambda report, include_training=False: 0,
    )
    monkeypatch.setattr(
        "decomp_clarifier.evaluation.checkpoint_eval.find_latest_completed_checkpoint",
        lambda paths, stage: temp_paths.runs_dir / f"train-{stage}-latest" / "model",
    )

    dataset_file = temp_paths.processed_sft_dir / "function_dataset.jsonl"
    dataset_file.parent.mkdir(parents=True, exist_ok=True)
    dataset_file.write_text(
        "\n".join(sample.model_dump_json() for sample in sample_dataset_samples) + "\n",
        encoding="utf-8",
    )

    runner = CliRunner()
    assert runner.invoke(app, ["doctor"]).exit_code == 0
    assert runner.invoke(app, ["doctor", "--training"]).exit_code == 0
    assert runner.invoke(app, ["generate-projects"]).exit_code == 0
    assert runner.invoke(app, ["compile-projects"]).exit_code == 0
    assert runner.invoke(app, ["export-ghidra"]).exit_code == 0
    assert runner.invoke(app, ["build-dataset"]).exit_code == 0
    assert (
        runner.invoke(
            app,
            [
                "run-baselines",
                "--generation-model-id",
                "openai/gpt-5.4-mini",
                "--strong-model-id",
                "openai/gpt-5.4-xhigh",
                "--base-model-id",
                "Qwen/Qwen3.5-2B",
            ],
        ).exit_code
        == 0
    )
    assert runner.invoke(app, ["eval"]).exit_code == 0
    assert runner.invoke(app, ["report"]).exit_code == 0
    assert runner.invoke(app, ["demo"]).exit_code == 0
    assert runner.invoke(app, ["train-sft"]).exit_code == 0
    assert runner.invoke(app, ["train-grpo"]).exit_code == 0
    assert runner.invoke(app, ["eval-sft-checkpoint"]).exit_code == 0
    assert runner.invoke(app, ["eval-grpo-checkpoint"]).exit_code == 0

    baseline_predictions = sorted(
        temp_paths.runs_dir.glob("baseline-*/baseline_predictions.jsonl")
    )[0]
    assert baseline_predictions.exists()
    baseline_payload = baseline_predictions.read_text(encoding="utf-8")
    assert '"system":"generation_model"' in baseline_payload
    assert '"system":"strong_model"' in baseline_payload
    assert '"system":"base_qwen"' in baseline_payload
    assert '"json_valid":true' in baseline_payload
    assert '"raw_text":"{\\"summary\\":\\"base model\\"}"' in baseline_payload
    assert (temp_paths.processed_rl_dir / "rl_records.jsonl").exists()
    report_file = sorted(temp_paths.reports_dir.glob("*.md"))[0]
    assert report_file.exists()
