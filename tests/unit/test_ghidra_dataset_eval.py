from __future__ import annotations

import json
from pathlib import Path

import pytest

from decomp_clarifier.adapters.compiler_clang import resolve_clang_executable
from decomp_clarifier.adapters.ghidra_headless import GhidraHeadlessAdapter
from decomp_clarifier.baselines import naming_only, raw_ghidra
from decomp_clarifier.baselines.simple_llm_cleanup import heuristic_cleanup
from decomp_clarifier.compilation.binary_inventory import binary_format_for_host
from decomp_clarifier.dataset.packers import pack_rl_records, pack_sft_records, write_jsonl_records
from decomp_clarifier.dataset.splitters import split_project_ids
from decomp_clarifier.evaluation.behavior_eval import behavior_similarity, is_behavior_improvement
from decomp_clarifier.evaluation.compile_eval import compile_candidate
from decomp_clarifier.evaluation.metrics import aggregate_metric, field_complete, placeholder_ratio
from decomp_clarifier.evaluation.naming_eval import normalized_name_similarity
from decomp_clarifier.evaluation.readability_eval import readability_improvement, score_readability
from decomp_clarifier.evaluation.report_builder import build_report, write_report
from decomp_clarifier.ghidra_export.aligner import align_functions
from decomp_clarifier.ghidra_export.export_runner import GhidraExportRunner
from decomp_clarifier.ghidra_export.parse_exports import parse_ghidra_export_dir
from decomp_clarifier.inference.explain import summarize_improvements
from decomp_clarifier.inference.formatter import normalize_output
from decomp_clarifier.inference.runner import InferenceRunner
from decomp_clarifier.logging import configure_logging
from decomp_clarifier.schemas.compiler import BinaryArtifact, CompileManifest
from decomp_clarifier.schemas.evaluation import SampleEvaluation


def test_ghidra_adapter_builds_command(tmp_path: Path, temp_app_config, monkeypatch) -> None:
    monkeypatch.setenv(
        "DECOMP_CLARIFIER_GHIDRA_ANALYZE_HEADLESS", "/opt/ghidra/support/analyzeHeadless"
    )
    adapter = GhidraHeadlessAdapter(temp_app_config.ghidra, root=tmp_path)
    command = adapter.build_command(
        binary_path=tmp_path / "binary",
        output_dir=tmp_path / "exports",
        project_name="proj_sample",
    )
    assert Path(command[0]) == Path("/opt/ghidra/support/analyzeHeadless")
    assert "ExportFunctions.java" in command

    monkeypatch.setattr(
        "decomp_clarifier.adapters.ghidra_headless.run_subprocess",
        lambda *args, **kwargs: type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})(),
    )
    result = adapter.run(
        binary_path=tmp_path / "binary", output_dir=tmp_path / "exports", project_name="proj_sample"
    )
    assert result.returncode == 0


def test_ghidra_adapter_windows_default_path(tmp_path: Path, temp_app_config, monkeypatch) -> None:
    monkeypatch.delenv("DECOMP_CLARIFIER_GHIDRA_ANALYZE_HEADLESS", raising=False)
    monkeypatch.delenv("DECOMP_CLARIFIER_GHIDRA_DIR", raising=False)
    monkeypatch.setattr("decomp_clarifier.adapters.ghidra_headless.os.name", "nt")
    monkeypatch.setattr("decomp_clarifier.adapters.ghidra_headless.Path.home", lambda: tmp_path)

    adapter = GhidraHeadlessAdapter(temp_app_config.ghidra, root=tmp_path)

    assert adapter.analyze_headless_path().name == "analyzeHeadless.bat"


def test_parse_exports_align_dataset_and_export_runner(
    tmp_path: Path,
    sample_project,
    sample_compile_manifest,
    sample_parsed_ghidra_project,
    sample_dataset_samples,
    dataset_config,
) -> None:
    export_dir = tmp_path / "exports" / "sample_project"
    export_dir.mkdir(parents=True)
    (export_dir / "project_manifest.json").write_text(
        sample_parsed_ghidra_project.manifest.model_dump_json(),
        encoding="utf-8",
    )
    (export_dir / "functions.jsonl").write_text(
        "\n".join(row.model_dump_json() for row in sample_parsed_ghidra_project.functions) + "\n",
        encoding="utf-8",
    )
    parsed = parse_ghidra_export_dir(export_dir)
    aligned = align_functions(sample_project, parsed)
    assert len(aligned) == 3
    assert len(sample_dataset_samples) == 9

    records = pack_sft_records(sample_dataset_samples)
    manifest = write_jsonl_records(tmp_path / "packed" / "sft.jsonl", records)
    assert manifest.record_count == 9
    assert set(manifest.task_counts) == {"full_clarify", "cleanup", "rename"}
    rl_records = pack_rl_records(sample_dataset_samples)
    assert len(rl_records) == len(sample_dataset_samples)
    assert rl_records[0].raw_code == sample_dataset_samples[0].ghidra_decompiled_code
    assert "#include <stdio.h>" in rl_records[0].compile_reference_source

    class FakeAdapter:
        def run(self, *, binary_path, output_dir, project_name):
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "project_manifest.json").write_text("{}", encoding="utf-8")
            (output_dir / "functions.jsonl").write_text("", encoding="utf-8")
            return type("Result", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    runner = GhidraExportRunner(FakeAdapter())
    manifest_with_binary = CompileManifest.model_validate(
        sample_compile_manifest.model_dump(mode="python")
        | {
            "binaries": [
                BinaryArtifact(
                    path=str(tmp_path / "fake_binary"),
                    binary_format="macho",
                    arch="arm64",
                    stripped=False,
                )
            ]
        }
    )
    export_path = runner.export_manifest(manifest_with_binary, tmp_path / "ghidra_exports")
    assert (export_path / "source_compile_manifest.json").exists()


def test_baselines_inference_and_evaluation(sample_dataset_samples, tmp_path: Path) -> None:
    sample = sample_dataset_samples[0]
    raw = raw_ghidra.predict(sample)
    renamed = naming_only.predict(sample)
    cleaned = heuristic_cleanup(sample)

    assert raw.cleaned_c == sample.ghidra_decompiled_code
    assert renamed.renamings
    assert cleaned.cleaned_c != ""
    assert field_complete(cleaned)
    assert placeholder_ratio(sample.ghidra_decompiled_code) >= 0.0
    assert placeholder_ratio("") == 0.0
    assert aggregate_metric([], "readability_score") == 0.0
    assert normalized_name_similarity(sample.rename_map_target, sample.rename_map_target) == 1.0
    assert score_readability("") == 0.0
    assert score_readability(cleaned.cleaned_c) >= 0.0
    assert readability_improvement(cleaned.cleaned_c, sample.ghidra_decompiled_code) >= -1.0
    assert behavior_similarity(sample.target_clean_code, sample.target_clean_code) == 1.0
    assert is_behavior_improvement(
        sample.target_clean_code,
        sample.ghidra_decompiled_code,
        sample.target_clean_code,
    )
    assert is_behavior_improvement(
        sample.target_clean_code,
        sample.target_clean_code,
        sample.target_clean_code,
    )
    assert not is_behavior_improvement(
        sample.ghidra_decompiled_code,
        sample.ghidra_decompiled_code,
        sample.target_clean_code,
    )
    compiler_available = resolve_clang_executable("clang") is not None
    assert (
        compile_candidate("int helper(void) { return 1; }", "int helper(void) { return 1; }")
        == compiler_available
    )

    output = normalize_output(
        'prefix {"summary":"ok","confidence":1.0,'
        '"renamings":{},"cleaned_c":"int x(void){return 1;}"} suffix'
    )
    runner = InferenceRunner(lambda _sample: output)
    assert runner.run([sample])[0].summary == "ok"
    assert summarize_improvements(sample, renamed)

    report = build_report(
        "eval-test",
        [
            SampleEvaluation(
                sample_id=sample.sample_id,
                system="raw_ghidra",
                json_valid=True,
                field_complete=True,
                placeholder_ratio=0.1,
                readability_score=0.5,
                naming_score=0.1,
                compile_success=False,
                behavior_success=False,
                notes=[],
            )
        ],
    )
    markdown_path, html_path, json_path = write_report(report, tmp_path / "reports")
    assert markdown_path.exists()
    assert html_path.exists()
    assert json.loads(json_path.read_text(encoding="utf-8"))["run_id"] == "eval-test"


def test_logging_splitters_and_inventory_branches(tmp_path: Path, monkeypatch) -> None:
    logger = configure_logging("INFO", tmp_path / "test.log", log_to_console=False)
    logger.info("hello")
    assert "hello" in (tmp_path / "test.log").read_text(encoding="utf-8")

    split_map = split_project_ids(
        ["a", "b", "c"], seed=7, train_ratio=0.34, val_ratio=0.33, test_ratio=0.33
    )
    assert set(split_map.values()) <= {"train", "val", "test"}
    with pytest.raises(ValueError):
        split_project_ids(["a"], seed=7, train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)

    monkeypatch.setattr(
        "decomp_clarifier.compilation.binary_inventory.platform.system", lambda: "Windows"
    )
    assert binary_format_for_host("windows") == "pe"
