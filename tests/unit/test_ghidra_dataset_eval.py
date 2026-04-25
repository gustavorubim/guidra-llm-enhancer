from __future__ import annotations

import json
from pathlib import Path

import pytest

from decomp_clarifier.adapters.compiler_clang import resolve_clang_executable
from decomp_clarifier.adapters.ghidra_headless import GhidraHeadlessAdapter
from decomp_clarifier.baselines import naming_only, raw_ghidra
from decomp_clarifier.baselines.simple_llm_cleanup import heuristic_cleanup
from decomp_clarifier.compilation.binary_inventory import binary_format_for_host
from decomp_clarifier.dataset.packers import (
    pack_rl_records,
    pack_sft_records,
    select_training_samples,
    write_jsonl_records,
)
from decomp_clarifier.dataset.prompt_formatter import format_prompt, format_rl_prompt
from decomp_clarifier.dataset.splitters import split_project_ids
from decomp_clarifier.evaluation.behavior_eval import behavior_similarity, is_behavior_improvement
from decomp_clarifier.evaluation.checkpoint_eval import (
    evaluate_prediction_records,
    load_baseline_reports,
    select_inspection_items,
    write_inspection_samples,
)
from decomp_clarifier.evaluation.compile_eval import compile_candidate
from decomp_clarifier.evaluation.metrics import aggregate_metric, field_complete, placeholder_ratio
from decomp_clarifier.evaluation.naming_eval import normalized_name_similarity
from decomp_clarifier.evaluation.readability_eval import readability_improvement, score_readability
from decomp_clarifier.evaluation.report_builder import (
    build_report,
    render_comparison_table,
    write_report,
)
from decomp_clarifier.ghidra_export.aligner import align_functions, extract_source_functions
from decomp_clarifier.ghidra_export.export_runner import GhidraExportRunner
from decomp_clarifier.ghidra_export.parse_exports import (
    ParsedGhidraProject,
    parse_ghidra_export_dir,
)
from decomp_clarifier.inference.agentic_repair import (
    agentic_prompt,
    build_repair_prompt,
    validate_agentic_answer,
)
from decomp_clarifier.inference.checkpoint_predictor import (
    _encode_prompt,
    _prepare_generation_prompt,
    _text_tokenizer,
)
from decomp_clarifier.inference.explain import summarize_improvements
from decomp_clarifier.inference.formatter import (
    normalize_output,
    normalize_output_with_status,
    strip_thinking_prefix,
)
from decomp_clarifier.inference.runner import InferenceRunner
from decomp_clarifier.logging import configure_logging
from decomp_clarifier.schemas.compiler import BinaryArtifact, CompileManifest
from decomp_clarifier.schemas.evaluation import SampleEvaluation
from decomp_clarifier.schemas.ghidra import GhidraFunctionRow
from decomp_clarifier.schemas.model_io import PredictionRecord


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
    curated_samples = select_training_samples(
        sample_dataset_samples,
        split="train",
        include_task_types=["full_clarify", "cleanup"],
        prompt_limit=4,
    )
    assert len(curated_samples) == 4
    assert {sample.task_type for sample in curated_samples} <= {"full_clarify", "cleanup"}
    rl_records = pack_rl_records(sample_dataset_samples)
    assert len(rl_records) == len(sample_dataset_samples)
    assert rl_records[0].source_function_name == sample_dataset_samples[0].source_function_name
    assert rl_records[0].raw_code == sample_dataset_samples[0].ghidra_decompiled_code
    assert "#include <stdio.h>" in rl_records[0].compile_reference_source
    assert "strlen" in rl_records[0].allowed_callees
    assert sample_dataset_samples[0].source_function_name in rl_records[0].allowed_callees
    assert rl_records[0].compiler_executable == sample_dataset_samples[0].compiler_executable
    assert rl_records[0].tests_ref == sample_dataset_samples[0].tests_ref
    assert rl_records[0].prompt_messages[0].role == "user"
    assert records[0].completion_messages[0].role == "assistant"
    assert "Assembly:" not in rl_records[0].prompt
    assert "Decompiler:" in rl_records[0].prompt

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
    assert (
        compile_candidate(
            sample.target_clean_code,
            sample.compile_reference_source or sample.source_code,
            function_name=sample.source_function_name,
        )
        == compiler_available
    )

    output = normalize_output(
        'prefix {"summary":"ok","confidence":1.0,'
        '"renamings":{},"cleaned_c":"int x(void){return 1;}"} suffix'
    )
    fallback_output, json_valid = normalize_output_with_status("{not valid json")
    assert not json_valid
    assert fallback_output.cleaned_c == "{not valid json"
    assert fallback_output.summary == ""
    thinking_text = (
        "check the function first\n</think>\n\n"
        '{"summary":"ok","confidence":1.0,"renamings":{},"cleaned_c":"int x(void){return 1;}"}'
    )
    thinking_output, thinking_json_valid = normalize_output_with_status(
        thinking_text,
        strip_thinking=True,
    )
    assert thinking_json_valid
    assert thinking_output.summary == "ok"
    assert not normalize_output_with_status(thinking_text)[1]
    assert strip_thinking_prefix(thinking_text).startswith('{"summary":"ok"')
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

    records = [
        PredictionRecord(sample_id=sample.sample_id, system="sft_checkpoint", output=cleaned),
        PredictionRecord(
            sample_id=sample_dataset_samples[1].sample_id,
            system="sft_checkpoint",
            output=raw_ghidra.predict(sample_dataset_samples[1]),
            json_valid=False,
            raw_text="not json",
        ),
    ]
    samples_by_id = {item.sample_id: item for item in sample_dataset_samples[:2]}
    evaluations = evaluate_prediction_records(samples_by_id, records)
    assert len(evaluations) == 2
    assert evaluations[1].json_valid is False
    assert evaluations[1].behavior_success is False
    assert evaluations[1].naming_score == 0.0
    inspection_items = select_inspection_items(
        samples_by_id,
        records,
        evaluations,
        limit=2,
    )
    inspection_md = tmp_path / "inspection.md"
    inspection_jsonl = tmp_path / "inspection.jsonl"
    write_inspection_samples(inspection_items, inspection_md, inspection_jsonl)
    assert inspection_md.exists()
    assert inspection_jsonl.exists()
    assert "### Decompiled" in inspection_md.read_text(encoding="utf-8")


def test_extract_source_functions_ignores_leading_comment_words() -> None:
    from decomp_clarifier.schemas.generation import (
        BuildSpec,
        GeneratedFile,
        GeneratedProject,
        SemanticHints,
    )

    project = GeneratedProject(
        project_id="comment_case",
        summary="test",
        difficulty="easy",
        files=[
            GeneratedFile(
                path="src/main.c",
                content="""#include <stdio.h>
/* Returns number of tokens found in a non-empty line. */
int tokenize(const char *text) {
    return text != NULL;
}
""",
            )
        ],
        tests=[],
        build=BuildSpec(entrypoints=["src/main.c"], c_standard="c11", compiler_family="clang"),
        semantic_hints=SemanticHints(project_purpose="test", function_intents=[]),
    )

    functions = extract_source_functions(project)

    assert [item.name for item in functions] == ["tokenize"]
    assert functions[0].code.startswith("int tokenize(")


def test_align_functions_prefers_best_ghidra_row_for_duplicate_names(
    sample_project, sample_parsed_ghidra_project
) -> None:
    duplicate = GhidraFunctionRow.model_validate(
        sample_parsed_ghidra_project.functions[0].model_dump(mode="python")
        | {
            "function_address": "140000001",
            "instruction_count": 1,
            "basic_block_count": 1,
            "callees": [],
            "callers": ["main"],
            "decompiled_text": "int helper(int param_1) { return helper(param_1); }",
        }
    )
    preferred = GhidraFunctionRow.model_validate(
        sample_parsed_ghidra_project.functions[0].model_dump(mode="python")
        | {
            "function_address": "140000999",
            "instruction_count": 42,
            "basic_block_count": 7,
        }
    )
    parsed = ParsedGhidraProject(
        manifest=sample_parsed_ghidra_project.manifest,
        functions=[
            duplicate,
            preferred,
            *sample_parsed_ghidra_project.functions[1:],
        ],
    )

    aligned = align_functions(sample_project, parsed)
    helper_matches = [
        item for item in aligned if item.source.name == preferred.ghidra_function_name
    ]

    assert len(aligned) == 3
    assert len(helper_matches) == 1
    assert helper_matches[0].ghidra.function_address == preferred.function_address


def test_checkpoint_eval_loader_rejects_duplicate_sample_ids(tmp_path: Path) -> None:
    duplicate_row = {
        "sample_id": "dup",
        "project_id": "project",
        "split": "val",
        "task_type": "full_clarify",
        "host_os": "windows",
        "compiler": "clang",
        "opt_level": "O0",
        "binary_format": "pe",
        "source_function_name": "helper",
        "source_code": "int helper(void) { return 0; }",
        "compile_reference_source": "int helper(void) { return 0; }",
        "target_clean_code": "int helper(void) { return 0; }",
        "ghidra_function_name": "helper",
        "ghidra_decompiled_code": "int helper(void) { return 0; }",
        "assembly": "nop",
        "strings": [],
        "imports": [],
        "callers": [],
        "callees": [],
        "semantic_summary": "helper",
        "rename_map_target": {},
        "tests_ref": None,
        "difficulty": "easy",
    }
    dataset_path = tmp_path / "function_dataset.jsonl"
    dataset_path.write_text(
        "\n".join(json.dumps(duplicate_row) for _ in range(2)) + "\n",
        encoding="utf-8",
    )

    from decomp_clarifier.evaluation.checkpoint_eval import load_dataset_split

    with pytest.raises(ValueError, match="Duplicate sample_id"):
        load_dataset_split(dataset_path, split="val")


def test_render_comparison_table_and_baseline_loader_backward_compat(
    temp_paths, sample_dataset_samples
) -> None:
    samples_by_id = {sample.sample_id: sample for sample in sample_dataset_samples[:2]}
    baseline_run = temp_paths.runs_dir / "baseline-compat"
    baseline_run.mkdir(parents=True, exist_ok=True)
    legacy_record = {
        "sample_id": sample_dataset_samples[0].sample_id,
        "system": "raw_ghidra",
        "output": raw_ghidra.predict(sample_dataset_samples[0]).model_dump(mode="python"),
    }
    baseline_run.joinpath("baseline_predictions.jsonl").write_text(
        json.dumps(legacy_record) + "\n",
        encoding="utf-8",
    )

    baseline_metrics = load_baseline_reports(temp_paths, samples_by_id)

    assert baseline_metrics["raw_ghidra"]["json_valid_rate"] == 1.0
    table = render_comparison_table(
        {
            "raw_ghidra": baseline_metrics["raw_ghidra"],
            "sft_checkpoint": {
                "json_valid_rate": 0.5,
                "compile_success_rate": 0.25,
                "readability_score": 0.75,
            },
        }
    )
    assert "| Metric | raw_ghidra | sft_checkpoint |" in table
    assert "| json_valid_rate | 1.000 | 0.500 |" in table


def test_checkpoint_prompt_encoding_supports_processor_and_tokenizer() -> None:
    class FakeTokenizer:
        pad_token = None
        eos_token = "<eos>"

        def __call__(self, prompt, return_tensors):
            assert return_tensors == "pt"
            return {"input_ids": prompt}

    class FakeProcessor:
        def __init__(self) -> None:
            self.tokenizer = FakeTokenizer()

        def __call__(self, *, text, return_tensors):
            assert text == "hello"
            assert return_tensors == "pt"
            return {"input_ids": "processor"}

    tokenizer = FakeTokenizer()
    processor = FakeProcessor()

    assert _text_tokenizer(tokenizer) is tokenizer
    assert _text_tokenizer(processor) is processor.tokenizer
    assert _encode_prompt(tokenizer, "hello") == {"input_ids": "hello"}
    assert _encode_prompt(processor, "hello") == {"input_ids": "processor"}
    assert _prepare_generation_prompt(tokenizer, tokenizer, "hello") == {"input_ids": "hello"}

    class FakeChatTokenizer(FakeTokenizer):
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            assert tokenize is False
            assert add_generation_prompt is True
            assert messages == [{"role": "user", "content": "hello"}]
            return "<chat>hello</chat>"

    chat_tokenizer = FakeChatTokenizer()
    assert _prepare_generation_prompt(chat_tokenizer, chat_tokenizer, "hello") == {
        "input_ids": "<chat>hello</chat>"
    }

    class FakeThinkingChatTokenizer(FakeTokenizer):
        def apply_chat_template(
            self,
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        ):
            assert tokenize is False
            assert add_generation_prompt is True
            assert messages == [{"role": "user", "content": "hello"}]
            assert enable_thinking is True
            return "<chat><think>hello"

    thinking_tokenizer = FakeThinkingChatTokenizer()
    assert _prepare_generation_prompt(
        thinking_tokenizer,
        thinking_tokenizer,
        "hello",
        enable_thinking=True,
    ) == {"input_ids": "<chat><think>hello"}


def test_agentic_repair_helpers_build_feedback(sample_dataset_samples) -> None:
    sample = sample_dataset_samples[0]
    thinking_prompt = agentic_prompt(format_prompt(sample), enable_thinking=True)
    assert "final answer after </think>" in thinking_prompt
    assert "or <think> blocks" not in thinking_prompt

    output, json_valid, verification, feedback = validate_agentic_answer(
        sample,
        "reasoning</think>\n{not valid json",
        strip_thinking=True,
    )
    assert not json_valid
    assert not verification.field_complete
    assert output.cleaned_c == "{not valid json"
    assert "strict JSON" in feedback[0]

    repair_prompt = build_repair_prompt(
        original_prompt="Original task",
        previous_answer="bad answer",
        feedback=["compile failed", "behavior failed"],
        attempt_index=1,
    )
    assert "Tool feedback:" in repair_prompt
    assert "- compile failed" in repair_prompt
    assert "<answer>\nbad answer\n</answer>" in repair_prompt


def test_rl_prompt_is_compact_relative_to_sft_prompt(sample_dataset_samples) -> None:
    sample = sample_dataset_samples[0]
    sft_prompt = format_prompt(sample)
    rl_prompt = format_rl_prompt(sample)

    assert "Assembly:" in sft_prompt
    assert "Assembly:" not in rl_prompt
    assert len(rl_prompt) < len(sft_prompt)


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
