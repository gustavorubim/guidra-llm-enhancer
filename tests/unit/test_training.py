from __future__ import annotations

import json
import os
import sys
import types
from contextlib import suppress
from importlib import metadata as importlib_metadata
from pathlib import Path

import pytest

# Pre-import torch so its __init__.py runs before any test monkeypatches
# platform.system. torch's _load_global_deps() calls platform.system() to
# decide between .dll / .so / .dylib, and the windows_guard monkeypatch
# briefly makes it return "Darwin", which would cause torch to look for
# libtorch_global_deps.dylib on Windows.
with suppress(ImportError):
    import torch as _torch_preload  # noqa: F401

from decomp_clarifier.adapters.compiler_clang import resolve_clang_executable
from decomp_clarifier.evaluation.behavior_eval import (
    behavior_similarity,
    evaluate_execution_behavior,
)
from decomp_clarifier.schemas.model_io import ClarifiedFunctionOutput
from decomp_clarifier.training.grpo.data import (
    load_rl_records,
    prompt_from_record,
    reward_fields_from_record,
)
from decomp_clarifier.training.grpo.rewards import (
    behavior_reward,
    cleanup_reward,
    compile_reward,
    decompiler_type_penalty,
    format_reward,
    hallucination_penalty,
    invalid_completion_length_penalty,
    invalid_scope_penalty,
    multi_function_penalty,
    naming_reward,
    overshoot_penalty,
    readability_reward,
    safety_gate_factor,
    signature_reward,
    truncation_penalty,
    unknown_constant_penalty,
    unsupported_bool_penalty,
    weighted_reward,
)
from decomp_clarifier.training.grpo.train import (
    _completion_text,
    _resolve_multi_reward_weights,
    compute_completion_reward,
)
from decomp_clarifier.training.grpo.verifier import verify_output
from decomp_clarifier.training.sft.callbacks import write_training_summary
from decomp_clarifier.training.sft.data import (
    combine_prompt_and_response,
    load_sft_records,
    prompt_completion_from_record,
)
from decomp_clarifier.training.utils import telemetry as training_telemetry
from decomp_clarifier.training.utils.hardware import detect_hardware
from decomp_clarifier.training.utils.memory_profiles import select_memory_profile
from decomp_clarifier.training.utils.trl_compat import (
    ensure_model_warnings_issued,
    normalize_optional_flag,
)
from decomp_clarifier.training.utils.version_lock import (
    collect_versions,
    validate_version_lock,
)
from decomp_clarifier.training.windows_guard import (
    TrainingEnvironmentError,
    ensure_windows_cuda,
    prepare_model_runtime_environment,
)

_ORIGINAL_METADATA_VERSION = importlib_metadata.version


def _validated_training_versions() -> dict[str, str]:
    return {
        "unsloth": "2026.4.1",
        "trl": "0.24.0",
        "transformers": "5.5.0",
        "datasets": "4.3.0",
        "accelerate": "1.13.0",
        "tensorboard": "2.20.0",
        "matplotlib": "3.10.3",
    }


def _version_with_fallback(name: str):
    validated = _validated_training_versions()
    if name in validated:
        return validated[name]
    return _ORIGINAL_METADATA_VERSION(name)


def test_resolve_multi_reward_weights_rejects_wrong_length() -> None:
    with pytest.raises(ValueError, match="must match the number of reward functions"):
        _resolve_multi_reward_weights([1.0, 2.0])


def test_completion_text_extracts_conversational_completion() -> None:
    assert (
        _completion_text(
            [
                {"role": "assistant", "content": '{"summary":"ok","cleaned_c":"int x;"}'}
            ]
        )
        == '{"summary":"ok","cleaned_c":"int x;"}'
    )
    assert _completion_text({"content": "plain content"}) == "plain content"
    assert _completion_text(["prefix", {"content": "body"}]) == "prefix\nbody"


def test_model_source_access_sets_offline_mode_for_cached_remote_model(monkeypatch) -> None:
    from decomp_clarifier.training.sft import model as sft_model

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(sft_model, "_is_local_model_reference", lambda model_name: False)
    monkeypatch.setattr(
        sft_model,
        "_cached_remote_snapshot_dir",
        lambda model_name: Path("C:/hf-cache/Qwen3.5-2B"),
    )
    monkeypatch.setattr(sft_model, "_can_resolve_huggingface", lambda: False)

    resolved = sft_model._resolve_model_source("Qwen/Qwen3.5-2B")

    assert Path(resolved) == Path("C:/hf-cache/Qwen3.5-2B")
    assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_model_source_access_raises_clear_error_for_uncached_remote_model(monkeypatch) -> None:
    from decomp_clarifier.training.sft import model as sft_model

    monkeypatch.setattr(sft_model, "_is_local_model_reference", lambda model_name: False)
    monkeypatch.setattr(sft_model, "_cached_remote_snapshot_dir", lambda model_name: None)
    monkeypatch.setattr(sft_model, "_can_resolve_huggingface", lambda: False)
    monkeypatch.setattr(sft_model, "_install_public_dns_fallback_if_needed", lambda: False)

    with pytest.raises(RuntimeError, match="Restore DNS/internet access to Hugging Face"):
        sft_model._resolve_model_source("unsloth/gemma-4-E2B-it")


def test_model_source_access_uses_public_dns_fallback_for_uncached_remote_model(
    monkeypatch,
) -> None:
    from decomp_clarifier.training.sft import model as sft_model

    monkeypatch.setattr(sft_model, "_is_local_model_reference", lambda model_name: False)
    monkeypatch.setattr(sft_model, "_cached_remote_snapshot_dir", lambda model_name: None)
    monkeypatch.setattr(sft_model, "_can_resolve_huggingface", lambda: False)
    monkeypatch.setattr(sft_model, "_install_public_dns_fallback_if_needed", lambda: True)

    resolved = sft_model._resolve_model_source("unsloth/gemma-4-E2B-it")

    assert resolved == "unsloth/gemma-4-E2B-it"


def test_parse_nslookup_addresses_ignores_dns_server_address() -> None:
    from decomp_clarifier.training.sft import model as sft_model

    output = """
Server:  one.one.one.one
Address:  1.1.1.1

Name:    huggingface.co
Addresses:  3.166.152.44
          3.166.152.105
"""

    assert sft_model._parse_nslookup_addresses(output) == ["3.166.152.44", "3.166.152.105"]


def test_candidate_remote_model_ids_prefers_unsloth_4bit_repo() -> None:
    from decomp_clarifier.training.sft import model as sft_model

    assert sft_model._candidate_remote_model_ids("unsloth/gemma-4-E2B-it", True) == [
        "unsloth/gemma-4-E2B-it-unsloth-bnb-4bit",
    ]
    assert sft_model._candidate_remote_model_ids("Qwen/Qwen3.5-2B", True) == [
        "Qwen/Qwen3.5-2B"
    ]


def test_transient_snapshot_error_detection() -> None:
    from decomp_clarifier.training.sft import model as sft_model

    assert sft_model._is_transient_snapshot_error(
        RuntimeError("[WinError 10051] A socket operation was attempted to an unreachable network")
    )
    assert sft_model._is_transient_snapshot_error(
        RuntimeError(
            "[WinError 10054] An existing connection was forcibly closed by the remote host"
        )
    )
    assert not sft_model._is_transient_snapshot_error(RuntimeError("Repository not found"))


def test_prefetch_remote_snapshot_uses_cached_unsloth_4bit_repo(monkeypatch) -> None:
    from decomp_clarifier.training.sft import model as sft_model

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(sft_model, "_is_local_model_reference", lambda model_name: False)
    monkeypatch.setattr(
        sft_model,
        "_cached_remote_snapshot_dir",
        lambda repo_id: (
            Path("C:/hf-cache/gemma-4bit")
            if repo_id == "unsloth/gemma-4-E2B-it-unsloth-bnb-4bit"
            else None
        ),
    )
    monkeypatch.setattr(sft_model, "_snapshot_dir_has_required_files", lambda snapshot_dir: True)

    resolved = sft_model._prefetch_remote_snapshot_dir("unsloth/gemma-4-E2B-it", True)

    assert resolved == Path("C:/hf-cache/gemma-4bit")
    assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_snapshot_dir_requires_all_shards(tmp_path: Path) -> None:
    from decomp_clarifier.training.sft import model as sft_model

    snapshot_dir = tmp_path / "snapshot"
    snapshot_dir.mkdir()
    (snapshot_dir / "config.json").write_text("{}", encoding="utf-8")
    (snapshot_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layer_0": "model-00001-of-00002.safetensors",
                    "layer_1": "model-00002-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    (snapshot_dir / "model-00001-of-00002.safetensors").write_bytes(b"shard-one")

    assert not sft_model._snapshot_dir_has_required_files(snapshot_dir)

    (snapshot_dir / "model-00002-of-00002.safetensors").write_bytes(b"shard-two")

    assert sft_model._snapshot_dir_has_required_files(snapshot_dir)


def test_snapshot_dir_rejects_sharded_weights_without_index(tmp_path: Path) -> None:
    from decomp_clarifier.training.sft import model as sft_model

    snapshot_dir = tmp_path / "snapshot-no-index"
    snapshot_dir.mkdir()
    (snapshot_dir / "config.json").write_text("{}", encoding="utf-8")
    (snapshot_dir / "model-00001-of-00003.safetensors").write_bytes(b"partial-shard")

    assert not sft_model._snapshot_dir_has_required_files(snapshot_dir)


def test_load_model_and_tokenizer_prefers_prefetched_snapshot(monkeypatch) -> None:
    from decomp_clarifier.settings import TrainingConfig
    from decomp_clarifier.training.sft import model as sft_model

    captured: dict[str, object] = {}

    class FakeFastLanguageModel:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            captured["model_name"] = model_name
            return object(), object()

        @staticmethod
        def get_peft_model(model, **kwargs):
            return model

    monkeypatch.setitem(
        sys.modules,
        "unsloth",
        types.SimpleNamespace(FastLanguageModel=FakeFastLanguageModel),
    )
    monkeypatch.setattr(
        sft_model,
        "_prefetch_remote_snapshot_dir",
        lambda model_name, load_in_4bit: Path("C:/hf-cache/gemma-4bit"),
    )
    monkeypatch.setattr(
        sft_model,
        "_resolve_model_source",
        lambda model_name: "C:/should-not-be-used",
    )
    config = TrainingConfig.model_validate(
        {
            "model": {"base_model_id": "unsloth/gemma-4-E2B-it", "loader_variant": "unsloth"},
            "training": {"load_in_4bit": True, "lora_rank": 8, "max_seq_length": 512},
        }
    )

    sft_model.load_model_and_tokenizer(config)

    assert Path(str(captured["model_name"])) == Path("C:/hf-cache/gemma-4bit")


def test_training_utilities_and_rewards(
    monkeypatch, tmp_path: Path, sample_dataset_samples
) -> None:
    monkeypatch.setattr("decomp_clarifier.training.windows_guard.platform.system", lambda: "Darwin")
    with pytest.raises(TrainingEnvironmentError):
        ensure_windows_cuda()

    assert select_memory_profile(12) == "windows_cuda_16gb"
    assert select_memory_profile(24) == "windows_cuda_24gb"
    assert select_memory_profile(64) == "windows_cuda_48gb"
    assert select_memory_profile(None) == "unknown"
    assert "os" in detect_hardware()

    monkeypatch.setattr(
        "decomp_clarifier.training.utils.version_lock.metadata.version",
        _version_with_fallback,
    )
    versions = validate_version_lock()
    assert versions["unsloth"] == "2026.4.1"

    from importlib.metadata import PackageNotFoundError

    monkeypatch.setattr(
        "decomp_clarifier.training.utils.version_lock.metadata.version",
        lambda name: (_ for _ in ()).throw(PackageNotFoundError(name)),
    )
    missing = collect_versions()
    assert missing["unsloth"] is None

    with pytest.raises(RuntimeError, match="validated versions"):
        validate_version_lock()

    monkeypatch.setattr(
        "decomp_clarifier.training.utils.version_lock.metadata.version",
        _version_with_fallback,
    )

    summary_path = write_training_summary(tmp_path / "summary.json", {"loss": 0.1})
    assert json.loads(summary_path.read_text(encoding="utf-8"))["loss"] == 0.1

    dataset_path = tmp_path / "sft.jsonl"
    dataset_path.write_text(
        '{"prompt":"p","response_json":"r","prompt_messages":[{"role":"user","content":"p"}],'
        '"completion_messages":[{"role":"assistant","content":"r"}]}\n',
        encoding="utf-8",
    )
    records = load_sft_records(dataset_path)
    assert combine_prompt_and_response(records[0]) == "p\nr"
    assert combine_prompt_and_response(records[0], eos_token="<eos>") == "p\nr<eos>"
    assert prompt_completion_from_record(records[0]) == {
        "prompt": [{"role": "user", "content": "p"}],
        "completion": [{"role": "assistant", "content": "r"}],
    }

    sample = sample_dataset_samples[0]
    output = ClarifiedFunctionOutput(
        summary="Counts characters.",
        confidence=0.9,
        renamings=sample.rename_map_target,
        cleaned_c=sample.target_clean_code,
    )
    assert format_reward(output) == 1.0
    assert format_reward(output, exact_json=False) == 0.75
    assert cleanup_reward(output, sample.ghidra_decompiled_code) >= 0.0
    assert naming_reward(output, sample.rename_map_target) == 1.0
    assert compile_reward(True) == 1.0
    assert behavior_reward(0.4) == 0.4
    assert behavior_reward(2.0) == 1.0
    assert readability_reward(output, sample.ghidra_decompiled_code) >= 0.0
    assert signature_reward(output, sample.target_clean_code, sample.source_function_name) == 1.0
    assert decompiler_type_penalty(output) == 0.0
    assert hallucination_penalty(output, sample.imports, sample.callees) >= 0.0
    assert safety_gate_factor(compile_success=True, behavior_success=True) == 1.0
    assert safety_gate_factor(compile_success=False, behavior_success=True) < 1.0
    assert (
        weighted_reward(
            output=output,
            json_valid=True,
            raw_code=sample.ghidra_decompiled_code,
            target_clean_code=sample.target_clean_code,
            source_function_name=sample.source_function_name,
            target_renamings=sample.rename_map_target,
            compile_success=True,
            behavior_success=True,
            allowed_imports=sample.imports,
            allowed_callees=sample.callees,
            weights={
                "format": 1.0,
                "cleanup": 1.0,
                "naming": 1.0,
                "compile": 1.0,
                "behavior": 1.0,
                "readability": 1.0,
                "signature": 1.0,
                "hallucination_penalty": 1.0,
                "decompiler_type_penalty": 1.0,
            },
        )
        >= 0.0
    )
    decompiler_output = ClarifiedFunctionOutput(
        summary="Counts characters.",
        confidence=0.9,
        renamings={},
        cleaned_c="ulong64 helper(undefined8 param_1) { return 0; }",
    )
    assert decompiler_type_penalty(decompiler_output) == pytest.approx(2 / 3)
    hallucinating_output = ClarifiedFunctionOutput(
        summary="Calls too many helpers.",
        confidence=0.9,
        renamings={},
        cleaned_c="int helper(void) { foo(); bar(); baz(); qux(); return 0; }",
    )
    assert hallucination_penalty(hallucinating_output, ["puts"], []) == 1.0
    known_constant_output = ClarifiedFunctionOutput(
        summary="Uses known constants.",
        confidence=0.9,
        renamings={},
        cleaned_c="int helper(char *dst, char *src) { return copy(dst, src, MAX_VAL); }",
    )
    max_val_reference = "int helper(char *dst, char *src) { return copy(dst, src, MAX_VAL); }"
    assert (
        unknown_constant_penalty(
            known_constant_output,
            raw_code=max_val_reference,
            target_clean_code=max_val_reference,
        )
        == 0.0
    )
    invented_constant_output = ClarifiedFunctionOutput(
        summary="Invents a similar constant.",
        confidence=0.9,
        renamings={},
        cleaned_c="int helper(char *dst, char *src) { return copy(dst, src, MAX_VALUE); }",
    )
    assert (
        unknown_constant_penalty(
            invented_constant_output,
            raw_code=max_val_reference,
            target_clean_code=max_val_reference,
        )
        > 0.0
    )
    bool_output = ClarifiedFunctionOutput(
        summary="Uses a boolean flag.",
        confidence=0.9,
        renamings={},
        cleaned_c="int helper(const char *s) { bool seen = false; return seen; }",
    )
    assert (
        unsupported_bool_penalty(
            bool_output,
            target_clean_code="int helper(const char *s) { int seen = 0; return seen; }",
        )
        == 1.0
    )
    assert (
        unsupported_bool_penalty(
            bool_output,
            target_clean_code="int helper(const char *s) { int seen = 0; return seen; }",
            compile_reference_source="#include <stdbool.h>\n",
        )
        == 0.0
    )
    assert (
        overshoot_penalty("int helper(void) { return 0; }", "int helper(void) { return 0; }")
        == 0.0
    )
    assert (
        overshoot_penalty(
            "int helper(void) { int total = 0; total += 1; total += 2; total += 3; return total; }",
            "int helper(void) { return 0; }",
            max_completion_ratio=1.5,
        )
        > 0.0
    )
    assert multi_function_penalty("int helper(void) { return 0; }") == 0.0
    assert (
        multi_function_penalty(
            "int helper(void) { return 0; }\nint other(void) { return 1; }",
            max_function_count=1,
        )
        == pytest.approx(1.0)
    )
    assert (
        invalid_completion_length_penalty(
            '{"cleaned_c":"int main(void) { return 0; int x = 1; int y = 2; }"',
            "int main(void) { return 0; }",
            max_invalid_completion_ratio=0.9,
        )
        > 0.0
    )
    assert (
        truncation_penalty(
            '{"cleaned_c":"int main(void) { return 0; }", "confidence": 1.0, "summary": "'
        )
        > 0.0
    )
    assert (
        invalid_scope_penalty(
            '{"cleaned_c":"int main(void) { return 0; }", "summary": "',
            "count_flags",
        )
        == pytest.approx(1.0)
    )
    assert (
        invalid_scope_penalty(
            '{"cleaned_c":"int count_flags(void) { return 0; }", "summary": "',
            "count_flags",
        )
        == pytest.approx(0.0)
    )
    assert signature_reward(
        decompiler_output, sample.target_clean_code, sample.source_function_name
    ) < 1.0
    typed_target = "int helper(size_t length) { return 0; }"
    typed_mismatch_output = ClarifiedFunctionOutput(
        summary="Returns zero.",
        confidence=0.9,
        renamings={},
        cleaned_c="int helper(ulong64 length) { return 0; }",
    )
    assert signature_reward(typed_mismatch_output, typed_target, "helper") < 1.0
    verification = verify_output(sample, output)
    assert verification.field_complete

    rl_path = tmp_path / "rl.jsonl"
    rl_path.write_text(
        '{"prompt":"prompt text","prompt_messages":[{"role":"user","content":"prompt text"}]}\n',
        encoding="utf-8",
    )
    rl_records = load_rl_records(rl_path)
    assert prompt_from_record(rl_records[0]) == [{"role": "user", "content": "prompt text"}]
    assert prompt_from_record({"prompt": "fallback prompt"}) == "fallback prompt"
    empty_fields = reward_fields_from_record({})
    assert empty_fields["task_type"] == "full_clarify"
    assert empty_fields["raw_code"] == ""
    assert empty_fields["compile_reference_source"] == ""
    assert empty_fields["target_renamings"] == "{}"
    assert empty_fields["compiler_executable"] is None
    assert empty_fields["tests_ref"] == ""

    populated_fields = reward_fields_from_record(
        {
            "prompt": "p",
            "task_type": "rename",
            "source_function_name": "helper",
            "raw_code": "raw",
            "compile_reference_source": "#include <stdio.h>\nint helper(void) { return 0; }\n",
            "target_clean_code": "int helper(void) { return 0; }",
            "target_renamings": '{"local_10":"result"}',
            "allowed_imports": '["printf"]',
            "allowed_callees": '["printf"]',
            "compiler_executable": "clang",
            "tests_ref": "sample_project/project_manifest.json",
        }
    )
    assert populated_fields["task_type"] == "rename"
    assert populated_fields["source_function_name"] == "helper"
    assert populated_fields["compile_reference_source"].startswith("#include <stdio.h>")
    assert populated_fields["compiler_executable"] == "clang"
    assert populated_fields["tests_ref"] == "sample_project/project_manifest.json"

    assert behavior_similarity("int helper(void) { return 0; }", "") == 0.0


def test_prepare_model_runtime_environment_sanitizes_invalid_cert_paths(
    monkeypatch, tmp_path: Path
) -> None:
    missing_file = tmp_path / "missing-cert.pem"
    missing_dir = tmp_path / "missing-certs"
    existing_file = tmp_path / "existing-cert.pem"
    existing_dir = tmp_path / "existing-certs"
    existing_file.write_text("dummy", encoding="utf-8")
    existing_dir.mkdir()

    monkeypatch.setenv("SSL_CERT_FILE", str(missing_file))
    monkeypatch.setenv("SSL_CERT_DIR", str(missing_dir))
    monkeypatch.delenv("UNSLOTH_DISABLE_STATISTICS", raising=False)

    prepare_model_runtime_environment()

    assert "SSL_CERT_FILE" not in os.environ
    assert "SSL_CERT_DIR" not in os.environ
    assert os.environ["HF_HUB_DISABLE_XET"] == "1"
    assert os.environ["UNSLOTH_DISABLE_STATISTICS"] == "1"

    monkeypatch.setenv("SSL_CERT_FILE", str(existing_file))
    monkeypatch.setenv("SSL_CERT_DIR", str(existing_dir))

    prepare_model_runtime_environment()

    assert os.environ["SSL_CERT_FILE"] == str(existing_file)
    assert os.environ["SSL_CERT_DIR"] == str(existing_dir)


def test_model_source_access_attempts_dns_fallback_before_online_lookup(monkeypatch) -> None:
    from decomp_clarifier.training.sft import model as sft_model

    calls: list[str] = []

    monkeypatch.setattr(sft_model, "_is_local_model_reference", lambda model_name: False)
    monkeypatch.setattr(sft_model, "_cached_remote_snapshot_dir", lambda model_name: None)
    monkeypatch.setattr(
        sft_model,
        "_install_public_dns_fallback_if_needed",
        lambda: calls.append("install") or True,
    )
    monkeypatch.setattr(sft_model, "_can_resolve_huggingface", lambda: True)

    resolved = sft_model._resolve_model_source("unsloth/gemma-4-E2B-it")

    assert resolved == "unsloth/gemma-4-E2B-it"
    assert calls == ["install"]
    compile_only_reward = compute_completion_reward(
        completion=(
            '{"summary":"ok","confidence":1.0,"renamings":{},'
            '"cleaned_c":"int helper(void){ printf(\\"hi\\\\n\\"); return 0; }"}'
        ),
        task_type="full_clarify",
        source_function_name="helper",
        raw_code='int helper(void){ undefined8 local_10; printf("hi\\n"); return 0; }',
        compile_reference_source='#include <stdio.h>\nint helper(void) { return 0; }\n',
        target_clean_code="int helper(void) { return 0; }",
        target_renamings_json="{}",
        allowed_imports_json='["printf"]',
        allowed_callees_json='["printf"]',
        compiler_executable="clang",
        tests_ref="",
        weights={
            "format": 0.0,
            "cleanup": 0.0,
            "naming": 0.0,
            "compile": 1.0,
            "behavior": 0.0,
            "readability": 0.0,
            "hallucination_penalty": 0.0,
        },
    )
    expected_compile_only = (
        1.0
        if resolve_clang_executable("clang") is not None
        else 0.0
    )
    assert compile_only_reward == expected_compile_only
    continuous_behavior_reward = weighted_reward(
        output=ClarifiedFunctionOutput(
            summary="ok",
            confidence=1.0,
            renamings={},
            cleaned_c="int helper(void) { return 0; }",
        ),
        json_valid=True,
        raw_code="int helper(void){ undefined8 local_10; return 0; }",
        target_clean_code="int helper(int value) { return value; }",
        source_function_name="helper",
        target_renamings={},
        compile_success=True,
        behavior_success=False,
        behavior_score=0.25,
        behavior_improvement=True,
        allowed_imports=[],
        allowed_callees=[],
        weights={
            "format": 0.0,
            "cleanup": 0.0,
            "naming": 0.0,
            "compile": 0.0,
            "behavior": 1.0,
            "readability": 0.0,
            "signature": 0.0,
            "hallucination_penalty": 0.0,
            "decompiler_type_penalty": 0.0,
        },
    )
    assert continuous_behavior_reward == pytest.approx(0.0)
    regressive_behavior_reward = weighted_reward(
        output=ClarifiedFunctionOutput(
            summary="ok",
            confidence=1.0,
            renamings={},
            cleaned_c="int helper(void) { return 0; }",
        ),
        json_valid=True,
        raw_code="int helper(void){ undefined8 local_10; return 0; }",
        target_clean_code="int helper(int value) { return value; }",
        source_function_name="helper",
        target_renamings={},
        compile_success=True,
        behavior_success=False,
        behavior_score=0.8,
        behavior_improvement=False,
        allowed_imports=[],
        allowed_callees=[],
        weights={
            "format": 0.0,
            "cleanup": 0.0,
            "naming": 0.0,
            "compile": 0.0,
            "behavior": 1.0,
            "readability": 0.0,
            "signature": 0.0,
            "hallucination_penalty": 0.0,
            "decompiler_type_penalty": 0.0,
        },
    )
    assert regressive_behavior_reward == pytest.approx(-0.25)
    compile_failure_capped_reward = weighted_reward(
        output=ClarifiedFunctionOutput(
            summary="ok",
            confidence=1.0,
            renamings={"local_10": "result"},
            cleaned_c="int helper(void) { return 0; }",
        ),
        json_valid=True,
        raw_code="int helper(void){ undefined8 local_10; return 0; }",
        target_clean_code="int helper(void) { return 0; }",
        source_function_name="helper",
        target_renamings={"local_10": "result"},
        compile_success=False,
        behavior_success=True,
        behavior_score=1.0,
        behavior_improvement=True,
        allowed_imports=[],
        allowed_callees=[],
        weights={
            "format": 1.0,
            "cleanup": 1.0,
            "naming": 1.0,
            "compile": 3.0,
            "behavior": 3.0,
            "readability": 1.0,
            "signature": 1.0,
            "hallucination_penalty": 0.0,
            "decompiler_type_penalty": 0.0,
        },
    )
    compile_success_reward = weighted_reward(
        output=ClarifiedFunctionOutput(
            summary="ok",
            confidence=1.0,
            renamings={"local_10": "result"},
            cleaned_c="int helper(void) { return 0; }",
        ),
        json_valid=True,
        raw_code="int helper(void){ undefined8 local_10; return 0; }",
        target_clean_code="int helper(void) { return 0; }",
        source_function_name="helper",
        target_renamings={"local_10": "result"},
        compile_success=True,
        behavior_success=True,
        behavior_score=1.0,
        behavior_improvement=True,
        allowed_imports=[],
        allowed_callees=[],
        weights={
            "format": 1.0,
            "cleanup": 1.0,
            "naming": 1.0,
            "compile": 3.0,
            "behavior": 3.0,
            "readability": 1.0,
            "signature": 1.0,
            "hallucination_penalty": 0.0,
            "decompiler_type_penalty": 0.0,
        },
    )
    assert compile_failure_capped_reward <= 0.0
    assert compile_failure_capped_reward < compile_success_reward

    guarded_multi_function_reward = weighted_reward(
        output=ClarifiedFunctionOutput(
            summary="ok",
            confidence=1.0,
            renamings={"local_10": "result"},
            cleaned_c="int helper(void) { return 0; }\nint other(void) { return 1; }",
        ),
        json_valid=True,
        raw_code="int helper(void){ undefined8 local_10; return 0; }",
        target_clean_code="int helper(void) { return 0; }",
        source_function_name="helper",
        target_renamings={"local_10": "result"},
        compile_success=True,
        behavior_success=True,
        behavior_score=1.0,
        behavior_improvement=True,
        allowed_imports=[],
        allowed_callees=[],
        weights={
            "format": 1.0,
            "cleanup": 0.0,
            "naming": 0.0,
            "compile": 1.0,
            "behavior": 1.0,
            "readability": 0.0,
            "signature": 1.0,
            "hallucination_penalty": 0.0,
            "decompiler_type_penalty": 0.0,
            "overshoot_penalty": 0.0,
            "multi_function_penalty": 3.0,
        },
        max_function_count=1,
    )
    single_function_reward = weighted_reward(
        output=ClarifiedFunctionOutput(
            summary="ok",
            confidence=1.0,
            renamings={"local_10": "result"},
            cleaned_c="int helper(void) { return 0; }",
        ),
        json_valid=True,
        raw_code="int helper(void){ undefined8 local_10; return 0; }",
        target_clean_code="int helper(void) { return 0; }",
        source_function_name="helper",
        target_renamings={"local_10": "result"},
        compile_success=True,
        behavior_success=True,
        behavior_score=1.0,
        behavior_improvement=True,
        allowed_imports=[],
        allowed_callees=[],
        weights={
            "format": 1.0,
            "cleanup": 0.0,
            "naming": 0.0,
            "compile": 1.0,
            "behavior": 1.0,
            "readability": 0.0,
            "signature": 1.0,
            "hallucination_penalty": 0.0,
            "decompiler_type_penalty": 0.0,
            "overshoot_penalty": 0.0,
            "multi_function_penalty": 3.0,
        },
        max_function_count=1,
    )
    assert guarded_multi_function_reward < single_function_reward

    guarded_overshoot_reward = weighted_reward(
        output=ClarifiedFunctionOutput(
            summary="ok",
            confidence=1.0,
            renamings={},
            cleaned_c=(
                "int helper(void) { int total = 0; total += 1; total += 2; total += 3; "
                "total += 4; total += 5; return total; }"
            ),
        ),
        json_valid=True,
        raw_code="int helper(void){ return 0; }",
        target_clean_code="int helper(void) { return 0; }",
        source_function_name="helper",
        target_renamings={},
        compile_success=True,
        behavior_success=True,
        behavior_score=1.0,
        behavior_improvement=True,
        allowed_imports=[],
        allowed_callees=[],
        weights={
            "format": 1.0,
            "cleanup": 0.0,
            "naming": 0.0,
            "compile": 1.0,
            "behavior": 1.0,
            "readability": 0.0,
            "signature": 1.0,
            "hallucination_penalty": 0.0,
            "decompiler_type_penalty": 0.0,
            "overshoot_penalty": 2.0,
            "multi_function_penalty": 0.0,
        },
        max_completion_ratio=1.25,
    )
    bounded_reward = weighted_reward(
        output=ClarifiedFunctionOutput(
            summary="ok",
            confidence=1.0,
            renamings={},
            cleaned_c="int helper(void) { return 0; }",
        ),
        json_valid=True,
        raw_code="int helper(void){ return 0; }",
        target_clean_code="int helper(void) { return 0; }",
        source_function_name="helper",
        target_renamings={},
        compile_success=True,
        behavior_success=True,
        behavior_score=1.0,
        behavior_improvement=True,
        allowed_imports=[],
        allowed_callees=[],
        weights={
            "format": 1.0,
            "cleanup": 0.0,
            "naming": 0.0,
            "compile": 1.0,
            "behavior": 1.0,
            "readability": 0.0,
            "signature": 1.0,
            "hallucination_penalty": 0.0,
            "decompiler_type_penalty": 0.0,
            "overshoot_penalty": 2.0,
            "multi_function_penalty": 0.0,
        },
        max_completion_ratio=1.25,
    )
    assert guarded_overshoot_reward < bounded_reward

    assert compute_completion_reward(
        completion=(
            '{"summary":"x","confidence":0.5,"renamings":{},'
            '"cleaned_c":"int f(void){return 0;}"}'
        ),
        task_type="full_clarify",
        source_function_name="f",
        raw_code="",
        compile_reference_source="",
        target_clean_code="",
        target_renamings_json="not valid json",
        allowed_imports_json="[]",
        allowed_callees_json="[]",
        compiler_executable=None,
        tests_ref="",
        weights={},
    ) == 0.0
    assert (
        compute_completion_reward(
            completion=(
                '{"cleaned_c":"int main(void) { return 0; int x = 1; int y = 2; }", '
                '"confidence": 1.0, "summary": "'
            ),
            task_type="full_clarify",
            source_function_name="main",
            raw_code="int main(void) { return 0; }",
            compile_reference_source="",
            target_clean_code="int main(void) { return 0; }",
            target_renamings_json="{}",
            allowed_imports_json="[]",
            allowed_callees_json="[]",
            compiler_executable=None,
            tests_ref="",
            weights={
                "invalid_json_penalty": 0.25,
                "invalid_length_penalty": 1.0,
                "truncation_penalty": 2.0,
                "invalid_scope_penalty": 2.0,
            },
            max_invalid_completion_ratio=0.9,
        )
        < 0.0
    )

    assert not normalize_optional_flag((False, None))
    assert normalize_optional_flag((True, "1.0"))
    assert not normalize_optional_flag(False)

    leaf = types.SimpleNamespace()
    middle = types.SimpleNamespace(model=leaf)
    root = types.SimpleNamespace(base_model=middle)
    assert ensure_model_warnings_issued(root) == 3
    assert isinstance(root.warnings_issued, dict)
    assert isinstance(middle.warnings_issued, dict)
    assert isinstance(leaf.warnings_issued, dict)


def test_execution_behavior_uses_project_tests(
    tmp_path: Path,
    sample_project,
) -> None:
    if resolve_clang_executable("clang") is None:
        pytest.skip("clang is required for execution-backed behavior checks")

    project_root = tmp_path / sample_project.project_id
    project_root.mkdir(parents=True, exist_ok=True)
    manifest_path = project_root / "project_manifest.json"
    manifest_path.write_text(sample_project.model_dump_json(indent=2), encoding="utf-8")

    correct_candidate = """static int count_letters(const char *text) {
    int total = 0;
    while (*text != '\\0') {
        total += 1;
        text += 1;
    }
    return total;
}"""
    correct_result = evaluate_execution_behavior(
        correct_candidate,
        source_function_name="count_letters",
        tests_ref=str(manifest_path),
    )
    assert correct_result is not None
    assert correct_result.compile_success is True
    assert correct_result.pass_rate == pytest.approx(1.0)

    broken_result = evaluate_execution_behavior(
        "static int count_letters(const char *text) { return 0; }",
        source_function_name="count_letters",
        tests_ref=str(manifest_path),
    )
    assert broken_result is not None
    assert broken_result.compile_success is True
    assert broken_result.pass_rate == pytest.approx(0.0)


def test_execution_behavior_supports_real_generated_project_manifest() -> None:
    compiler = resolve_clang_executable("clang")
    if compiler is None:
        pytest.skip("clang is required for execution-backed behavior checks")

    root = Path(__file__).resolve().parents[2]
    rl_path = root / "data" / "processed" / "rl" / "rl_records.jsonl"
    if not rl_path.exists():
        pytest.skip("rl_records.jsonl fixture is not available")

    for line in rl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        tests_ref = row.get("tests_ref")
        if not tests_ref:
            continue
        result = evaluate_execution_behavior(
            row["target_clean_code"],
            source_function_name=row["source_function_name"],
            compiler_executable=compiler,
            compiler_family="clang",
            tests_ref=tests_ref,
        )
        assert result is not None
        assert result.compile_success is True
        assert result.pass_rate == pytest.approx(1.0)
        return

    pytest.fail("expected at least one RL record with tests_ref")


def test_min_train_samples_gate(monkeypatch, tmp_path: Path) -> None:
    from decomp_clarifier.settings import TrainingConfig
    from decomp_clarifier.training.sft.train import run_sft_training

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def get_device_properties(_index: int):
            return types.SimpleNamespace(name="GPU", total_memory=24 * 1024**3)

        @staticmethod
        def is_bf16_supported() -> bool:
            return True

    fake_torch = types.SimpleNamespace(
        __version__="2.7.0", cuda=FakeCuda(), version=types.SimpleNamespace(cuda="12.4")
    )

    class FakeSmallDataset:
        def __len__(self) -> int:
            return 5

        def map(self, _func):
            return self

    class FakeDatasetsSmall(types.SimpleNamespace):
        @staticmethod
        def load_dataset(_kind, data_files, split):
            return FakeSmallDataset()

    class FakeSFTTrainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.state = types.SimpleNamespace(global_step=0, epoch=0.0, log_history=[])

        def train(self):
            callbacks = self.kwargs.get("callbacks", [])
            self.state.global_step = 1
            self.state.epoch = 1.0
            log_row = {"loss": 0.75, "learning_rate": 2e-4}
            self.state.log_history.append({"step": 1, "epoch": 1.0, **log_row})
            for callback in callbacks:
                callback.on_log(None, self.state, None, logs=log_row)
            return None

        def save_model(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)

    fake_trl = types.SimpleNamespace(
        SFTConfig=lambda **kwargs: types.SimpleNamespace(**kwargs),
        SFTTrainer=FakeSFTTrainer,
        GRPOConfig=lambda **kwargs: types.SimpleNamespace(**kwargs),
        GRPOTrainer=FakeSFTTrainer,
    )

    monkeypatch.setattr(
        "decomp_clarifier.training.windows_guard.platform.system", lambda: "Windows"
    )
    monkeypatch.setattr(
        "decomp_clarifier.training.utils.version_lock.metadata.version",
        _version_with_fallback,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(
        sys.modules,
        "unsloth",
        types.SimpleNamespace(
            FastLanguageModel=type(
                "FL",
                (),
                {
                    "from_pretrained": staticmethod(lambda *args, **kwargs: (object(), object())),
                    "get_peft_model": staticmethod(lambda model, **kwargs: model),
                },
            )
        ),
    )
    monkeypatch.setitem(sys.modules, "datasets", FakeDatasetsSmall())
    monkeypatch.setitem(sys.modules, "trl", fake_trl)

    dataset_path = tmp_path / "records.jsonl"
    dataset_path.write_text('{"prompt":"p","response_json":"r"}\n', encoding="utf-8")
    config = TrainingConfig.model_validate(
        {
            "model": {"base_model_id": "Qwen/Qwen3.5-4B", "loader_variant": "unsloth"},
            "training": {"min_train_samples": 300, "max_seq_length": 512, "batch_size": 1},
        }
    )
    with pytest.raises(ValueError, match="min_train_samples"):
        run_sft_training(dataset_path, tmp_path / "sft_out", config)


def test_run_training_wrappers_with_fake_modules(
    monkeypatch, tmp_path: Path, sample_dataset_samples
) -> None:
    from decomp_clarifier.settings import TrainingConfig
    from decomp_clarifier.training.grpo.train import run_grpo_training
    from decomp_clarifier.training.sft.train import run_sft_training

    captured_grpo_args: list[object] = []

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def get_device_properties(_index: int):
            return types.SimpleNamespace(name="Fake GPU", total_memory=24 * 1024**3)

        @staticmethod
        def is_bf16_supported() -> bool:
            return True

    fake_torch = types.SimpleNamespace(
        __version__="2.7.0", cuda=FakeCuda(), version=types.SimpleNamespace(cuda="12.4")
    )

    class FakeFastLanguageModel:
        @staticmethod
        def from_pretrained(model_name, max_seq_length, load_in_4bit, **kwargs):
            return object(), object()

        @staticmethod
        def get_peft_model(model, **kwargs):
            return model

    class FakeDataset:
        def map(self, _func):
            return self

    class FakeDatasetsModule(types.SimpleNamespace):
        @staticmethod
        def load_dataset(_kind, data_files, split):
            return FakeDataset()

    class FakeSFTTrainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.state = types.SimpleNamespace(global_step=0, epoch=0.0, log_history=[])

        def train(self):
            callbacks = self.kwargs.get("callbacks", [])
            self.state.global_step = 1
            self.state.epoch = 0.5
            log_row = {"loss": 0.42, "learning_rate": 2e-4}
            self.state.log_history.append({"step": 1, "epoch": 0.5, **log_row})
            for callback in callbacks:
                callback.on_log(None, self.state, None, logs=log_row)
            return types.SimpleNamespace(metrics={"train_loss": 0.42})

        def save_model(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)

    captured_reward_func_counts: list[int] = []

    class FakeGRPOTrainer(FakeSFTTrainer):
        def __init__(self, **kwargs):
            assert "processing_class" in kwargs
            assert "tokenizer" not in kwargs
            captured_grpo_args.append(kwargs["args"])
            captured_reward_func_counts.append(len(kwargs.get("reward_funcs", [])))
            super().__init__(**kwargs)

        def train(self):
            reward_funcs = self.kwargs.get("reward_funcs", [])
            for reward_func in reward_funcs:
                reward_func(
                    [
                        [
                            {
                                "role": "assistant",
                                "content": (
                                    '{"summary":"ok","confidence":1.0,"renamings":{},'
                                    '"cleaned_c":"int helper(void) { return 0; }"}'
                                ),
                            }
                        ]
                    ],
                    source_function_name=["helper"],
                    raw_code=["int uVar3(void){ int local_10; return local_10; }"],
                    compile_reference_source=["int helper(void) { return 0; }"],
                    target_clean_code=["int helper(void) { return 0; }"],
                    target_renamings=['{"local_10":"result"}'],
                    allowed_imports=["[]"],
                    allowed_callees=["[]"],
                )
            callbacks = self.kwargs.get("callbacks", [])
            self.state.global_step = 2
            self.state.epoch = 1.0
            log_row = {"reward": 0.9, "kl": 0.05}
            self.state.log_history.append({"step": 2, "epoch": 1.0, **log_row})
            for callback in callbacks:
                callback.on_log(None, self.state, None, logs=log_row)
            return types.SimpleNamespace(metrics={"reward": 0.9})

    fake_trl = types.SimpleNamespace(
        SFTConfig=lambda **kwargs: types.SimpleNamespace(**kwargs),
        SFTTrainer=FakeSFTTrainer,
        GRPOConfig=lambda **kwargs: types.SimpleNamespace(**kwargs),
        GRPOTrainer=FakeGRPOTrainer,
    )

    monkeypatch.setattr(
        "decomp_clarifier.training.windows_guard.platform.system", lambda: "Windows"
    )
    monkeypatch.setattr(
        "decomp_clarifier.training.utils.version_lock.metadata.version",
        _version_with_fallback,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(
        sys.modules, "unsloth", types.SimpleNamespace(FastLanguageModel=FakeFastLanguageModel)
    )
    monkeypatch.setitem(sys.modules, "datasets", FakeDatasetsModule())
    monkeypatch.setitem(sys.modules, "trl", fake_trl)
    monkeypatch.setattr(
        "decomp_clarifier.training.grpo.train.patch_trl_optional_availability",
        lambda: None,
    )

    dataset_path = tmp_path / "records.jsonl"
    dataset_path.write_text('{"prompt":"prompt","response_json":"response"}\n', encoding="utf-8")
    config = TrainingConfig.model_validate(
        {
            "model": {"base_model_id": "Qwen/Qwen3.5-4B", "loader_variant": "unsloth"},
            "training": {
                "max_seq_length": 1024,
                "load_in_4bit": True,
                "lora_rank": 8,
                "batch_size": 1,
                "grad_accum_steps": 1,
                "epochs": 1,
                "max_prompt_length": 128,
                "max_completion_length": 64,
                "generations_per_prompt": 2,
                "loss_type": "dr_grpo",
                "multi_reward_weights": [1.0, 2.0, 0.5],
                "scale_rewards": "group",
                "beta": 0.0,
                "mask_truncated_completions": True,
                "learning_rate": 7e-6,
                "adam_beta1": 0.85,
                "adam_beta2": 0.97,
                "weight_decay": 0.05,
                "warmup_ratio": 0.2,
                "lr_scheduler_type": "linear",
                "optim": "adamw_8bit",
                "max_grad_norm": 0.2,
                "rollout_temperature": 0.8,
                "rollout_top_p": 0.9,
                "rollout_top_k": 50,
                "rollout_min_p": 0.05,
                "rollout_repetition_penalty": 1.02,
                "save_steps": 17,
                "behavior_similarity_threshold": 0.0,
            },
        }
    )
    sft_manifest = run_sft_training(dataset_path, tmp_path / "sft", config)
    grpo_manifest = run_grpo_training(dataset_path, tmp_path / "grpo", config)
    assert sft_manifest.exists()
    assert grpo_manifest.exists()
    assert len(captured_grpo_args) == 1
    assert captured_reward_func_counts == [3]
    assert captured_grpo_args[0].learning_rate == 7e-6
    assert captured_grpo_args[0].adam_beta1 == 0.85
    assert captured_grpo_args[0].adam_beta2 == 0.97
    assert captured_grpo_args[0].weight_decay == 0.05
    assert captured_grpo_args[0].warmup_ratio == 0.2
    assert captured_grpo_args[0].lr_scheduler_type == "linear"
    assert captured_grpo_args[0].optim == "adamw_8bit"
    assert captured_grpo_args[0].max_grad_norm == 0.2
    assert captured_grpo_args[0].temperature == 0.8
    assert captured_grpo_args[0].top_p == 0.9
    assert captured_grpo_args[0].top_k == 50
    assert captured_grpo_args[0].min_p == 0.05
    assert captured_grpo_args[0].repetition_penalty == 1.02
    assert captured_grpo_args[0].save_steps == 17
    assert captured_grpo_args[0].loss_type == "dr_grpo"
    assert captured_grpo_args[0].reward_weights == [1.0, 2.0, 0.5]
    assert captured_grpo_args[0].scale_rewards == "group"
    assert captured_grpo_args[0].beta == 0.0
    assert captured_grpo_args[0].mask_truncated_completions is True

    sft_payload = json.loads(sft_manifest.read_text(encoding="utf-8"))
    grpo_payload = json.loads(grpo_manifest.read_text(encoding="utf-8"))
    assert grpo_payload["model"]["base_model_id"] == "Qwen/Qwen3.5-4B"
    trainer_payload = grpo_payload["trainer"]
    assert trainer_payload["class"].endswith("FakeGRPOTrainer")
    assert trainer_payload["loss_type"] == "dr_grpo"
    assert trainer_payload["scale_rewards"] == "group"
    assert trainer_payload["reward_objectives"] == [
        {"name": "correctness", "field": "core_total"},
        {"name": "style", "field": "style_total"},
        {"name": "constraints", "field": "constraint_total"},
    ]
    assert trainer_payload["reward_weights"] == [1.0, 2.0, 0.5]
    assert trainer_payload["source_path"]
    assert trainer_payload["source_sha256"]

    for payload, stage, plot_name in (
        (sft_payload, "sft", "loss"),
        (grpo_payload, "grpo", "reward"),
    ):
        telemetry = payload["telemetry"]
        assert telemetry["row_count"] >= 1
        assert Path(telemetry["metrics_jsonl"]).exists()
        assert Path(telemetry["metrics_csv"]).exists()
        assert Path(telemetry["tensorboard_dir"]).exists()
        assert Path(telemetry["plots"][plot_name]["path"]).exists()
        summary_path = tmp_path / stage / "logs" / f"{stage}_summary.json"
        assert summary_path.exists()

    sft_jsonl = (tmp_path / "sft" / "logs" / "sft_metrics.jsonl").read_text(encoding="utf-8")
    grpo_jsonl = (tmp_path / "grpo" / "logs" / "grpo_metrics.jsonl").read_text(encoding="utf-8")
    assert "loss" in sft_jsonl
    assert "reward_mean" in grpo_jsonl


def test_grpo_reward_plot_prefers_reward_func_mean(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_plot_metrics(
        *,
        rows: list[dict[str, object]],
        metric_keys: list[str],
        output_path: Path,
        title: str,
        ylabel: str,
    ) -> dict[str, object]:
        captured["rows"] = rows
        captured["metric_keys"] = metric_keys
        captured["title"] = title
        captured["ylabel"] = ylabel
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"png")
        return {
            "path": str(output_path),
            "metrics": metric_keys,
            "rendered": bool(metric_keys),
            "reason": None if metric_keys else "no matching metrics",
        }

    monkeypatch.setattr(training_telemetry, "_plot_metrics", fake_plot_metrics)
    telemetry = training_telemetry.TrainingTelemetry("grpo", tmp_path)
    telemetry.record_metrics(
        {
            "reward_count": 4,
            "reward_max": 8.0,
            "reward_mean": 3.5,
            "reward_min": 1.0,
            "reward_std": 2.0,
        },
        step=1,
        source="reward_func",
    )
    telemetry.record_metrics(
        {
            "reward": 3.5,
            "reward_std": 2.1,
            "rewards/reward_func/mean": 3.5,
        },
        step=5,
        source="trainer",
    )

    summary = telemetry.finalize()

    assert summary["plots"]["reward"]["metrics"] == ["reward_mean"]
    assert captured["metric_keys"] == ["reward_mean"]
    assert all(row["source"] == "reward_func" for row in captured["rows"])


def test_grpo_reward_plot_falls_back_to_trainer_reward(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_plot_metrics(
        *,
        rows: list[dict[str, object]],
        metric_keys: list[str],
        output_path: Path,
        title: str,
        ylabel: str,
    ) -> dict[str, object]:
        captured["rows"] = rows
        captured["metric_keys"] = metric_keys
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"png")
        return {
            "path": str(output_path),
            "metrics": metric_keys,
            "rendered": bool(metric_keys),
            "reason": None if metric_keys else "no matching metrics",
        }

    monkeypatch.setattr(training_telemetry, "_plot_metrics", fake_plot_metrics)
    telemetry = training_telemetry.TrainingTelemetry("grpo", tmp_path)
    telemetry.record_metrics(
        {
            "reward": 2.25,
            "reward_std": 0.75,
            "rewards/reward_func/mean": 2.25,
        },
        step=9,
        source="trainer",
    )
    telemetry.record_metrics(
        {
            "loss": 0.1,
        },
        step=10,
        source="train_result",
    )

    summary = telemetry.finalize()

    assert summary["plots"]["reward"]["metrics"] == ["reward"]
    assert captured["metric_keys"] == ["reward"]
    assert all(row["source"] == "trainer" for row in captured["rows"])
