from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

# Pre-import torch so its __init__.py runs before any test monkeypatches
# platform.system. torch's _load_global_deps() calls platform.system() to
# decide between .dll / .so / .dylib, and the windows_guard monkeypatch
# briefly makes it return "Darwin", which would cause torch to look for
# libtorch_global_deps.dylib on Windows.
try:
    import torch as _torch_preload  # noqa: F401
except ImportError:
    pass

from decomp_clarifier.adapters.compiler_clang import resolve_clang_executable
from decomp_clarifier.evaluation.behavior_eval import behavior_similarity
from decomp_clarifier.schemas.model_io import ClarifiedFunctionOutput
from decomp_clarifier.training.grpo.data import load_rl_records, prompt_from_record, reward_fields_from_record
from decomp_clarifier.training.grpo.rewards import (
    behavior_reward,
    cleanup_reward,
    compile_reward,
    format_reward,
    hallucination_penalty,
    naming_reward,
    readability_reward,
    weighted_reward,
)
from decomp_clarifier.training.grpo.train import compute_completion_reward
from decomp_clarifier.training.grpo.verifier import verify_output
from decomp_clarifier.training.sft.callbacks import write_training_summary
from decomp_clarifier.training.sft.data import combine_prompt_and_response, load_sft_records
from decomp_clarifier.training.utils.hardware import detect_hardware
from decomp_clarifier.training.utils.memory_profiles import select_memory_profile
from decomp_clarifier.training.utils.version_lock import collect_versions, validate_version_lock
from decomp_clarifier.training.windows_guard import TrainingEnvironmentError, ensure_windows_cuda


def _validated_training_versions() -> dict[str, str]:
    return {
        "unsloth": "2026.4.1",
        "trl": "0.24.0",
        "transformers": "5.5.0",
        "datasets": "4.3.0",
        "accelerate": "1.13.0",
        "tensorboard": "2.19.0",
        "matplotlib": "3.10.3",
    }


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
        lambda name: _validated_training_versions()[name],
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
        lambda name: _validated_training_versions()[name],
    )

    summary_path = write_training_summary(tmp_path / "summary.json", {"loss": 0.1})
    assert json.loads(summary_path.read_text(encoding="utf-8"))["loss"] == 0.1

    dataset_path = tmp_path / "sft.jsonl"
    dataset_path.write_text('{"prompt":"p","response_json":"r"}\n', encoding="utf-8")
    records = load_sft_records(dataset_path)
    assert combine_prompt_and_response(records[0]) == "p\n\nr"

    sample = sample_dataset_samples[0]
    output = ClarifiedFunctionOutput(
        summary="Counts characters.",
        confidence=0.9,
        renamings=sample.rename_map_target,
        cleaned_c=sample.target_clean_code,
    )
    assert format_reward(output) == 1.0
    assert cleanup_reward(output, sample.ghidra_decompiled_code) >= 0.0
    assert naming_reward(output, sample.rename_map_target) == 1.0
    assert compile_reward(True) == 1.0
    assert behavior_reward(True) == 1.0
    assert readability_reward(output, sample.ghidra_decompiled_code) >= 0.0
    assert hallucination_penalty(output, sample.imports, sample.callees) >= 0.0
    assert (
        weighted_reward(
            output=output,
            raw_code=sample.ghidra_decompiled_code,
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
                "hallucination_penalty": 1.0,
            },
        )
        >= 0.0
    )
    verification = verify_output(sample, output)
    assert verification.field_complete

    rl_path = tmp_path / "rl.jsonl"
    rl_path.write_text('{"prompt":"prompt text"}\n', encoding="utf-8")
    rl_records = load_rl_records(rl_path)
    assert prompt_from_record(rl_records[0]) == "prompt text"
    empty_fields = reward_fields_from_record({})
    assert empty_fields["raw_code"] == ""
    assert empty_fields["compile_reference_source"] == ""
    assert empty_fields["target_renamings"] == "{}"

    populated_fields = reward_fields_from_record(
        {
            "prompt": "p",
            "raw_code": "raw",
            "compile_reference_source": "#include <stdio.h>\nint helper(void) { return 0; }\n",
            "target_clean_code": "int helper(void) { return 0; }",
            "target_renamings": '{"local_10":"result"}',
            "allowed_imports": '["printf"]',
            "allowed_callees": '["printf"]',
        }
    )
    assert populated_fields["compile_reference_source"].startswith("#include <stdio.h>")

    assert behavior_similarity("int helper(void) { return 0; }", "") == 0.0
    compile_only_reward = compute_completion_reward(
        completion='{"summary":"ok","confidence":1.0,"renamings":{},"cleaned_c":"int helper(void){ printf(\\"hi\\\\n\\"); return 0; }"}',
        raw_code='int helper(void){ undefined8 local_10; printf("hi\\n"); return 0; }',
        compile_reference_source='#include <stdio.h>\nint helper(void) { return 0; }\n',
        target_clean_code="int helper(void) { return 0; }",
        target_renamings_json="{}",
        allowed_imports_json='["printf"]',
        allowed_callees_json='["printf"]',
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
    assert compile_only_reward == (1.0 if resolve_clang_executable("clang") is not None else 0.0)

    assert compute_completion_reward(
        completion='{"summary":"x","confidence":0.5,"renamings":{},"cleaned_c":"int f(void){return 0;}"}',
        raw_code="",
        compile_reference_source="",
        target_clean_code="",
        target_renamings_json="not valid json",
        allowed_imports_json="[]",
        allowed_callees_json="[]",
        weights={},
    ) == 0.0


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
        lambda name: _validated_training_versions()[name],
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

    class FakeGRPOTrainer(FakeSFTTrainer):
        def train(self):
            reward_funcs = self.kwargs.get("reward_funcs", [])
            if reward_funcs:
                reward_funcs[0](
                    ["test completion"],
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
        lambda name: _validated_training_versions()[name],
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(
        sys.modules, "unsloth", types.SimpleNamespace(FastLanguageModel=FakeFastLanguageModel)
    )
    monkeypatch.setitem(sys.modules, "datasets", FakeDatasetsModule())
    monkeypatch.setitem(sys.modules, "trl", fake_trl)

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
                "behavior_similarity_threshold": 0.0,
            },
        }
    )
    sft_manifest = run_sft_training(dataset_path, tmp_path / "sft", config)
    grpo_manifest = run_grpo_training(dataset_path, tmp_path / "grpo", config)
    assert sft_manifest.exists()
    assert grpo_manifest.exists()

    sft_payload = json.loads(sft_manifest.read_text(encoding="utf-8"))
    grpo_payload = json.loads(grpo_manifest.read_text(encoding="utf-8"))

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
