from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from decomp_clarifier.schemas.model_io import ClarifiedFunctionOutput
from decomp_clarifier.training.grpo.data import load_rl_records, prompt_from_record
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
from decomp_clarifier.training.grpo.verifier import verify_output
from decomp_clarifier.training.sft.callbacks import write_training_summary
from decomp_clarifier.training.sft.data import combine_prompt_and_response, load_sft_records
from decomp_clarifier.training.utils.hardware import detect_hardware
from decomp_clarifier.training.utils.memory_profiles import select_memory_profile
from decomp_clarifier.training.utils.version_lock import validate_version_lock
from decomp_clarifier.training.windows_guard import TrainingEnvironmentError, ensure_windows_cuda


def test_training_utilities_and_rewards(
    monkeypatch, tmp_path: Path, sample_dataset_samples
) -> None:
    monkeypatch.setattr("decomp_clarifier.training.windows_guard.platform.system", lambda: "Darwin")
    with pytest.raises(TrainingEnvironmentError):
        ensure_windows_cuda()

    assert select_memory_profile(12) == "windows_cuda_16gb"
    assert select_memory_profile(24) == "windows_cuda_24gb"
    assert select_memory_profile(64) == "windows_cuda_48gb"
    assert "os" in detect_hardware()

    monkeypatch.setattr(
        "decomp_clarifier.training.utils.version_lock.metadata.version",
        lambda name: {
            "unsloth": "2025.4.1",
            "trl": "0.17.1",
            "transformers": "4.51.1",
            "datasets": "3.1.0",
            "accelerate": "1.2.0",
        }[name],
    )
    versions = validate_version_lock()
    assert versions["unsloth"] == "2025.4.1"

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
        def from_pretrained(model_name, max_seq_length, load_in_4bit):
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

        def train(self):
            return None

        def save_model(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)

    class FakeGRPOTrainer(FakeSFTTrainer):
        pass

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
        lambda name: {
            "unsloth": "2025.4.1",
            "trl": "0.17.1",
            "transformers": "4.51.1",
            "datasets": "3.1.0",
            "accelerate": "1.2.0",
        }[name],
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
            },
        }
    )
    sft_manifest = run_sft_training(dataset_path, tmp_path / "sft", config)
    grpo_manifest = run_grpo_training(dataset_path, tmp_path / "grpo", config)
    assert sft_manifest.exists()
    assert grpo_manifest.exists()
