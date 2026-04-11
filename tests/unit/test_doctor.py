from __future__ import annotations

from decomp_clarifier import doctor as doctor_module


def test_doctor_exit_code_and_render() -> None:
    report = {
        "python": {
            "ok": True,
            "version": "3.13.0",
            "executable": "python.exe",
            "venv_active": True,
        },
        "compiler": {"ok": True, "resolved": "clang.exe", "requested": "clang"},
        "ghidra": {"ok": True, "path": "ghidra/support/analyzeHeadless.bat"},
        "openrouter": {"ok": False, "api_key_present": False},
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
            "xformers_import": {"ok": True, "stdout_tail": "0.0.35", "stderr_tail": None},
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
        },
    }

    rendered = doctor_module.render_doctor_report(report, include_training=True)

    assert "[ok] Python: 3.13.0" in rendered
    assert "[warn] OpenRouter API key present: False" in rendered
    assert "[ok] TensorBoard import: 2.20.0" in rendered
    assert doctor_module.doctor_exit_code(report, include_training=True) == 0

    report["compiler"]["ok"] = False
    assert doctor_module.doctor_exit_code(report, include_training=False) == 1


def test_build_doctor_report_calls_training_probe(temp_paths, monkeypatch) -> None:
    monkeypatch.setattr(
        doctor_module,
        "_probe_python",
        lambda root: {
            "ok": True,
            "version": "3.13.0",
            "executable": "python.exe",
            "venv_active": True,
        },
    )
    monkeypatch.setattr(
        doctor_module,
        "_probe_compiler",
        lambda root: {"ok": True, "requested": "clang", "resolved": "clang.exe"},
    )
    monkeypatch.setattr(
        doctor_module,
        "_probe_ghidra",
        lambda root: {"ok": True, "path": "ghidra/support/analyzeHeadless.bat"},
    )
    monkeypatch.setattr(
        doctor_module,
        "_probe_openrouter",
        lambda: {"ok": True, "api_key_present": True},
    )
    monkeypatch.setattr(
        doctor_module,
        "_probe_training",
        lambda root: {"ok": True, "hardware": {"cuda_available": True}},
    )

    report = doctor_module.build_doctor_report(temp_paths, include_training=True)

    assert report["python"]["ok"]
    assert report["compiler"]["resolved"] == "clang.exe"
    assert report["ghidra"]["ok"]
    assert report["openrouter"]["api_key_present"]
    assert report["training"]["ok"]
