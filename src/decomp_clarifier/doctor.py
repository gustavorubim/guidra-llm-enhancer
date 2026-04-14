from __future__ import annotations

import os
import platform
import sys
from pathlib import Path
from typing import Any

from decomp_clarifier.adapters.compiler_clang import resolve_clang_executable
from decomp_clarifier.adapters.ghidra_headless import GhidraHeadlessAdapter
from decomp_clarifier.adapters.subprocess_utils import run_subprocess
from decomp_clarifier.paths import ProjectPaths
from decomp_clarifier.settings import load_compile_config, load_ghidra_config
from decomp_clarifier.training.utils.hardware import detect_hardware
from decomp_clarifier.training.utils.version_lock import collect_versions, validate_version_lock
from decomp_clarifier.training.windows_guard import (
    TrainingEnvironmentError,
    ensure_windows_cuda,
    prepare_model_runtime_environment,
)


def _tail(text: str) -> str | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else None


def _repo_pythonpath_env(root: Path) -> dict[str, str]:
    env = prepare_model_runtime_environment(dict(os.environ))
    repo_src = str(root / "src")
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = repo_src if not current else repo_src + os.pathsep + current
    return env


def _probe_python(root: Path) -> dict[str, Any]:
    venv_active = sys.prefix != sys.base_prefix or bool(os.getenv("VIRTUAL_ENV"))
    return {
        "ok": sys.version_info >= (3, 13) and venv_active,
        "version": platform.python_version(),
        "executable": sys.executable,
        "venv_active": venv_active,
        "pythonpath_has_src": str(root / "src") in sys.path,
    }


def _probe_compiler(root: Path) -> dict[str, Any]:
    config = load_compile_config(root, name="clang_o0")
    resolved = resolve_clang_executable(config.compiler.executable)
    return {
        "ok": resolved is not None,
        "requested": config.compiler.executable,
        "resolved": resolved,
    }


def _probe_ghidra(root: Path) -> dict[str, Any]:
    config = load_ghidra_config(root, name="default")
    adapter = GhidraHeadlessAdapter(config, root=root)
    candidate = adapter.analyze_headless_path()
    return {
        "ok": candidate.exists(),
        "path": str(candidate),
    }


def _probe_openrouter() -> dict[str, Any]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    return {
        "ok": bool(api_key),
        "api_key_present": bool(api_key),
    }


def _probe_import(root: Path, code: str) -> dict[str, Any]:
    result = run_subprocess(
        [sys.executable, "-c", code],
        cwd=root,
        env=_repo_pythonpath_env(root),
        timeout_seconds=60,
    )
    return {
        "ok": result.returncode == 0,
        "stdout_tail": _tail(result.stdout),
        "stderr_tail": _tail(result.stderr),
    }


def _probe_training(root: Path) -> dict[str, Any]:
    versions = collect_versions()
    version_lock: dict[str, Any]
    try:
        validate_version_lock()
        version_lock = {"ok": True, "versions": versions}
    except RuntimeError as exc:
        version_lock = {"ok": False, "versions": versions, "error": str(exc)}

    guard: dict[str, Any]
    try:
        ensure_windows_cuda()
        guard = {"ok": True}
    except TrainingEnvironmentError as exc:
        guard = {"ok": False, "error": str(exc)}

    unsloth_import = _probe_import(
        root, "from unsloth import FastLanguageModel; print(FastLanguageModel.__name__)"
    )
    xformers_import = _probe_import(root, "import xformers; print(xformers.__version__)")
    bitsandbytes_import = _probe_import(root, "import bitsandbytes as bnb; print(bnb.__version__)")
    tensorboard_import = _probe_import(
        root,
        "import tensorboard; from tensorboard.main import run_main; print(tensorboard.__version__)",
    )
    trl_grpo_import = _probe_import(
        root,
        "import unsloth; "
        "from decomp_clarifier.training.utils.trl_compat import patch_trl_optional_availability; "
        "patch_trl_optional_availability(); "
        "from trl import GRPOConfig, GRPOTrainer; "
        "print('GRPOTrainer')",
    )

    return {
        "ok": (
            guard["ok"]
            and version_lock["ok"]
            and unsloth_import["ok"]
            and xformers_import["ok"]
            and bitsandbytes_import["ok"]
            and tensorboard_import["ok"]
            and trl_grpo_import["ok"]
        ),
        "windows_cuda_guard": guard,
        "version_lock": version_lock,
        "hardware": detect_hardware(),
        "unsloth_import": unsloth_import,
        "xformers_import": xformers_import,
        "bitsandbytes_import": bitsandbytes_import,
        "tensorboard_import": tensorboard_import,
        "trl_grpo_import": trl_grpo_import,
    }


def build_doctor_report(paths: ProjectPaths, include_training: bool = False) -> dict[str, Any]:
    root = paths.root
    report: dict[str, Any] = {
        "python": _probe_python(root),
        "compiler": _probe_compiler(root),
        "ghidra": _probe_ghidra(root),
        "openrouter": _probe_openrouter(),
    }
    if include_training:
        report["training"] = _probe_training(root)
    return report


def doctor_exit_code(report: dict[str, Any], include_training: bool = False) -> int:
    required = ("python", "compiler", "ghidra")
    if any(not bool(report.get(section, {}).get("ok")) for section in required):
        return 1
    if include_training and not bool(report.get("training", {}).get("ok")):
        return 1
    return 0


def _status(ok: bool) -> str:
    return "ok" if ok else "fail"


def _format_optional(ok: bool) -> str:
    return "ok" if ok else "warn"


def _format_import_probe(label: str, payload: dict[str, Any]) -> str:
    detail = payload.get("stdout_tail") or payload.get("stderr_tail") or "no output"
    return f"[{_status(bool(payload.get('ok')))}] {label}: {detail}"


def render_doctor_report(report: dict[str, Any], include_training: bool = False) -> str:
    python = report["python"]
    compiler = report["compiler"]
    ghidra = report["ghidra"]
    openrouter = report["openrouter"]

    lines = [
        f"[{_status(bool(python['ok']))}] Python: {python['version']} ({python['executable']})",
        f"[{_status(bool(python['venv_active']))}] Virtualenv active: {python['venv_active']}",
        f"[{_status(bool(compiler['ok']))}] Clang: {compiler['resolved'] or compiler['requested']}",
        f"[{_status(bool(ghidra['ok']))}] Ghidra analyzeHeadless: {ghidra['path']}",
        (
            f"[{_format_optional(bool(openrouter['ok']))}] OpenRouter API key present: "
            f"{openrouter['api_key_present']}"
        ),
    ]

    if not include_training:
        return "\n".join(lines)

    training = report["training"]
    hardware = training["hardware"]
    guard = training["windows_cuda_guard"]
    version_lock = training["version_lock"]
    gpu_label = hardware.get("gpu_name") or "unavailable"
    lines.extend(
        [
            f"[{_status(bool(training['ok']))}] Training stack overall",
            f"[{_status(bool(guard['ok']))}] Windows CUDA guard: {guard.get('error', 'ok')}",
            (
                f"[{_status(bool(version_lock['ok']))}] Version lock: "
                f"{version_lock.get('error', 'ok')}"
            ),
            (
                f"[{_status(bool(hardware.get('cuda_available')))}] Torch CUDA: "
                f"{hardware.get('torch_version')} / "
                f"CUDA {hardware.get('cuda_version')} / {gpu_label}"
            ),
            _format_import_probe("Unsloth import", training["unsloth_import"]),
            _format_import_probe("xFormers import", training["xformers_import"]),
            _format_import_probe("bitsandbytes import", training["bitsandbytes_import"]),
            _format_import_probe("TensorBoard import", training["tensorboard_import"]),
            _format_import_probe("TRL GRPO import", training["trl_grpo_import"]),
        ]
    )
    return "\n".join(lines)
