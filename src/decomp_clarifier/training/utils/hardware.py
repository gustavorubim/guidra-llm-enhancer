from __future__ import annotations

import platform
from importlib import metadata
from typing import Any


def detect_hardware() -> dict[str, Any]:
    hardware: dict[str, Any] = {
        "os": platform.platform(),
        "python_version": platform.python_version(),
    }
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        hardware["cuda_available"] = False
        return hardware

    hardware["torch_version"] = torch.__version__
    hardware["cuda_available"] = bool(torch.cuda.is_available())
    if torch.cuda.is_available():  # pragma: no cover - hardware specific
        properties = torch.cuda.get_device_properties(0)
        hardware["gpu_name"] = properties.name
        hardware["gpu_vram_gb"] = round(properties.total_memory / (1024**3), 2)
        hardware["cuda_version"] = getattr(torch.version, "cuda", None)
        hardware["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
    for package in (
        "unsloth",
        "trl",
        "transformers",
        "datasets",
        "accelerate",
        "tensorboard",
        "matplotlib",
    ):
        try:
            hardware[f"{package}_version"] = metadata.version(package)
        except metadata.PackageNotFoundError:
            hardware[f"{package}_version"] = None
    return hardware
