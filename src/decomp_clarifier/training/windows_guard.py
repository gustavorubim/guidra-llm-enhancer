from __future__ import annotations

import os
import platform
from pathlib import Path


class TrainingEnvironmentError(RuntimeError):
    """Raised when the Windows CUDA training path is unavailable."""


def prepare_model_runtime_environment(env: dict[str, str] | None = None) -> dict[str, str]:
    target = os.environ if env is None else env
    for variable in ("SSL_CERT_FILE", "SSL_CERT_DIR"):
        value = target.get(variable)
        if value and not Path(value).expanduser().exists():
            target.pop(variable, None)
    target.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")
    return target


def ensure_windows_cuda() -> None:
    if platform.system() != "Windows":
        raise TrainingEnvironmentError("training commands are only supported on native Windows")
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise TrainingEnvironmentError(
            "torch is not installed in the training environment"
        ) from exc
    if not torch.cuda.is_available():  # pragma: no cover - hardware specific
        raise TrainingEnvironmentError("CUDA is not available")
