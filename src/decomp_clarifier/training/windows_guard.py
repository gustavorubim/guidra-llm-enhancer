from __future__ import annotations

import platform


class TrainingEnvironmentError(RuntimeError):
    """Raised when the Windows CUDA training path is unavailable."""


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
