from __future__ import annotations


def select_memory_profile(vram_gb: float | None) -> str:
    if vram_gb is None:
        return "unknown"
    if vram_gb < 20:
        return "windows_cuda_16gb"
    if vram_gb < 36:
        return "windows_cuda_24gb"
    return "windows_cuda_48gb"
