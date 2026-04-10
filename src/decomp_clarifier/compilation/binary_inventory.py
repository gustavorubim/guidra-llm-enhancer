from __future__ import annotations

import platform
from pathlib import Path

from decomp_clarifier.schemas.compiler import BinaryArtifact


def host_os_name() -> str:
    system = platform.system().lower()
    if system == "darwin":
        return "macos"
    if system == "windows":
        return "windows"
    return system


def binary_format_for_host(host_os: str) -> str:
    if host_os == "macos":
        return "macho"
    if host_os == "windows":
        return "pe"
    return "elf"


def artifact_for_binary(path: Path, stripped: bool = False) -> BinaryArtifact:
    host_os = host_os_name()
    arch = platform.machine().lower()
    return BinaryArtifact(
        path=str(path),
        binary_format=binary_format_for_host(host_os),
        arch=arch,
        stripped=stripped,
    )
