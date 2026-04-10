from __future__ import annotations

from pathlib import Path

from decomp_clarifier.schemas.compiler import CompileCommand
from decomp_clarifier.settings import CompilerProfile


def build_compile_command_record(
    executable: str,
    args: list[str],
    cwd: Path,
) -> CompileCommand:
    return CompileCommand(executable=executable, args=args, cwd=str(cwd))


def source_file_paths(project_root: Path) -> list[Path]:
    return sorted(path for path in project_root.rglob("*.c") if path.is_file())


def binary_name(project_id: str) -> str:
    return project_id


def compiler_flags(profile: CompilerProfile) -> list[str]:
    flags = [f"-std={profile.c_standard}", f"-{profile.opt_level}", *profile.extra_flags]
    if profile.warnings_as_errors:
        flags.append("-Werror")
    return flags
