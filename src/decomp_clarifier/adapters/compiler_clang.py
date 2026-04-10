from __future__ import annotations

from pathlib import Path

from decomp_clarifier.adapters.subprocess_utils import run_subprocess, which
from decomp_clarifier.settings import CompilerProfile


class ClangCompiler:
    def __init__(self, profile: CompilerProfile) -> None:
        self.profile = profile

    @property
    def executable(self) -> str:
        resolved = which(self.profile.executable)
        if resolved is None:
            raise FileNotFoundError(f"compiler not found: {self.profile.executable}")
        return resolved

    def version(self) -> str:
        result = run_subprocess([self.executable, "--version"])
        result.raise_for_error()
        return result.stdout.splitlines()[0] if result.stdout else "unknown"

    def build_command(self, sources: list[Path], output_path: Path) -> list[str]:
        args = [
            self.executable,
            f"-std={self.profile.c_standard}",
            f"-{self.profile.opt_level}",
            *self.profile.extra_flags,
        ]
        if self.profile.warnings_as_errors:
            args.append("-Werror")
        args.extend(str(path) for path in sources)
        args.extend(["-o", str(output_path)])
        return args

    def compile(
        self, sources: list[Path], output_path: Path, cwd: Path
    ) -> tuple[list[str], str, str, int]:
        command = self.build_command(sources, output_path)
        result = run_subprocess(command, cwd=cwd)
        return command, result.stdout, result.stderr, result.returncode
