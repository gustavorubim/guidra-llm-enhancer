from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SubprocessResult:
    args: list[str]
    cwd: str
    returncode: int
    stdout: str
    stderr: str

    def raise_for_error(self) -> None:
        if self.returncode != 0:
            raise subprocess.CalledProcessError(
                returncode=self.returncode,
                cmd=self.args,
                output=self.stdout,
                stderr=self.stderr,
            )


def which(command: str) -> str | None:
    return shutil.which(command)


def run_subprocess(
    args: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout_seconds: int | None = None,
) -> SubprocessResult:
    completed = subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    return SubprocessResult(
        args=args,
        cwd=str(cwd or Path.cwd()),
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
