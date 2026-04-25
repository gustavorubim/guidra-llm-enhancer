from __future__ import annotations

import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO


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
    stdout_sink: TextIO | None = None,
    stderr_sink: TextIO | None = None,
) -> SubprocessResult:
    if stdout_sink is None and stderr_sink is None:
        completed = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
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

    process = subprocess.Popen(
        args,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []

    def _tee(stream: TextIO | None, sink: TextIO | None, chunks: list[str]) -> None:
        if stream is None:
            return
        try:
            while True:
                chunk = stream.read(4096)
                if chunk == "":
                    break
                chunks.append(chunk)
                if sink is not None:
                    sink.write(chunk)
                    sink.flush()
        finally:
            stream.close()

    stdout_thread = threading.Thread(
        target=_tee,
        args=(process.stdout, stdout_sink or sys.stdout, stdout_chunks),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_tee,
        args=(process.stderr, stderr_sink or sys.stderr, stderr_chunks),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()
    try:
        returncode = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout_thread.join()
        stderr_thread.join()
        raise
    stdout_thread.join()
    stderr_thread.join()
    return SubprocessResult(
        args=args,
        cwd=str(cwd or Path.cwd()),
        returncode=returncode,
        stdout="".join(stdout_chunks),
        stderr="".join(stderr_chunks),
    )
