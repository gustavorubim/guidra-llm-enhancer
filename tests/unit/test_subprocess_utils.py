from __future__ import annotations

import io
import sys

from decomp_clarifier.adapters.subprocess_utils import run_subprocess


def test_run_subprocess_captures_output_without_streaming(tmp_path) -> None:
    result = run_subprocess(
        [
            sys.executable,
            "-c",
            "import sys; print('hello'); print('warn', file=sys.stderr)",
        ],
        cwd=tmp_path,
    )

    assert result.returncode == 0
    assert result.stdout == "hello\n"
    assert result.stderr == "warn\n"


def test_run_subprocess_streams_and_captures_output(tmp_path) -> None:
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    result = run_subprocess(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "print('step-1'); "
                "sys.stdout.flush(); "
                "print('problem', file=sys.stderr); "
                "sys.stderr.flush()"
            ),
        ],
        cwd=tmp_path,
        stdout_sink=stdout_buffer,
        stderr_sink=stderr_buffer,
    )

    assert result.returncode == 0
    assert result.stdout == "step-1\n"
    assert result.stderr == "problem\n"
    assert stdout_buffer.getvalue() == result.stdout
    assert stderr_buffer.getvalue() == result.stderr
