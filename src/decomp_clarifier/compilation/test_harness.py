from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path

from decomp_clarifier.adapters.subprocess_utils import SubprocessResult
from decomp_clarifier.schemas.compiler import TestExecutionResult
from decomp_clarifier.schemas.generation import GeneratedTestCase


def run_program(binary_path: Path, stdin_text: str, timeout_seconds: int = 5) -> SubprocessResult:
    completed = subprocess.run(
        [str(binary_path)],
        input=stdin_text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
        check=False,
    )
    return SubprocessResult(
        args=[str(binary_path)],
        cwd=str(binary_path.parent),
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def run_stdio_tests(
    binary_path: Path, tests: Sequence[GeneratedTestCase]
) -> list[TestExecutionResult]:
    results: list[TestExecutionResult] = []
    for test_case in tests:
        result = run_program(binary_path=binary_path, stdin_text=test_case.input)
        passed = result.returncode == 0 and result.stdout.strip() == test_case.expected.strip()
        results.append(
            TestExecutionResult(
                name=test_case.name,
                passed=passed,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        )
    return results
