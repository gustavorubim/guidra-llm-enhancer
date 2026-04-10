from __future__ import annotations

from pathlib import Path

import nox


@nox.session
def tests(session: nox.Session) -> None:
    session.install("-e", ".[dev,test]")
    session.env["PYTHONPATH"] = str(Path("src").resolve())
    session.run(
        "pytest",
        "--cov=src/decomp_clarifier",
        "--cov-config=coverage.toml",
        "--cov-report=term-missing",
        "--cov-fail-under=90",
    )


@nox.session
def lint(session: nox.Session) -> None:
    session.install("-e", ".[dev]")
    session.env["PYTHONPATH"] = str(Path("src").resolve())
    session.run("ruff", "check", ".")
    session.run("ruff", "format", "--check", ".")


@nox.session
def typecheck(session: nox.Session) -> None:
    session.install("-e", ".[dev]")
    session.env["PYTHONPATH"] = str(Path("src").resolve())
    session.run("mypy", "src")
