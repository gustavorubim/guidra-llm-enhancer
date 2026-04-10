from __future__ import annotations

import re
import tempfile
from pathlib import Path

from decomp_clarifier.adapters.subprocess_utils import run_subprocess, which


def _extract_includes(reference_source: str) -> str:
    return "\n".join(
        line for line in reference_source.splitlines() if line.strip().startswith("#include")
    )


def compile_candidate(cleaned_c: str, reference_source: str, compiler: str = "clang") -> bool:
    resolved = which(compiler)
    if resolved is None or not cleaned_c.strip():
        return False
    includes = _extract_includes(reference_source)
    has_main = re.search(r"\bmain\s*\(", cleaned_c) is not None
    stub_main = "" if has_main else "\nint main(void) { return 0; }\n"
    source = f"{includes}\n{cleaned_c}\n{stub_main}"
    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = Path(tmpdir) / "candidate.c"
        source_path.write_text(source, encoding="utf-8")
        result = run_subprocess([resolved, "-fsyntax-only", str(source_path)], cwd=Path(tmpdir))
        return result.returncode == 0
