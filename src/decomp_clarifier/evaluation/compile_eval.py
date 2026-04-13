from __future__ import annotations

import re
import tempfile
from pathlib import Path

from decomp_clarifier.adapters.compiler_clang import resolve_clang_executable
from decomp_clarifier.adapters.subprocess_utils import run_subprocess, which
from decomp_clarifier.c_source import function_name_from_signature_line, replace_function_definition


def _extract_includes(reference_source: str) -> str:
    return "\n".join(
        line for line in reference_source.splitlines() if line.strip().startswith("#include")
    )


def _compile_source_text(source: str, compiler: str) -> bool:
    resolved = resolve_clang_executable(compiler) or which(compiler)
    if resolved is None or not source.strip():
        return False
    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = Path(tmpdir) / "candidate.c"
        source_path.write_text(source, encoding="utf-8")
        result = run_subprocess([resolved, "-fsyntax-only", str(source_path)], cwd=Path(tmpdir))
        return result.returncode == 0


def compile_candidate(
    cleaned_c: str,
    reference_source: str,
    compiler: str = "clang",
    *,
    function_name: str | None = None,
) -> bool:
    if not cleaned_c.strip():
        return False
    inferred_name = function_name
    if inferred_name is None:
        signature_line = cleaned_c.splitlines()[0] if cleaned_c.splitlines() else ""
        inferred_name = function_name_from_signature_line(signature_line)
    if reference_source.strip():
        replacement_source = replace_function_definition(reference_source, inferred_name, cleaned_c)
        if replacement_source is not None:
            return _compile_source_text(replacement_source, compiler)
    includes = _extract_includes(reference_source)
    has_main = re.search(r"\bmain\s*\(", cleaned_c) is not None
    stub_main = "" if has_main else "\nint main(void) { return 0; }\n"
    return _compile_source_text(f"{includes}\n{cleaned_c}\n{stub_main}", compiler)
