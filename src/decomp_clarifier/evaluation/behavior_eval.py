from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

from decomp_clarifier.compilation.build_runner import BuildRunner
from decomp_clarifier.c_source import replace_function_definition
from decomp_clarifier.paths import ProjectPaths
from decomp_clarifier.schemas.generation import GeneratedFile, GeneratedProject
from decomp_clarifier.settings import CompileConfig, CompilerProfile


@dataclass(frozen=True)
class ExecutionBehaviorResult:
    compile_success: bool
    pass_rate: float
    test_count: int


def _tokenize(code: str) -> set[str]:
    return set(re.findall(r"[A-Za-z_]\w*", code))


def behavior_similarity(candidate_code: str, reference_code: str) -> float:
    candidate_tokens = _tokenize(candidate_code)
    reference_tokens = _tokenize(reference_code)
    if not reference_tokens:
        return 0.0
    overlap = len(candidate_tokens & reference_tokens)
    union = len(candidate_tokens | reference_tokens)
    return overlap / union if union else 0.0


def is_behavior_improvement(
    candidate_code: str,
    raw_code: str,
    reference_code: str,
) -> bool:
    sim_to_ref = behavior_similarity(candidate_code, reference_code)
    sim_to_raw = behavior_similarity(candidate_code, raw_code)
    # Treat ties as non-regressions because the token-overlap proxy is weak.
    return sim_to_ref >= sim_to_raw


def _repo_root() -> Path:
    return ProjectPaths.discover(Path(__file__).resolve().parent)


def _resolve_tests_manifest(tests_ref: str) -> Path | None:
    if not tests_ref:
        return None
    candidate = Path(tests_ref)
    if candidate.is_absolute():
        return candidate if candidate.exists() else None
    root = _repo_root()
    generated_project_manifest = root / "data" / "raw" / "generated_projects" / candidate
    if generated_project_manifest.exists():
        return generated_project_manifest
    fallback = root / candidate
    return fallback if fallback.exists() else None


@lru_cache(maxsize=256)
def _load_generated_project(tests_ref: str) -> GeneratedProject | None:
    manifest_path = _resolve_tests_manifest(tests_ref)
    if manifest_path is None:
        return None
    try:
        return GeneratedProject.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - reward fallback should remain available
        return None


def _replace_project_function(
    project: GeneratedProject,
    *,
    source_function_name: str,
    cleaned_c: str,
) -> GeneratedProject | None:
    updated_files: list[GeneratedFile] = []
    replaced = False
    for file in project.files:
        content = file.content
        if file.path.endswith((".c", ".h")) and not replaced:
            replacement = replace_function_definition(content, source_function_name, cleaned_c)
            if replacement is not None:
                content = replacement
                replaced = True
        updated_files.append(file.model_copy(update={"content": content}))
    if not replaced:
        return None
    return project.model_copy(update={"files": updated_files})


def _materialize_project(project: GeneratedProject, root: Path) -> None:
    project_root = root / project.project_id
    project_root.mkdir(parents=True, exist_ok=True)
    for file in project.files:
        destination = project_root / file.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(file.content, encoding="utf-8")
    manifest_path = project_root / "project_manifest.json"
    manifest_path.write_text(
        json.dumps(project.model_dump(mode="python"), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _normalized_compiler_family(family: str | None) -> str:
    normalized = (family or "").strip().lower()
    if normalized in {"", "clang", "gcc", "cc"}:
        return "clang"
    return normalized


def _compiler_executable_for_project(
    project: GeneratedProject,
    *,
    compiler_executable: str | None,
    compiler_family: str | None,
) -> str:
    if compiler_executable:
        return compiler_executable
    return _normalized_compiler_family(compiler_family or project.build.compiler_family)


def evaluate_execution_behavior(
    candidate_code: str,
    *,
    source_function_name: str,
    tests_ref: str,
    compiler_executable: str | None = None,
    compiler_family: str | None = None,
) -> ExecutionBehaviorResult | None:
    project = _load_generated_project(tests_ref)
    if project is None or not project.tests:
        return None
    candidate_project = _replace_project_function(
        project,
        source_function_name=source_function_name,
        cleaned_c=candidate_code,
    )
    if candidate_project is None:
        return None
    compile_config = CompileConfig(
        compiler=CompilerProfile(
            family=_normalized_compiler_family(compiler_family or candidate_project.build.compiler_family),
            executable=_compiler_executable_for_project(
                candidate_project,
                compiler_executable=compiler_executable,
                compiler_family=compiler_family,
            ),
            c_standard=candidate_project.build.c_standard,
        )
    )
    try:
        with TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            source_root = temp_root / "projects"
            output_root = temp_root / "binaries"
            _materialize_project(candidate_project, source_root)
            manifest = BuildRunner(compile_config).compile_project(
                candidate_project,
                source_root,
                output_root,
            )
    except Exception:  # noqa: BLE001 - fallback to similarity proxy if execution is unavailable
        return None

    test_count = len(candidate_project.tests)
    if not manifest.binaries:
        return ExecutionBehaviorResult(
            compile_success=False,
            pass_rate=0.0,
            test_count=test_count,
        )
    if test_count == 0:
        return None
    passed = sum(1 for result in manifest.test_results if result.passed)
    return ExecutionBehaviorResult(
        compile_success=True,
        pass_rate=passed / test_count,
        test_count=test_count,
    )
