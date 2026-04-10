from __future__ import annotations

from pathlib import Path

from decomp_clarifier.compilation.binary_inventory import (
    artifact_for_binary,
    binary_format_for_host,
    host_os_name,
)
from decomp_clarifier.compilation.build_runner import BuildRunner
from decomp_clarifier.compilation.compile_db import binary_name, compiler_flags, source_file_paths
from decomp_clarifier.settings import CompileConfig


def test_build_runner_compiles_and_runs_tests(tmp_path: Path, sample_project) -> None:
    project_root = tmp_path / "generated_projects" / sample_project.project_id
    project_root.mkdir(parents=True)
    for file in sample_project.files:
        path = project_root / file.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(file.content, encoding="utf-8")

    runner = BuildRunner(
        CompileConfig.model_validate({"compiler": {"executable": "clang", "opt_level": "O0"}})
    )
    manifest = runner.compile_project(
        sample_project,
        project_root=tmp_path / "generated_projects",
        output_root=tmp_path / "binaries",
    )
    assert manifest.binaries
    assert all(test.passed for test in manifest.test_results)
    binary_path = Path(manifest.binaries[0].path)
    assert binary_path.exists()

    assert binary_name("sample_project") == "sample_project"
    assert source_file_paths(tmp_path / "generated_projects" / sample_project.project_id)
    assert "-O0" in compiler_flags(runner.config.compiler)
    assert artifact_for_binary(binary_path).binary_format == binary_format_for_host(host_os_name())
