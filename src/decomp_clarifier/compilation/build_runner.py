from __future__ import annotations

import json
import uuid
from pathlib import Path

from decomp_clarifier.adapters.compiler_clang import ClangCompiler
from decomp_clarifier.compilation.binary_inventory import (
    artifact_for_binary,
    binary_format_for_host,
    host_os_name,
)
from decomp_clarifier.compilation.compile_db import (
    binary_name,
    build_compile_command_record,
    source_file_paths,
)
from decomp_clarifier.compilation.test_harness import run_stdio_tests
from decomp_clarifier.schemas.compiler import CompileManifest
from decomp_clarifier.schemas.generation import GeneratedProject
from decomp_clarifier.settings import CompileConfig


class BuildRunner:
    def __init__(self, config: CompileConfig) -> None:
        self.config = config
        self.compiler = ClangCompiler(config.compiler)

    def compile_project(
        self, project: GeneratedProject, project_root: Path, output_root: Path
    ) -> CompileManifest:
        source_root = project_root / project.project_id
        binary_root = output_root / project.project_id
        binary_root.mkdir(parents=True, exist_ok=True)
        output_name = binary_name(project.project_id)
        if host_os_name() == "windows":
            output_name += ".exe"
        output_path = binary_root / output_name

        sources = source_file_paths(source_root)
        command, stdout, stderr, returncode = self.compiler.compile(
            sources, output_path, cwd=source_root
        )
        manifest = CompileManifest(
            project_id=project.project_id,
            build_id=str(uuid.uuid4()),
            compiler_family=self.config.compiler.family,
            compiler_version=self.compiler.version(),
            host_os=host_os_name(),
            binary_format=binary_format_for_host(host_os_name()),
            arch=artifact_for_binary(output_path).arch,
            opt_level=self.config.compiler.opt_level,
            source_root=str(source_root),
            output_root=str(binary_root),
            build_log="\n".join(part for part in (stdout, stderr) if part),
            compile_commands=[
                build_compile_command_record(
                    executable=command[0],
                    args=command[1:],
                    cwd=source_root,
                )
            ],
            binaries=[artifact_for_binary(output_path)] if returncode == 0 else [],
            test_results=[],
        )
        if returncode == 0 and project.tests:
            manifest.test_results = run_stdio_tests(output_path, project.tests)

        manifest_path = binary_root / "compile_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest.model_dump(mode="python"), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return manifest
