from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from decomp_clarifier.dataset.builders import build_function_dataset
from decomp_clarifier.paths import ProjectPaths
from decomp_clarifier.schemas.compiler import CompileManifest
from decomp_clarifier.schemas.generation import (
    BuildSpec,
    FunctionIntent,
    GeneratedFile,
    GeneratedProject,
    GeneratedTestCase,
    SemanticHints,
)
from decomp_clarifier.schemas.ghidra import GhidraFunctionRow, GhidraProjectManifest, VariableRecord
from decomp_clarifier.settings import (
    AppConfig,
    DatasetConfig,
    DatasetConfigData,
    GhidraConfig,
    PathsConfig,
    RunConfig,
)


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def sample_project() -> GeneratedProject:
    return GeneratedProject(
        project_id="sample_project",
        summary="Counts letters from stdin.",
        difficulty="easy",
        files=[
            GeneratedFile(
                path="src/main.c",
                content="""#include <stdio.h>
#include <string.h>

static void trim_newline(char *text) {
    size_t length = strlen(text);
    if (length > 0 && text[length - 1] == '\\n') {
        text[length - 1] = '\\0';
    }
}

static int count_letters(const char *text) {
    int total = 0;
    while (*text != '\\0') {
        total += 1;
        text += 1;
    }
    return total;
}

int main(void) {
    char buffer[128];
    if (fgets(buffer, sizeof(buffer), stdin) == NULL) {
        return 1;
    }
    trim_newline(buffer);
    printf("%d\\n", count_letters(buffer));
    return 0;
}
""",
            )
        ],
        tests=[GeneratedTestCase(name="counts_letters", input="abcd\n", expected="4")],
        build=BuildSpec(entrypoints=["src/main.c"], c_standard="c11", compiler_family="clang"),
        semantic_hints=SemanticHints(
            project_purpose="Count input characters after trimming the newline.",
            function_intents=[
                FunctionIntent(function_name="trim_newline", intent="Remove a trailing newline."),
                FunctionIntent(
                    function_name="count_letters", intent="Count characters in a string."
                ),
                FunctionIntent(
                    function_name="main", intent="Read stdin and print the character count."
                ),
            ],
        ),
    )


@pytest.fixture
def sample_compile_manifest() -> CompileManifest:
    return CompileManifest(
        project_id="sample_project",
        build_id="build-1",
        compiler_family="clang",
        compiler_version="clang 22.1.2",
        host_os="macos",
        binary_format="macho",
        arch="arm64",
        opt_level="O0",
        source_root="/tmp/generated/sample_project",
        output_root="/tmp/binaries/sample_project",
        compile_commands=[],
        binaries=[],
        test_results=[],
    )


@pytest.fixture
def sample_parsed_ghidra_project(tmp_path: Path):
    from decomp_clarifier.ghidra_export.parse_exports import ParsedGhidraProject

    manifest = GhidraProjectManifest(
        project_id="sample_project",
        binary_path="/tmp/binaries/sample_project/sample_project",
        binary_name="sample_project",
        output_dir=str(tmp_path / "sample_project"),
        functions_path=str(tmp_path / "sample_project" / "functions.jsonl"),
    )
    functions = [
        GhidraFunctionRow(
            project_id="sample_project",
            binary_path=manifest.binary_path,
            binary_name=manifest.binary_name,
            function_address="1000",
            ghidra_function_name="trim_newline",
            signature="void trim_newline(char * param_1)",
            return_type="void",
            parameters=[VariableRecord(name="param_1", type="char *", storage="register")],
            local_variables=[],
            decompiled_text="""void trim_newline(char *param_1) {
  int local_10;
  local_10 = strlen(param_1);
  if ((0 < local_10) && (param_1[local_10 + -1] == '\\n')) {
    param_1[local_10 + -1] = '\\0';
  }
}""",
            disassembly_text="ldr x0, [sp]\nbl strlen",
            strings=[],
            imports=["strlen"],
            callees=["strlen"],
            callers=["main"],
            basic_block_count=2,
            instruction_count=6,
        ),
        GhidraFunctionRow(
            project_id="sample_project",
            binary_path=manifest.binary_path,
            binary_name=manifest.binary_name,
            function_address="1010",
            ghidra_function_name="count_letters",
            signature="int count_letters(char * param_1)",
            return_type="int",
            parameters=[VariableRecord(name="param_1", type="char *", storage="register")],
            local_variables=[VariableRecord(name="local_10", type="int", storage="stack")],
            decompiled_text="""int count_letters(char *param_1) {
  int local_10;
  local_10 = 0;
  while (*param_1 != '\\0') {
    local_10 = local_10 + 1;
    param_1 = param_1 + 1;
  }
  return local_10;
}""",
            disassembly_text="mov w0, #0\ncbz x1",
            strings=[],
            imports=[],
            callees=[],
            callers=["main"],
            basic_block_count=2,
            instruction_count=8,
        ),
        GhidraFunctionRow(
            project_id="sample_project",
            binary_path=manifest.binary_path,
            binary_name=manifest.binary_name,
            function_address="1020",
            ghidra_function_name="FUN_001020",
            signature="int FUN_001020(void)",
            return_type="int",
            parameters=[],
            local_variables=[VariableRecord(name="local_80", type="char [128]", storage="stack")],
            decompiled_text="""int FUN_001020(void) {
  char local_80 [128];
  if (fgets(local_80,0x80,stdin) == (char *)0x0) {
    return 1;
  }
  trim_newline(local_80);
  printf("%d\\n",count_letters(local_80));
  return 0;
}""",
            disassembly_text="bl fgets\nbl trim_newline\nbl printf",
            strings=["%d\\n"],
            imports=["fgets", "printf"],
            callees=["trim_newline", "count_letters", "fgets", "printf"],
            callers=[],
            basic_block_count=3,
            instruction_count=12,
        ),
    ]
    return ParsedGhidraProject(manifest=manifest, functions=functions)


@pytest.fixture
def dataset_config() -> DatasetConfig:
    return DatasetConfig(
        dataset=DatasetConfigData(
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            seed=7,
            task_mix={"full_clarify": 0.5, "cleanup": 0.3, "rename": 0.2},
        )
    )


@pytest.fixture
def sample_dataset_samples(
    sample_project: GeneratedProject,
    sample_compile_manifest: CompileManifest,
    sample_parsed_ghidra_project,
    dataset_config: DatasetConfig,
):
    return build_function_dataset(
        projects=[sample_project],
        compile_manifests=[sample_compile_manifest],
        parsed_exports=[sample_parsed_ghidra_project],
        config=dataset_config,
    )


@pytest.fixture
def temp_app_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        run=RunConfig(
            log_level="INFO",
            openrouter_base_url="https://openrouter.ai/api/v1",
            log_to_console=False,
        ),
        paths=PathsConfig(
            generated_projects_dir="generated_projects",
            manifests_dir="manifests",
            binaries_dir="binaries",
            ghidra_exports_dir="ghidra_exports",
            aligned_projects_dir="aligned_projects",
            aligned_functions_dir="aligned_functions",
            processed_sft_dir="processed_sft",
            processed_rl_dir="processed_rl",
            processed_eval_dir="processed_eval",
            reports_dir="reports",
            runs_dir="runs",
            logs_dir="logs",
        ),
        ghidra=GhidraConfig(project_dir="ghidra/project", script_dir="ghidra/scripts"),
    )


@pytest.fixture
def temp_paths(tmp_path: Path, temp_app_config: AppConfig) -> ProjectPaths:
    paths = ProjectPaths.from_config(tmp_path, temp_app_config)
    paths.ensure()
    return paths


@pytest.fixture
def test_logger() -> logging.Logger:
    logger = logging.getLogger("decomp_clarifier_test")
    logger.handlers.clear()
    logger.setLevel("INFO")
    return logger


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
