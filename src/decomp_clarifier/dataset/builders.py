from __future__ import annotations

import hashlib
import json
from pathlib import Path

from decomp_clarifier.dataset.splitters import split_project_ids
from decomp_clarifier.dataset.transforms import derive_rename_map, normalize_source_for_target
from decomp_clarifier.ghidra_export.aligner import align_functions
from decomp_clarifier.ghidra_export.parse_exports import ParsedGhidraProject
from decomp_clarifier.schemas.compiler import CompileManifest
from decomp_clarifier.schemas.dataset import FunctionDatasetSample
from decomp_clarifier.schemas.generation import GeneratedProject
from decomp_clarifier.settings import DatasetConfig


def build_function_dataset(
    *,
    projects: list[GeneratedProject],
    compile_manifests: list[CompileManifest],
    parsed_exports: list[ParsedGhidraProject],
    config: DatasetConfig,
    output_dir: Path | None = None,
) -> list[FunctionDatasetSample]:
    projects_by_id = {project.project_id: project for project in projects}
    manifests_by_id = {manifest.project_id: manifest for manifest in compile_manifests}
    split_map = split_project_ids(
        sorted(projects_by_id),
        seed=config.dataset.seed,
        train_ratio=config.dataset.train_ratio,
        val_ratio=config.dataset.val_ratio,
        test_ratio=config.dataset.test_ratio,
    )
    samples: list[FunctionDatasetSample] = []
    task_types = list(config.dataset.task_mix) or ["full_clarify"]
    for parsed_project in parsed_exports:
        project = projects_by_id[parsed_project.manifest.project_id]
        compile_manifest = manifests_by_id[project.project_id]
        source_files = {file.path: file.content for file in project.files}
        intent_map = {
            item.function_name: item.intent for item in project.semantic_hints.function_intents
        }
        for aligned in align_functions(project, parsed_project):
            cleaned_source = normalize_source_for_target(aligned.source.code)
            for task_type in task_types:
                sample = FunctionDatasetSample(
                    sample_id=hashlib.sha256(
                        f"{project.project_id}:{aligned.source.name}:{task_type}:{compile_manifest.opt_level}:{compile_manifest.compiler_family}".encode()
                    ).hexdigest()[:16],
                    project_id=project.project_id,
                    split=split_map[project.project_id],
                    task_type=task_type,
                    host_os=compile_manifest.host_os,
                    compiler=compile_manifest.compiler_family,
                    opt_level=compile_manifest.opt_level,
                    binary_format=compile_manifest.binary_format,
                    source_function_name=aligned.source.name,
                    source_code=cleaned_source,
                    compile_reference_source=source_files.get(
                        aligned.source.file_path, aligned.source.code
                    ),
                    target_clean_code=cleaned_source,
                    ghidra_function_name=aligned.ghidra.ghidra_function_name,
                    ghidra_decompiled_code=aligned.ghidra.decompiled_text,
                    assembly=aligned.ghidra.disassembly_text,
                    strings=aligned.ghidra.strings,
                    imports=aligned.ghidra.imports,
                    callers=aligned.ghidra.callers,
                    callees=aligned.ghidra.callees,
                    semantic_summary=intent_map.get(
                        aligned.source.name, project.semantic_hints.project_purpose
                    ),
                    rename_map_target=derive_rename_map(
                        cleaned_source, aligned.ghidra.decompiled_text
                    ),
                    tests_ref=f"{project.project_id}/project_manifest.json",
                    difficulty=project.difficulty,
                )
                samples.append(sample)
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "function_dataset.jsonl").write_text(
            "\n".join(sample.model_dump_json() for sample in samples) + ("\n" if samples else ""),
            encoding="utf-8",
        )
        (output_dir / "dataset_manifest.json").write_text(
            json.dumps(
                {
                    "record_count": len(samples),
                    "project_count": len(projects),
                    "split_counts": {
                        split: sum(1 for sample in samples if sample.split == split)
                        for split in {"train", "val", "test"}
                    },
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    return samples
