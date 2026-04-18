from __future__ import annotations

import json
from pathlib import Path

from decomp_clarifier.c_source import extract_called_functions
from decomp_clarifier.dataset.prompt_formatter import (
    completion_messages,
    format_prompt,
    format_rl_prompt,
    prompt_messages,
)
from decomp_clarifier.schemas.dataset import (
    DatasetManifest,
    FunctionDatasetSample,
    PackedRLRecord,
    PackedSFTRecord,
)


def _allowed_callees(sample: FunctionDatasetSample) -> list[str]:
    # Ghidra import/callee recovery is sparse on some samples. Fall back to calls
    # seen in the cleaned target so reward shaping does not punish target-valid calls.
    # Include the source function name as well so recursive/self references are not
    # misclassified as hallucinated calls by the GRPO reward stack.
    return list(
        dict.fromkeys(
            [
                *sample.callees,
                *extract_called_functions(sample.target_clean_code),
                sample.source_function_name,
            ]
        )
    )


def pack_sft_records(samples: list[FunctionDatasetSample]) -> list[PackedSFTRecord]:
    records: list[PackedSFTRecord] = []
    for sample in samples:
        prompt = format_prompt(sample)
        response_json = json.dumps(
            {
                "summary": sample.semantic_summary,
                "confidence": 1.0,
                "renamings": sample.rename_map_target,
                "cleaned_c": sample.target_clean_code,
            },
            sort_keys=True,
        )
        records.append(
            PackedSFTRecord(
                sample_id=sample.sample_id,
                task_type=sample.task_type,
                prompt=prompt,
                response_json=response_json,
                prompt_messages=prompt_messages(prompt),
                completion_messages=completion_messages(response_json),
            )
        )
    return records


def pack_rl_records(samples: list[FunctionDatasetSample]) -> list[PackedRLRecord]:
    records: list[PackedRLRecord] = []
    for sample in samples:
        prompt = format_rl_prompt(sample)
        records.append(
            PackedRLRecord(
                sample_id=sample.sample_id,
                task_type=sample.task_type,
                prompt=prompt,
                prompt_messages=prompt_messages(prompt),
                source_function_name=sample.source_function_name,
                raw_code=sample.ghidra_decompiled_code,
                compile_reference_source=sample.compile_reference_source or sample.source_code,
                target_clean_code=sample.target_clean_code,
                target_renamings=json.dumps(sample.rename_map_target, sort_keys=True),
                allowed_imports=json.dumps(sample.imports),
                allowed_callees=json.dumps(_allowed_callees(sample)),
                compiler_executable=sample.compiler_executable,
                tests_ref=sample.tests_ref,
            )
        )
    return records


def write_jsonl_records(path: Path, records: list[PackedSFTRecord]) -> DatasetManifest:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(record.model_dump_json() for record in records) + ("\n" if records else ""),
        encoding="utf-8",
    )
    task_counts: dict[str, int] = {}
    for record in records:
        task_counts[record.task_type] = task_counts.get(record.task_type, 0) + 1
    manifest = DatasetManifest(
        record_count=len(records),
        split_counts={},
        task_counts=task_counts,
        output_path=str(path),
    )
    return manifest
