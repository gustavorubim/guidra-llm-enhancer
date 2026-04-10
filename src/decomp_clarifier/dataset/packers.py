from __future__ import annotations

import json
from pathlib import Path

from decomp_clarifier.dataset.prompt_formatter import format_prompt
from decomp_clarifier.schemas.dataset import DatasetManifest, FunctionDatasetSample, PackedSFTRecord


def pack_sft_records(samples: list[FunctionDatasetSample]) -> list[PackedSFTRecord]:
    return [
        PackedSFTRecord(
            sample_id=sample.sample_id,
            task_type=sample.task_type,
            prompt=format_prompt(sample),
            response_json=json.dumps(
                {
                    "summary": sample.semantic_summary,
                    "confidence": 1.0,
                    "renamings": sample.rename_map_target,
                    "cleaned_c": sample.target_clean_code,
                },
                sort_keys=True,
            ),
        )
        for sample in samples
    ]


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
