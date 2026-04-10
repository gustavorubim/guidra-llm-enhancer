from __future__ import annotations

import random


def split_project_ids(
    project_ids: list[str],
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, str]:
    if round(train_ratio + val_ratio + test_ratio, 6) != 1.0:
        raise ValueError("split ratios must sum to 1.0")
    ordered = list(project_ids)
    random.Random(seed).shuffle(ordered)
    train_cut = round(len(ordered) * train_ratio)
    val_cut = train_cut + round(len(ordered) * val_ratio)
    mapping: dict[str, str] = {}
    for index, project_id in enumerate(ordered):
        if index < train_cut:
            mapping[project_id] = "train"
        elif index < val_cut:
            mapping[project_id] = "val"
        else:
            mapping[project_id] = "test"
    return mapping
