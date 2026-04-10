from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_training_summary(path: Path, metrics: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return path
