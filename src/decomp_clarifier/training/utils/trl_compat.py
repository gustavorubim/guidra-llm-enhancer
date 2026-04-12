from __future__ import annotations

from typing import Any


def normalize_optional_flag(value: Any) -> bool:
    if isinstance(value, tuple):
        return bool(value[0])
    return bool(value)


def patch_trl_optional_availability() -> None:
    import trl.import_utils as trl_import_utils  # type: ignore[import-not-found]

    for name in dir(trl_import_utils):
        if not (name.startswith("_") and name.endswith("_available")):
            continue
        value = getattr(trl_import_utils, name)
        if isinstance(value, tuple):
            setattr(trl_import_utils, name, normalize_optional_flag(value))
