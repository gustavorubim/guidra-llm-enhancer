from __future__ import annotations

from importlib import metadata

VALIDATED_VERSIONS = {
    "unsloth": "2026.",
    "trl": "0.24.",
    "transformers": "5.",
    "datasets": "4.",
    "accelerate": "1.",
    "tensorboard": "2.19.",
    "matplotlib": "3.10.",
}


def collect_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in VALIDATED_VERSIONS:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def validate_version_lock() -> dict[str, str | None]:
    versions = collect_versions()
    mismatches = [
        package
        for package, prefix in VALIDATED_VERSIONS.items()
        if versions[package] is None or not versions[package].startswith(prefix)
    ]
    if mismatches:
        mismatch_text = ", ".join(mismatches)
        raise RuntimeError(
            f"training environment does not match validated versions: {mismatch_text}"
        )
    return versions
