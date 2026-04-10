from __future__ import annotations

from importlib import metadata

VALIDATED_VERSIONS = {
    "unsloth": "2025.",
    "trl": "0.17.",
    "transformers": "4.51.",
    "datasets": "3.",
    "accelerate": "1.",
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
