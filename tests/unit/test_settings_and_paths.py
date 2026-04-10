from __future__ import annotations

import os
from pathlib import Path

from decomp_clarifier.adapters.filesystem_cache import FilesystemCache
from decomp_clarifier.paths import ProjectPaths
from decomp_clarifier.settings import (
    deep_merge,
    dump_yaml,
    load_app_config,
    load_dotenv,
    load_ghidra_config,
)


def test_deep_merge_and_dump_yaml(tmp_path: Path, temp_app_config) -> None:
    merged = deep_merge({"a": {"b": 1}, "x": 1}, {"a": {"c": 2}, "x": 3})
    assert merged == {"a": {"b": 1, "c": 2}, "x": 3}

    path = tmp_path / "config.yaml"
    dump_yaml(path, temp_app_config)
    assert "generated_projects_dir" in path.read_text(encoding="utf-8")


def test_load_configs_and_paths(repo_root: Path, monkeypatch) -> None:
    monkeypatch.setenv("DECOMP_CLARIFIER_GHIDRA_DIR", "/tmp/ghidra")
    app_config = load_app_config(repo_root)
    ghidra_config = load_ghidra_config(repo_root)
    assert app_config.paths.generated_projects_dir == "data/raw/generated_projects"
    assert ghidra_config.install_dir == "/tmp/ghidra"

    root = ProjectPaths.discover(repo_root)
    assert root == repo_root


def test_project_paths_and_cache_roundtrip(tmp_path: Path, temp_app_config) -> None:
    paths = ProjectPaths.from_config(tmp_path, temp_app_config)
    paths.ensure()
    run_dir = paths.run_dir("run-1")
    log_file = paths.log_file("run-1")
    assert run_dir.exists()
    assert log_file.parent.exists()

    cache = FilesystemCache(tmp_path / "cache")
    key = cache.key_for_payload({"a": 1}, "model", "v1")
    cache.set(key, {"ok": True})
    assert cache.get(key) == {"ok": True}


def test_load_dotenv_aliases(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / ".env").write_text("open_router='secret-value'\n", encoding="utf-8")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    load_dotenv(tmp_path)
    assert os.getenv("OPENROUTER_API_KEY") == "secret-value"
