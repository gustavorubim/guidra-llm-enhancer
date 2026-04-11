from __future__ import annotations

import os
from pathlib import Path

from decomp_clarifier.adapters.subprocess_utils import SubprocessResult, run_subprocess
from decomp_clarifier.settings import GhidraConfig


class GhidraHeadlessAdapter:
    def __init__(self, config: GhidraConfig, root: Path) -> None:
        self.config = config
        self.root = root

    def analyze_headless_path(self) -> Path:
        explicit = self.config.analyze_headless_path or os.getenv(
            "DECOMP_CLARIFIER_GHIDRA_ANALYZE_HEADLESS"
        )
        if explicit:
            return Path(explicit).expanduser()

        install_dir = self.config.install_dir or os.getenv("DECOMP_CLARIFIER_GHIDRA_DIR")
        if install_dir:
            suffix = "analyzeHeadless.bat" if os.name == "nt" else "analyzeHeadless"
            return Path(install_dir).expanduser() / "support" / suffix

        default_name = "analyzeHeadless.bat" if os.name == "nt" else "analyzeHeadless"
        default_candidate = (
            Path.home() / "Downloads" / "ghidra_12.0.4_PUBLIC" / "support" / default_name
        )
        return default_candidate

    def build_command(
        self,
        *,
        binary_path: Path,
        output_dir: Path,
        project_name: str,
    ) -> list[str]:
        analyze = self.analyze_headless_path()
        project_dir = self.root / self.config.project_dir
        script_dir = self.root / self.config.script_dir
        return [
            str(analyze),
            str(project_dir),
            project_name,
            "-import",
            str(binary_path),
            "-scriptPath",
            str(script_dir),
            "-postScript",
            self.config.script_name,
            str(output_dir),
            "-deleteProject",
            "-analysisTimeoutPerFile",
            str(self.config.timeout_seconds),
        ]

    def run(self, *, binary_path: Path, output_dir: Path, project_name: str) -> SubprocessResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        command = self.build_command(
            binary_path=binary_path, output_dir=output_dir, project_name=project_name
        )
        return run_subprocess(command, cwd=self.root, timeout_seconds=self.config.timeout_seconds)
