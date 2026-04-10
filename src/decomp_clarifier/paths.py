from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from decomp_clarifier.settings import AppConfig


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    generated_projects_dir: Path
    manifests_dir: Path
    binaries_dir: Path
    ghidra_exports_dir: Path
    aligned_projects_dir: Path
    aligned_functions_dir: Path
    processed_sft_dir: Path
    processed_rl_dir: Path
    processed_eval_dir: Path
    reports_dir: Path
    runs_dir: Path
    logs_dir: Path

    @classmethod
    def discover(cls, start: Path | None = None) -> Path:
        current = (start or Path.cwd()).resolve()
        for candidate in (current, *current.parents):
            if (candidate / "pyproject.toml").exists() or (candidate / "SPEC.md").exists():
                return candidate
        return current

    @classmethod
    def from_config(cls, root: Path, config: AppConfig) -> ProjectPaths:
        path_config = config.paths
        return cls(
            root=root,
            generated_projects_dir=root / path_config.generated_projects_dir,
            manifests_dir=root / path_config.manifests_dir,
            binaries_dir=root / path_config.binaries_dir,
            ghidra_exports_dir=root / path_config.ghidra_exports_dir,
            aligned_projects_dir=root / path_config.aligned_projects_dir,
            aligned_functions_dir=root / path_config.aligned_functions_dir,
            processed_sft_dir=root / path_config.processed_sft_dir,
            processed_rl_dir=root / path_config.processed_rl_dir,
            processed_eval_dir=root / path_config.processed_eval_dir,
            reports_dir=root / path_config.reports_dir,
            runs_dir=root / path_config.runs_dir,
            logs_dir=root / path_config.logs_dir,
        )

    def ensure(self) -> None:
        for directory in (
            self.generated_projects_dir,
            self.manifests_dir,
            self.binaries_dir,
            self.ghidra_exports_dir,
            self.aligned_projects_dir,
            self.aligned_functions_dir,
            self.processed_sft_dir,
            self.processed_rl_dir,
            self.processed_eval_dir,
            self.reports_dir,
            self.runs_dir,
            self.logs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def run_dir(self, run_id: str) -> Path:
        path = self.runs_dir / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def log_file(self, run_id: str) -> Path:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        return self.logs_dir / f"{run_id}.log"

    def resolve(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path
