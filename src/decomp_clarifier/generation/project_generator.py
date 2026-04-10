from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from decomp_clarifier.generation.canonicalize import canonicalize_project
from decomp_clarifier.generation.prompt_builder import build_project_generation_prompt
from decomp_clarifier.generation.validators import validate_project
from decomp_clarifier.schemas.generation import GeneratedProject
from decomp_clarifier.settings import GenerationConfig


class ProjectGenerator:
    def __init__(
        self,
        client: Any,
        config: GenerationConfig,
        prompt_template: str,
        project_root: Path,
        manifest_root: Path,
    ) -> None:
        self.client = client
        self.config = config
        self.prompt_template = prompt_template
        self.project_root = project_root
        self.manifest_root = manifest_root
        self.project_root.mkdir(parents=True, exist_ok=True)
        self.manifest_root.mkdir(parents=True, exist_ok=True)

    def generate_many(self, count: int | None = None) -> list[GeneratedProject]:
        return [
            self.generate_one(index=index)
            for index in range(count or self.config.generation.project_count)
        ]

    def generate_one(self, index: int = 0) -> GeneratedProject:
        prompt = build_project_generation_prompt(self.prompt_template, self.config)
        payload = self.client.generate_json(
            model=self.config.model.model_id,
            fallback_models=self.config.model.fallback_models,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.config.model.max_tokens,
            temperature=self.config.model.temperature,
            response_schema=GeneratedProject.model_json_schema(),
            schema_version=f"generation-{index}",
        )
        project = canonicalize_project(GeneratedProject.model_validate(payload))
        validate_project(project, self.config.validation)
        return self.write_project(project)

    def _uniquify_project(self, project: GeneratedProject) -> GeneratedProject:
        base_id = project.project_id
        candidate_id = base_id
        suffix = 1
        while (
            (self.project_root / candidate_id).exists()
            or (self.manifest_root / f"{candidate_id}.json").exists()
        ):
            candidate_id = f"{base_id}_v{suffix}"
            suffix += 1
        if candidate_id == base_id:
            return project
        return project.model_copy(update={"project_id": candidate_id})

    def write_project(self, project: GeneratedProject) -> GeneratedProject:
        project = self._uniquify_project(project)
        destination = self.project_root / project.project_id
        destination.mkdir(parents=True, exist_ok=True)
        for file in project.files:
            output = destination / file.path
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(file.content, encoding="utf-8")
        manifest_path = destination / "project_manifest.json"
        manifest_json = json.dumps(project.model_dump(mode="python"), indent=2, sort_keys=True)
        manifest_path.write_text(manifest_json, encoding="utf-8")
        (self.manifest_root / f"{project.project_id}.json").write_text(
            manifest_json, encoding="utf-8"
        )
        return project

    @staticmethod
    def load_project(path: Path) -> GeneratedProject:
        return GeneratedProject.model_validate_json(path.read_text(encoding="utf-8"))
