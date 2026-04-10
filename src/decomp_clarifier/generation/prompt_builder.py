from __future__ import annotations

import json
from pathlib import Path

from decomp_clarifier.schemas.model_io import PromptInput
from decomp_clarifier.settings import GenerationConfig


def load_template(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def build_project_generation_prompt(template: str, config: GenerationConfig) -> str:
    return template.format(
        topics=json.dumps(config.generation.topic_weights, indent=2, sort_keys=True),
        difficulty_weights=json.dumps(
            config.generation.difficulty_weights, indent=2, sort_keys=True
        ),
        validation_rules=json.dumps(
            config.validation.model_dump(mode="python"), indent=2, sort_keys=True
        ),
    )


def build_cleanup_prompt(template: str, prompt_input: PromptInput) -> str:
    sections = [
        template.strip(),
        "",
        f"Task type: {prompt_input.task_type}",
        "",
        "Decompiler:",
        f"<code>\n{prompt_input.decompiled_code}\n</code>",
        "",
        "Assembly:",
        f"<asm>\n{prompt_input.assembly}\n</asm>",
        "",
        f"Strings: {json.dumps(prompt_input.strings)}",
        f"Imports: {json.dumps(prompt_input.imports)}",
        f"Callers: {json.dumps(prompt_input.callers)}",
        f"Callees: {json.dumps(prompt_input.callees)}",
    ]
    if prompt_input.semantic_summary:
        sections.extend(["", f"Semantic summary hint: {prompt_input.semantic_summary}"])
    return "\n".join(sections).strip()
