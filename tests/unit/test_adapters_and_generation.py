from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import yaml

from decomp_clarifier.adapters.filesystem_cache import FilesystemCache
from decomp_clarifier.adapters.openrouter_client import (
    OpenRouterClient,
    OpenRouterError,
    _content_text_from_body,
    _extract_json_object,
    _is_schema_rejection,
    _json_from_text,
)
from decomp_clarifier.generation.canonicalize import canonicalize_project
from decomp_clarifier.generation.project_generator import ProjectGenerator
from decomp_clarifier.generation.prompt_builder import (
    build_cleanup_prompt,
    build_project_generation_prompt,
)
from decomp_clarifier.generation.validators import ProjectValidationError, validate_project
from decomp_clarifier.schemas.model_io import PromptInput
from decomp_clarifier.settings import GenerationConfig


def test_openrouter_client_parses_and_caches(tmp_path: Path) -> None:
    calls = {"count": 0}
    json_text = (
        '{"project_id":"p","summary":"s","difficulty":"easy","files":[],"tests":[],'
        '"build":{"entrypoints":[],"c_standard":"c11","compiler_family":"clang"},'
        '"semantic_hints":{"project_purpose":"x","function_intents":[]}}'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json_text}}]},
        )

    client = OpenRouterClient(
        api_key="token",
        base_url="https://example.com",
        cache=FilesystemCache(tmp_path / "cache"),
        transport=httpx.MockTransport(handler),
    )
    payload = client.generate_json(
        model="model",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=10,
        temperature=0.1,
        response_schema=None,
        schema_version="v1",
    )
    second = client.generate_json(
        model="model",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=10,
        temperature=0.1,
        response_schema=None,
        schema_version="v1",
    )
    assert payload["project_id"] == "p"
    assert second["project_id"] == "p"
    assert calls["count"] == 1
    strict_schema = client.build_payload(
        model="model",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=10,
        temperature=0.1,
        response_schema={"type": "object", "properties": {"x": {"type": "string"}}},
    )["response_format"]["json_schema"]["schema"]
    assert strict_schema["additionalProperties"] is False
    assert strict_schema["required"] == ["x"]
    assert strict_schema["properties"]["x"]["type"] == "string"


def test_prompt_builder_and_validation(sample_project, repo_root: Path) -> None:
    config = GenerationConfig.model_validate(
        yaml.safe_load((repo_root / "configs/generation/default.yaml").read_text(encoding="utf-8"))
    )
    prompt = build_project_generation_prompt(
        "Topics: {topics}\nWeights: {difficulty_weights}\nValidation: {validation_rules}",
        config,
    )
    assert "string parsing" in prompt
    validate_project(sample_project, config.validation)


def test_prompt_builder_and_project_generator(
    tmp_path: Path, sample_project, repo_root: Path
) -> None:
    generation_config = GenerationConfig.model_validate_json(
        GenerationConfig(
            model={
                "model_id": "test-model",
                "fallback_models": [],
                "temperature": 0.2,
                "max_tokens": 10,
            },
            generation={
                "project_count": 1,
                "difficulty_weights": {"easy": 1.0},
                "topic_weights": {"string parsing": 1.0},
            },
            validation={
                "min_source_files": 1,
                "min_function_count": 3,
                "max_source_files": 8,
                "banned_includes": [],
                "banned_calls": [],
            },
        ).model_dump_json()
    )
    prompt = build_project_generation_prompt(
        "Topics: {topics}\nWeights: {difficulty_weights}\nValidation: {validation_rules}",
        generation_config,
    )
    cleanup_prompt = build_cleanup_prompt(
        "Return JSON.",
        PromptInput(task_type="cleanup", decompiled_code="int x(void){return 1;}", assembly="ret"),
    )
    assert "string parsing" in prompt
    assert "Decompiler:" in cleanup_prompt

    canonical = canonicalize_project(sample_project)
    validate_project(canonical, generation_config.validation)

    class FakeClient:
        def generate_json(self, **_: object):
            return sample_project.model_dump(mode="python")

    generator = ProjectGenerator(
        client=FakeClient(),
        config=generation_config,
        prompt_template=(
            "Topics: {topics}\nWeights: {difficulty_weights}\nValidation: {validation_rules}"
        ),
        project_root=tmp_path / "projects",
        manifest_root=tmp_path / "manifests",
    )
    project = generator.generate_one()
    assert project.project_id == "sample_project"
    assert (tmp_path / "projects" / "sample_project" / "project_manifest.json").exists()


def test_validate_project_rejects_banned_include(sample_project, repo_root: Path) -> None:
    bad_project = sample_project.model_copy(
        update={
            "files": [
                sample_project.files[0].model_copy(
                    update={"content": "#include <windows.h>\nint main(void){return 0;}\n"}
                )
            ]
        }
    )
    config = GenerationConfig(
        model={"model_id": "test-model"},
        validation={"banned_includes": ["windows.h"], "banned_calls": []},
    )
    with pytest.raises(ProjectValidationError):
        validate_project(bad_project, config.validation)


def test_openrouter_error_paths_and_prompt_helpers(tmp_path: Path) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "not-json"}}]})

    client = OpenRouterClient(
        api_key="token",
        base_url="https://example.com",
        cache=FilesystemCache(tmp_path / "cache"),
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(OpenRouterError):
        client.generate_json(
            model="model",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=10,
            temperature=0.1,
            response_schema=None,
        )


def test_openrouter_schema_rejection_retries_without_response_format(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = yaml.safe_load(request.content.decode("utf-8"))
        calls.append(payload)
        if "response_format" in payload:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": (
                            "Invalid schema for response_format 'decomp_clarifier_schema': "
                            "invalid_json_schema"
                        )
                    }
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                'prefix {"project_id":"p","summary":"s","difficulty":"easy",'
                                '"files":[],"tests":[],"build":{"entrypoints":[],"c_standard":"c11",'
                                '"compiler_family":"clang"},"semantic_hints":{"project_purpose":"x",'
                                '"function_intents":[]}} suffix'
                            )
                        }
                    }
                ]
            },
        )

    client = OpenRouterClient(
        api_key="token",
        base_url="https://example.com",
        cache=FilesystemCache(tmp_path / "cache"),
        transport=httpx.MockTransport(handler),
    )
    payload = client.generate_json(
        model="model",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=10,
        temperature=0.1,
        response_schema={"type": "object", "properties": {"x": {"type": "string"}}},
        schema_version="retry-case",
    )
    assert payload["project_id"] == "p"
    assert len(calls) == 2
    assert "response_format" in calls[0]
    assert "response_format" not in calls[1]


def test_openrouter_helper_functions() -> None:
    assert _extract_json_object('prefix {"a": 1} suffix') == '{"a": 1}'
    assert _json_from_text('prefix {"a": 1} suffix') == {"a": 1}
    assert (
        _content_text_from_body({"choices": [{"message": {"content": [{"text": "hello"}]}}]})
        == "hello"
    )
    assert _is_schema_rejection(OpenRouterError("invalid_json_schema"))
    with pytest.raises(OpenRouterError):
        _content_text_from_body({})
    with pytest.raises(OpenRouterError):
        _content_text_from_body({"choices": [{"message": {"content": 123}}]})
    with pytest.raises(OpenRouterError):
        _json_from_text("not json")
