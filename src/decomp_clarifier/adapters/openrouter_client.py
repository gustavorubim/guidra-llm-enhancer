from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

import httpx

from decomp_clarifier.adapters.filesystem_cache import FilesystemCache


class OpenRouterError(RuntimeError):
    """Raised when the OpenRouter request fails or returns an invalid payload."""


class OpenRouterClient:
    def __init__(
        self,
        api_key: str | None,
        base_url: str,
        cache: FilesystemCache,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.cache = cache
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            timeout=60.0,
            transport=transport,
        )

    def build_payload(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        response_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        strict_schema = _strict_json_schema(response_schema) if response_schema else None
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if strict_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "decomp_clarifier_schema",
                    "strict": True,
                    "schema": strict_schema,
                },
            }
        return payload

    def generate_json(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        response_schema: dict[str, Any] | None = None,
        fallback_models: list[str] | None = None,
        schema_version: str = "v1",
    ) -> dict[str, Any]:
        payload = self.build_payload(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            response_schema=response_schema,
        )
        key = self.cache.key_for_payload(payload, model, schema_version)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        models = [model, *(fallback_models or [])]
        last_error: Exception | None = None
        for selected_model in models:
            request_payload = dict(payload)
            request_payload["model"] = selected_model
            try:
                data = self._post_json(request_payload)
                self.cache.set(key, data)
                return data
            except OpenRouterError as exc:
                if response_schema and _is_schema_rejection(exc):
                    retry_payload = dict(request_payload)
                    retry_payload.pop("response_format", None)
                    try:
                        data = self._post_json(retry_payload)
                        self.cache.set(key, data)
                        return data
                    except Exception as retry_exc:  # pragma: no cover
                        last_error = retry_exc
                        continue
                last_error = exc
            except Exception as exc:  # pragma: no cover - defensive network retry path
                last_error = exc
        raise OpenRouterError(str(last_error) if last_error else "OpenRouter request failed")

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key and self._client.headers.get("Authorization") is None:
            raise OpenRouterError("OPENROUTER_API_KEY is not set")
        response = self._client.post("/chat/completions", json=payload)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = response.text.strip()
            if detail:
                raise OpenRouterError(detail) from exc
            raise
        body = response.json()
        text = _content_text_from_body(body)
        return _json_from_text(text)


def _content_text_from_body(body: dict[str, Any]) -> str:
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise OpenRouterError("Unexpected OpenRouter response shape") from exc

    if isinstance(content, list):
        text = "".join(
            part.get("text", "") for part in content if isinstance(part, dict) and "text" in part
        )
    elif isinstance(content, str):
        text = content
    else:
        raise OpenRouterError("Unsupported OpenRouter content format")
    return text


def _json_from_text(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        json_fragment = _extract_json_object(text)
        if json_fragment is None:
            raise OpenRouterError("OpenRouter response was not valid JSON") from None
        try:
            return json.loads(json_fragment)
        except json.JSONDecodeError as exc:
            raise OpenRouterError("OpenRouter response was not valid JSON") from exc


def _extract_json_object(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def _is_schema_rejection(exc: OpenRouterError) -> bool:
    message = str(exc)
    return "invalid_json_schema" in message or "Invalid schema for response_format" in message


def _strict_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    strict_schema = deepcopy(schema)

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            node_type = node.get("type")
            if node_type == "object" or "properties" in node:
                node.setdefault("additionalProperties", False)
                properties = node.get("properties")
                if isinstance(properties, dict):
                    node["required"] = list(properties.keys())
            for key in ("properties", "$defs", "definitions"):
                if isinstance(node.get(key), dict):
                    for child in node[key].values():
                        visit(child)
            if isinstance(node.get("items"), dict):
                visit(node["items"])
            for key in ("anyOf", "allOf", "oneOf", "prefixItems"):
                if isinstance(node.get(key), list):
                    for child in node[key]:
                        visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(strict_schema)
    return strict_schema
