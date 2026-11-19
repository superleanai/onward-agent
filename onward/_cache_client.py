"""Native requests with explicit control over what can be written to the prompt cache."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Literal

import httpx
from agentknit.inference_db import load_entry, resolve_key

Provider = Literal["anthropic", "openai"]
_ENDPOINTS: dict[str, tuple[Provider, str]] = {
    "https://api.anthropic.com/v1": ("anthropic", "https://api.anthropic.com/v1/messages"),
    "https://api.anthropic.com/v1/messages": ("anthropic", "https://api.anthropic.com/v1/messages"),
    "https://api.openai.com/v1": ("openai", "https://api.openai.com/v1/responses"),
    "https://api.openai.com/v1/responses": ("openai", "https://api.openai.com/v1/responses"),
}


def create_cache_client(spec: dict[str, Any]) -> CacheClient:
    """Resolve the endpoint and key from the configured inference-db entry."""
    endpoint = str(spec.get("endpoint", "")).rstrip("/")
    if endpoint not in _ENDPOINTS:
        raise ValueError("--cache-stable-prefix requires a direct Anthropic Messages or OpenAI Responses endpoint")
    provider, url = _ENDPOINTS[endpoint]
    key = resolve_key(load_entry(str(spec["inference_db"])))
    if not key:
        raise ValueError(f"--cache-stable-prefix requires an API key for {provider}")
    headers = dict(spec.get("extra_headers") or {})
    if provider == "anthropic":
        headers.update({"x-api-key": key, "anthropic-version": "2023-06-01"})
    else:
        headers["Authorization"] = f"Bearer {key}"
    return CacheClient(provider, url, httpx.Client(headers=headers, timeout=spec.get("request_timeout", 120)))


class CacheClient:
    """Adapt native text APIs to the completion interface used by Onward."""

    def __init__(self, provider: Provider, url: str, http: httpx.Client) -> None:
        self.provider = provider
        self.url = url
        self.http = http
        self.chat = SimpleNamespace(completions=self)

    def create(self, *, model: str, messages: list[dict[str, str]], cache_prefix: int,
               previous_cache_prefix: int = 0, extra_body: dict[str, Any] | None = None,
               **params: Any) -> SimpleNamespace:
        """Treat the first cache_prefix messages as stable, or write nothing when it is zero."""
        if not 0 <= cache_prefix <= len(messages):
            raise ValueError("cache prefix is outside the prompt")
        body = {**params, **(extra_body or {})}
        schema_format = body.pop("response_format", None)
        # Cache and input layout belong to this adapter. Inherited conversation or tool
        # fields could introduce cached content outside the selected prefix.
        for field in ("cache_control", "prompt_cache_options", "tools", "additional_tools",
                      "previous_response_id", "conversation", "prompt", "input", "system", "messages"):
            body.pop(field, None)
        body.update(model=model, stream=False)
        if self.provider == "anthropic":
            return self._anthropic(body, messages, cache_prefix, schema_format)
        # Responses does not cache at assistant output blocks. Keep the previous
        # eligible boundary so a growing prefix can reuse its earlier cache entry.
        latest = cache_prefix
        while latest and messages[latest - 1]["role"] == "assistant":
            latest -= 1
        breakpoints = sorted({position for position in (previous_cache_prefix, latest) if 0 < position <= latest})
        return self._openai(body, messages, breakpoints, schema_format)

    def _anthropic(self, body: dict[str, Any], messages: list[dict[str, str]], cache_prefix: int,
                   schema_format: dict[str, Any] | None) -> SimpleNamespace:
        system = []
        conversation = []
        for position, message in enumerate(messages, 1):
            block: dict[str, Any] = {"type": "text", "text": message["content"]}
            if position == cache_prefix:
                block["cache_control"] = {"type": "ephemeral"}
            if message["role"] == "system":
                system.append(block)
            else:
                conversation.append({"role": message["role"], "content": [block]})
        body.update(system=system, messages=conversation)
        body.setdefault("max_tokens", 8192)
        if schema_format:
            body["output_config"] = {**body.get("output_config", {}), "format": {
                "type": "json_schema", "schema": schema_format["json_schema"]["schema"],
            }}
        data = self._post(body)
        output = "".join(block["text"] for block in data["content"] if block["type"] == "text")
        usage = data["usage"]
        read = usage.get("cache_read_input_tokens", 0)
        written = usage.get("cache_creation_input_tokens", 0)
        return _completion(output, data.get("stop_reason"), usage["input_tokens"] + read + written,
                           usage["output_tokens"], read, written)

    def _openai(self, body: dict[str, Any], messages: list[dict[str, str]], breakpoints: list[int],
                schema_format: dict[str, Any] | None) -> SimpleNamespace:
        input_items = []
        for position, message in enumerate(messages, 1):
            role = "developer" if message["role"] == "system" else message["role"]
            block: dict[str, Any]
            if role == "assistant":
                block = {"type": "output_text", "text": message["content"], "annotations": []}
            else:
                block = {"type": "input_text", "text": message["content"]}
            if position in breakpoints:
                block["prompt_cache_breakpoint"] = {"mode": "explicit"}
            input_items.append({"role": role, "content": [block]})
        body.update(input=input_items, prompt_cache_options={"mode": "explicit"}, store=False)
        if "max_tokens" in body:
            body["max_output_tokens"] = body.pop("max_tokens")
        if schema_format:
            body["text"] = {**body.get("text", {}), "format": {
                "type": "json_schema", **schema_format["json_schema"],
            }}
        data = self._post(body)
        output = "".join(block["text"] for item in data["output"] if item["type"] == "message"
                         for block in item["content"] if block["type"] == "output_text")
        usage = data["usage"]
        details = usage.get("input_tokens_details") or {}
        reason = (data.get("incomplete_details") or {}).get("reason", data.get("status"))
        response = _completion(output, reason, usage["input_tokens"], usage["output_tokens"],
                               details.get("cached_tokens", 0), details.get("cache_write_tokens"))
        response.cache_breakpoints = breakpoints
        return response

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        response = self.http.post(self.url, json=body)
        if response.is_error:
            raise RuntimeError(f"{self.provider} request failed with HTTP {response.status_code}: {response.text}")
        data: dict[str, Any] = response.json()
        return data


def _completion(text: str, reason: str | None, input_tokens: int, output_tokens: int,
                cached_tokens: int, cache_write_tokens: int | None) -> SimpleNamespace:
    choice = SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=reason)
    usage = SimpleNamespace(prompt_tokens=input_tokens, completion_tokens=output_tokens,
                            cached_tokens=cached_tokens, cache_write_tokens=cache_write_tokens)
    return SimpleNamespace(choices=[choice], usage=usage)
