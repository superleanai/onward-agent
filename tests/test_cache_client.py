from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from onward.__main__ import main
from onward._cache_client import create_cache_client


@pytest.mark.parametrize("acting_provider,summary_provider", [("anthropic", "openai"), ("openai", "anthropic")])
def test_cli_cache_writes_stop_before_raw_results_and_skip_summary_inputs(
    acting_provider: str, summary_provider: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    db = tmp_path / "endpoints.toml"
    db.write_text("""
[endpoints.anthropic]
url = "https://api.anthropic.com/v1"
key = [{ source = "env", var = "ONWARD_TEST_ANTHROPIC_KEY" }]

[endpoints.openai]
url = "https://api.openai.com/v1"
key = [{ source = "env", var = "ONWARD_TEST_OPENAI_KEY" }]

[endpoints.anthropic.params.extra_body]
cache_control = { type = "ephemeral" }

[endpoints.openai.params.extra_body]
prompt_cache_options = { mode = "implicit" }
""")
    monkeypatch.setenv("INFERENCE_DB", str(db))
    monkeypatch.setenv("ONWARD_TEST_ANTHROPIC_KEY", "anthropic-fixture")
    monkeypatch.setenv("ONWARD_TEST_OPENAI_KEY", "openai-fixture")

    act = iter([
        {"tool_calls": [{"type": "tool", "tool": "exec_shell", "args": {"command": "echo raw-observation"},
                         "id": "result-a"}], "done": None},
        {"tool_calls": [{"type": "retrieve", "retrieve": "result-a"}], "done": None},
        {"tool_calls": [], "done": "ok"},
    ])
    summarize = iter([{"1": "Ran a command."}, {"1": "Command succeeded.", "2": "Requested result-a."},
                      {"1": "Read the stored result."}])
    requests: dict[str, list[dict[str, Any]]] = {"acting": [], "summary": []}

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        role = "acting" if body["model"] == "acting-model" else "summary"
        requests[role].append(body)
        provider = acting_provider if role == "acting" else summary_provider
        content = json.dumps(next(act if role == "acting" else summarize))
        if provider == "anthropic":
            assert str(request.url) == "https://api.anthropic.com/v1/messages"
            assert request.headers["x-api-key"] == "anthropic-fixture"
            return httpx.Response(200, json={
                "content": [{"type": "text", "text": content}], "stop_reason": "end_turn",
                "usage": {"input_tokens": 12, "output_tokens": 2,
                          "cache_read_input_tokens": 5, "cache_creation_input_tokens": 3},
            })
        assert str(request.url) == "https://api.openai.com/v1/responses"
        assert request.headers["authorization"] == "Bearer openai-fixture"
        for message in body["input"]:
            for block in message["content"]:
                if message["role"] == "assistant":
                    assert block["type"] == "output_text"
                    assert block["annotations"] == []
                    assert "prompt_cache_breakpoint" not in block
                else:
                    assert block["type"] == "input_text"
        return httpx.Response(200, json={
            "output": [{"type": "message", "content": [{"type": "output_text", "text": content}]}],
            "status": "completed", "usage": {"input_tokens": 20, "output_tokens": 2,
                                             "input_tokens_details": {"cached_tokens": 5, "cache_write_tokens": 3}},
        })

    original_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original_client(
        transport=httpx.MockTransport(respond), **kwargs))
    log_dir = tmp_path / "runs"
    assert main(["--inference-db", acting_provider, "--model", "acting-model",
                 "--summary-inference-db", summary_provider, "--summary-model", "summary-model",
                 "--cache-stable-prefix", "--trace-context", "--log-dir", str(log_dir), "task"]) == 0

    def blocks(body: dict[str, Any], provider: str) -> list[dict[str, Any]]:
        assert "cache_control" not in body
        if provider == "anthropic":
            assert "prompt_cache_options" not in body
            return body["system"] + [block for message in body["messages"] for block in message["content"]]
        assert body["prompt_cache_options"] == {"mode": "explicit"}
        return [block for message in body["input"] for block in message["content"]]

    def marked(entries: list[dict[str, Any]]) -> list[int]:
        return [index for index, block in enumerate(entries, 1)
                if "cache_control" in block or "prompt_cache_breakpoint" in block]

    acting = [blocks(body, acting_provider) for body in requests["acting"]]
    expected = [[2], [3], [5]] if acting_provider == "anthropic" else [[2], [2], [2, 4]]
    assert [marked(entries) for entries in acting] == expected
    assert "raw-observation" in acting[1][3]["text"]
    assert acting[2][3]["text"] == '[result "result-a"]\nCommand succeeded.'
    assert "raw-observation" in acting[2][5]["text"]
    assert all(entries[-1]["text"].startswith("[context]") for entries in acting)
    summaries = [blocks(body, summary_provider) for body in requests["summary"]]
    assert len(summaries) == 3 and all(not marked(entries) for entries in summaries)
    assert "raw-observation" in summaries[1][1]["text"]
    format_location = "output_config" if summary_provider == "anthropic" else "text"
    assert requests["summary"][1][format_location]["format"]["schema"]["required"] == ["1", "2"]

    records = [json.loads(line) for line in next(log_dir.glob("*.jsonl")).read_text().splitlines()]
    assert [record["cache_prefix_messages"] for record in records] == [indices[-1] for indices in expected]
    assert [record.get("cache_breakpoints_messages") for record in records] == [
        indices if len(indices) > 1 else None for indices in expected
    ]
    assert [record["usage"] for record in records] == [
        {"prompt_tokens": 20, "completion_tokens": 2, "cached_tokens": 5, "cache_write_tokens": 3},
    ] * 3


def test_rejects_transports_that_cannot_enforce_the_cache_boundary() -> None:
    with pytest.raises(ValueError, match="requires a direct Anthropic Messages or OpenAI Responses endpoint"):
        create_cache_client({"endpoint": "run:///codex"})
