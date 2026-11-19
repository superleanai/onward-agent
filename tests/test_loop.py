from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from agentknit import default_tool_spec

from onward._loop import run
from onward._results import ResultStore
from onward._summary import ContextSummarizer


class Scripted:
    """Fake OpenAI client replaying canned replies and recording the messages it got."""

    def __init__(self, replies: list[str], usage: dict[str, int] | None = None) -> None:
        self.replies = list(replies)
        self.usage = SimpleNamespace(**usage) if usage is not None else None
        self.calls: list[list[dict[str, str]]] = []
        self.kwargs: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, model: str, messages: list[dict[str, str]], **kw: Any) -> Any:
        self.calls.append(messages)
        self.kwargs.append(kw)
        msg = SimpleNamespace(content=self.replies.pop(0))
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=self.usage)


def _r(label: str, tool_calls: list[dict[str, Any]] | None = None, done: str | None = None) -> str:
    calls = []
    for n, call in enumerate(tool_calls or []):
        call_type = "retrieve" if "retrieve" in call else "file" if "file" in call else "tool"
        call = {"type": call_type, **call}
        if "retrieve" not in call:
            call = {"id": f"call-{label}-{n}", **call}
        calls.append(call)
    return json.dumps({"tool_calls": calls, "done": done})


@pytest.fixture
def tools() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    specs, disp = default_tool_spec()
    return [s for s in specs if s["function"]["name"] != "read_file"], disp


def _run(client: Scripted, tools: Any, tmp_path: Path, **kw: Any) -> Any:
    return run("task", client, "m", tools[0], tools[1], log_path=tmp_path / "log.jsonl", **kw)


def test_calls_are_compacted_immediately_and_results_after_one_request(tools: Any, tmp_path: Path) -> None:
    client = Scripted([
        _r("a", [{"tool": "exec_shell", "args": {"command": "echo first-output"}}]),
        _r("b", [{"tool": "exec_shell", "args": {"command": "echo second-output"}}]),
        _r("final", done="ok")])
    result = _run(client, tools, tmp_path)
    assert result.final == "ok" and result.turns == 3
    turn_two = [m["content"] for m in client.calls[1]]
    assert turn_two[2] == "[tool calls: 1]"
    assert "first-output" in turn_two[3]
    assert all("echo first-output" not in content for content in turn_two)
    contents = [m["content"] for m in client.calls[2]]
    assert contents[1:5] == ["task", "[tool calls: 1]", '[result "call-a-0"]', "[tool calls: 1]"]
    assert "second-output" in contents[5]
    assert result.items[-2].content == '[result "call-b-0"]'
    assert len(result.items) == 6


def test_cache_prefix_breaks_at_each_replaced_batch(tools: Any, tmp_path: Path) -> None:
    client = Scripted([
        _r("first", [{"tool": "exec_shell", "args": {"command": "echo a"}}]),
        _r("second", [{"tool": "exec_shell", "args": {"command": "echo b"}}]),
        _r("third"),
        _r("fourth", done="ok")])
    _run(client, tools, tmp_path, trace_context=True)
    records = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert [record["messages"] for record in records] == client.calls
    assert client.calls[1][:-3] == client.calls[0][:-1]
    assert client.calls[2][:3] == client.calls[1][:3]
    assert client.calls[2][3] != client.calls[1][3]
    assert client.calls[3][:5] == client.calls[2][:5]
    assert client.calls[3][5] != client.calls[2][5]


def test_summarizes_calls_and_consumed_results_with_separate_usage(tools: Any, tmp_path: Path) -> None:
    acting = Scripted([
        _r("first", [{"tool": "exec_shell", "args": {"command": "echo original-output"}}]),
        _r("reread", [{"retrieve": "call-first-0"}]),
        _r("final", done="ok"),
    ], {"prompt_tokens": 20, "completion_tokens": 10})
    summary_client = Scripted([
        '{"1":"Ran echo."}',
        '{"2":"Requested earlier result.","1":"Echo printed original-output."}',
        '{"1":"Read the saved output."}',
    ], {"prompt_tokens": 900, "completion_tokens": 80})
    result = _run(acting, tools, tmp_path, summarize=ContextSummarizer(summary_client, "summary-model", {}),
                  trace_context=True)

    assert result.final == "ok" and result.turns == 3
    assert acting.calls[1][2]["content"] == "[tool calls: 1]\nRan echo."
    original_result = acting.calls[1][3]["content"]
    assert "original-output" in original_result
    assert acting.calls[2][3]["content"] == '[result "call-first-0"]\nEcho printed original-output.'
    assert acting.calls[2][4]["content"] == "[tool calls: 1]\nRequested earlier result."
    assert acting.calls[2][5]["content"] == original_result
    assert result.items[-2].content == '[result "call-first-0"]\nRead the saved output.'
    payloads = [json.loads(messages[1]["content"]) for messages in summary_client.calls]
    assert [[item["kind"] for item in payload["entries"].values()] for payload in payloads] == [
        ["model_output"], ["tool_result", "model_output"], ["tool_result"],
    ]
    records = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert [record["usage"]["prompt_tokens"] for record in records] == [20, 20, 20]
    assert [record["model"] for record in records] == ["m", "m", "m"]
    assert [record["messages"] for record in records] == acting.calls


def test_tool_calls_appended_in_listed_order(tools: Any, tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("l1\nl2\nl3\n")
    client = Scripted([
        _r("inspect and run", [{"tool": "exec_shell", "args": {"command": "echo first"}},
                               {"file": str(tmp_path / "f.txt"), "lines": [2, 3]},
                               {"tool": "exec_shell", "args": {"command": "echo third"}}]),
        _r("observed all three", done="ok")])
    _run(client, tools, tmp_path)
    contents = [m["content"] for m in client.calls[1][3:-1]]
    assert "first" in contents[0] and "third" in contents[2]
    assert contents[1].endswith("f.txt lines 2-3\n2\tl2\n3\tl3\n")


def test_file_call_allows_edit(tools: Any, tmp_path: Path) -> None:
    f = tmp_path / "g.py"
    f.write_text("x = 1\n")
    client = Scripted([
        _r("read file", [{"file": str(f)}]),
        _r("saw x = 1", [{"tool": "str_replace",
                           "args": {"path": str(f), "old_str": "1", "new_str": "2"}}]),
        _r("changed x to 2", done="ok")])
    _run(client, tools, tmp_path)
    assert f.read_text() == "x = 2\n"
    assert "str_replace" in client.calls[2][-2]["content"]


def test_protocol_error_is_appended_without_losing_history(tools: Any, tmp_path: Path) -> None:
    malformed = _r("bad", [{"tool": "exec_shell", "args": "bad payload"}])
    client = Scripted([
        _r("before-error", [{"tool": "exec_shell", "args": {"command": "echo prior-output"}}]),
        malformed,
        _r("recovered", done="ok")])
    result = _run(client, tools, tmp_path)
    assert result.final == "ok"
    contents = [m["content"] for m in client.calls[2]]
    assert contents[3:6] == ['[result "call-before-error-0"]', "[invalid reply]",
                             "protocol error: tool call: args must be an object"]
    assert all("bad payload" not in content and "prior-output" not in content for content in contents)


def test_done_ignores_tool_calls(tools: Any, tmp_path: Path) -> None:
    marker = tmp_path / "not-created"
    client = Scripted([_r("finished", [{"tool": "write_file", "args": {
        "path": str(marker), "content": "bad"}}], done="ok")])
    result = _run(client, tools, tmp_path)
    assert result.final == "ok"
    assert not marker.exists()
    assert json.loads(result.items[-1].content) == {"tool_calls": [], "done": "ok"}


def test_log_contains_original_reply_and_new_items(tools: Any, tmp_path: Path) -> None:
    client = Scripted([
        _r("run a", [{"tool": "exec_shell", "args": {"command": "echo a"}}]),
        _r("a worked", done="ok")])
    _run(client, tools, tmp_path)
    records = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert "summary" not in records[0]
    assert "messages" not in records[0]
    assert json.loads(records[0]["reply"])["tool_calls"][0]["args"] == {"command": "echo a"}
    assert records[0]["new_items"][0][0] == "tool_result"
    assert json.loads(records[1]["reply"])["done"] == "ok"


def test_model_ids_label_results_and_retrieve_past_full_results(tools: Any, tmp_path: Path) -> None:
    client = Scripted([
        _r("make initial observations", [
            {"id": "shell-a", "tool": "exec_shell", "args": {"command": "echo a"}},
            {"id": "shell-b", "tool": "exec_shell", "args": {"command": "echo b"}}]),
        _r("wait"),
        _r("fetch old results", [{"retrieve": ["shell-a", "shell-b"]}]),
        _r("retrieved shell-a", done="ok")])
    result = _run(client, tools, tmp_path)
    first = client.calls[1][3]["content"]
    second = client.calls[1][4]["content"]
    assert first.startswith("<shell-a>\nexec_shell\n")
    assert second.startswith("<shell-b>\nexec_shell\n")
    assert [m["content"] for m in client.calls[2][3:5]] == ['[result "shell-a"]', '[result "shell-b"]']
    retrieved = client.calls[3][-2]["content"]
    assert retrieved == f"{first}\n\n{second}"
    assert retrieved.endswith('"returncode":0}')
    assert result.items[-2].content == '[result "shell-a", "shell-b"]'


def test_retrieval_does_not_rerun_tool(tools: Any, tmp_path: Path) -> None:
    marker = tmp_path / "once"
    client = Scripted([
        _r("write once", [{"id": "write-1", "tool": "write_file",
                            "args": {"path": str(marker), "content": "one"}}]),
        _r("retrieve write result", [{"retrieve": "write-1"}]),
        _r("retrieved write result", done="ok")])
    _run(client, tools, tmp_path)
    assert marker.read_text() == "one"
    assert "write_file" in client.calls[2][-2]["content"]


def test_retrieval_recovers_uncapped_past_result(tools: Any, tmp_path: Path) -> None:
    client = Scripted([
        _r("produce long output", [{"id": "long-output", "tool": "exec_shell",
                                     "args": {"command": "python3 -c 'print(12345)'"}}]),
        _r("retrieve long output", [{"retrieve": "long-output"}]),
        _r("retrieved full output", done="ok")])
    _run(client, tools, tmp_path, max_item_chars=50)
    first = client.calls[1][3]["content"]
    retrieved = client.calls[2][-2]["content"]
    assert "characters truncated by the host" in first
    assert "characters truncated by the host" not in retrieved
    assert "12345" in retrieved


def test_unknown_or_duplicate_ids_become_result_errors(tools: Any, tmp_path: Path) -> None:
    client = Scripted([
        _r("missing id", [{"retrieve": "missing"}]),
        _r("duplicate id", [{"id": "dup", "tool": "exec_shell", "args": {"command": "echo one"}},
                             {"id": "dup", "tool": "exec_shell", "args": {"command": "echo two"}}]),
        _r("finished", done="ok")])
    result = _run(client, tools, tmp_path)
    assert "unknown tool-call id 'missing'" in client.calls[1][3]["content"]
    assert client.calls[2][3]["content"] == '[error "missing"]'
    assert "not run: duplicate tool-call id 'dup'" in client.calls[2][-2]["content"]
    assert result.items[-2].content == '[error "dup"]'
    records = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert records[1]["new_items"][0][2] == "dup"


def test_max_turns_returns_without_final(tools: Any, tmp_path: Path) -> None:
    client = Scripted([
        _r("first", [{"tool": "exec_shell", "args": {"command": "echo pending-output"}}]),
        _r("second", done="ok")])
    store = ResultStore(tmp_path / "results")
    first = _run(client, tools, tmp_path, max_turns=1, results=store)
    assert first.final is None and first.turns == 1
    second = run("follow-up", client, "m", tools[0], tools[1], results=store, items=first.items, start_turn=2)
    assert second.final == "ok" and second.turns == 2
    assert "pending-output" in client.calls[1][3]["content"]
    assert client.calls[1][4]["content"] == "follow-up"
    assert second.items[2].content == '[result "call-first-0"]'
