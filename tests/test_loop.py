from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from agentknit import default_tool_spec

from selfctx._loop import run


class Scripted:
    """Fake OpenAI client replaying canned replies and recording the messages it got."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[list[dict[str, str]]] = []
        self.kwargs: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, model: str, messages: list[dict[str, str]], **kw: Any) -> Any:
        self.calls.append(messages)
        self.kwargs.append(kw)
        msg = SimpleNamespace(content=self.replies.pop(0))
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)


def _r(summary: str, tool_calls: list[dict[str, Any]] | None = None, done: str | None = None) -> str:
    calls = []
    for n, call in enumerate(tool_calls or []):
        if "retrieve" not in call:
            call = {"id": f"call-{summary}-{n}", **call}
        calls.append(call)
    return json.dumps({"summary": summary, "tool_calls": calls, "done": done})


@pytest.fixture
def tools() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    specs, disp = default_tool_spec()
    return [s for s in specs if s["function"]["name"] != "read_file"], disp


def _run(client: Scripted, tools: Any, tmp_path: Path, **kw: Any) -> Any:
    return run("task", client, "m", tools[0], tools[1], log_path=tmp_path / "log.jsonl", **kw)


def test_summaries_accumulate_and_tools_run(tools: Any, tmp_path: Path) -> None:
    client = Scripted([
        _r("initial approach", [{"tool": "exec_shell", "args": {"command": "echo a"}}]),
        _r("previous command printed a", [{"tool": "exec_shell", "args": {"command": "echo b"}}]),
        _r("previous command printed b", done="ok")])
    result = _run(client, tools, tmp_path)
    assert result.final == "ok" and result.turns == 3
    # Turn 3 contains both earlier summaries and both observations, in chronological order.
    contents = [m["content"] for m in client.calls[2]]
    assert contents[1] == "task"
    assert json.loads(contents[2])["summary"] == "initial approach"
    assert "echo a" in contents[3]
    assert json.loads(contents[4])["summary"] == "previous command printed a"
    assert "echo b" in contents[5]
    assert len(result.items) == 6  # final model output is appended too


def test_every_previous_message_is_byte_identical(tools: Any, tmp_path: Path) -> None:
    client = Scripted([
        _r("first", [{"tool": "exec_shell", "args": {"command": "echo a"}}]),
        _r("second", [{"tool": "exec_shell", "args": {"command": "echo b"}}]),
        _r("third", done="ok")])
    _run(client, tools, tmp_path)
    assert client.calls[1][:-3] == client.calls[0][:-1]
    assert client.calls[2][:-3] == client.calls[1][:-1]
    # The only changed message is the final cache-friendly instruction.
    assert client.calls[0][-1]["content"] != client.calls[1][-1]["content"]
    assert client.calls[0][0] == client.calls[2][0]


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
    client = Scripted(['{"thoughts": "old protocol"}', _r("recovered", done="ok")])
    result = _run(client, tools, tmp_path)
    assert result.final == "ok"
    assert client.calls[1][2] == {"role": "assistant", "content": '{"thoughts": "old protocol"}'}
    assert "summary must be a non-empty string" in client.calls[1][3]["content"]


def test_done_ignores_tool_calls(tools: Any, tmp_path: Path) -> None:
    marker = tmp_path / "not-created"
    client = Scripted([_r("finished", [{"tool": "write_file", "args": {
        "path": str(marker), "content": "bad"}}], done="ok")])
    result = _run(client, tools, tmp_path)
    assert result.final == "ok"
    assert not marker.exists()


def test_log_contains_durable_summary_and_new_items(tools: Any, tmp_path: Path) -> None:
    client = Scripted([
        _r("run a", [{"tool": "exec_shell", "args": {"command": "echo a"}}]),
        _r("a worked", done="ok")])
    _run(client, tools, tmp_path)
    records = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert records[0]["summary"] == "run a"
    assert records[0]["new_items"][0][0] == "tool_result"
    assert records[1]["summary"] == "a worked"
    assert "max_index" not in records[0]


def test_model_ids_label_results_and_retrieve_past_full_results(tools: Any, tmp_path: Path) -> None:
    client = Scripted([
        _r("make initial observations", [
            {"id": "shell-a", "tool": "exec_shell", "args": {"command": "echo a"}},
            {"id": "shell-b", "tool": "exec_shell", "args": {"command": "echo b"}}]),
        _r("initial results are stored", [{"retrieve": "shell-a"}]),
        _r("retrieved shell-a", done="ok")])
    result = _run(client, tools, tmp_path)
    assert client.calls[1][3]["content"].startswith("<shell-a>\nexec_shell ")
    assert client.calls[1][4]["content"].startswith("<shell-b>\nexec_shell ")
    retrieved = client.calls[2][-2]["content"]
    assert retrieved.startswith("<shell-a>\nexec_shell ")
    assert retrieved.endswith('"returncode":0}')
    assert result.items[-2].content == retrieved


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
    _run(client, tools, tmp_path)
    assert "unknown tool-call id 'missing'" in client.calls[1][3]["content"]
    assert "not run: duplicate tool-call id 'dup'" in client.calls[2][-2]["content"]
    records = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert records[1]["new_items"][0][2] == "dup"


def test_max_turns_returns_without_final(tools: Any, tmp_path: Path) -> None:
    client = Scripted([_r("continue")])
    result = _run(client, tools, tmp_path, max_turns=1)
    assert result.final is None and result.turns == 1
