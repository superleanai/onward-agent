from __future__ import annotations

import json

import pytest

from selfctx._protocol import (
    FileRequest,
    ProtocolError,
    ToolRequest,
    check_args,
    parse,
    system_prompt,
    turn_schema,
)

TOOLS = {"exec_shell", "write_file"}


def _reply(**kw: object) -> str:
    return json.dumps({"summary": "previous turn did t", "tool_calls": [], "done": None} | kw)


def test_parses_summary_and_tool_calls_in_order() -> None:
    out = parse(_reply(tool_calls=[
        {"tool": "exec_shell", "args": {"command": "ls"}},
        {"file": "a.py", "lines": [3, 9]},
        {"file": "b.py"}]), TOOLS)
    assert out.summary == "previous turn did t"
    assert out.tool_calls == [ToolRequest("exec_shell", {"command": "ls"}),
                              FileRequest("a.py", (3, 9)), FileRequest("b.py", None)]


def test_json_surrounded_by_text_is_accepted() -> None:
    assert parse("model output\n" + _reply() + "\n", TOOLS).summary == "previous turn did t"


@pytest.mark.parametrize("summary", [None, "", "  ", 3, True, []])
def test_summary_must_be_nonempty_string(summary: object) -> None:
    with pytest.raises(ProtocolError, match="summary"):
        parse(_reply(summary=summary), TOOLS)


def test_missing_summary_rejected() -> None:
    with pytest.raises(ProtocolError, match="summary"):
        parse('{"tool_calls": [], "done": null}', TOOLS)


@pytest.mark.parametrize("call", [
    {"tool": "rm_rf"}, {"tool": "exec_shell", "args": "ls"}, {"file": "a", "lines": [5, 2]},
    {"file": "a", "lines": [0, 2]}, {"file": 3}, {"x": 1}, "ls"])
def test_invalid_tool_call_rejected(call: object) -> None:
    with pytest.raises(ProtocolError):
        parse(_reply(tool_calls=[call]), TOOLS)


def test_tool_calls_must_be_list() -> None:
    with pytest.raises(ProtocolError, match="tool_calls"):
        parse(_reply(tool_calls={}), TOOLS)


def test_not_json_rejected() -> None:
    with pytest.raises(ProtocolError, match="JSON"):
        parse("I will now run ls", TOOLS)


def test_done() -> None:
    assert parse(_reply(done="fixed"), TOOLS).done == "fixed"


def test_system_prompt_is_monotonic_and_lists_tools() -> None:
    spec = [{"type": "function", "function": {"name": "exec_shell", "description": "run",
                                              "parameters": {"type": "object"}}}]
    prompt = system_prompt(spec)
    assert '"name": "exec_shell"' in prompt
    assert "cannot backtrack" in prompt
    assert "durable context" in prompt
    assert "numbered" in prompt


SPECS = [{"type": "function", "function": {"name": "exec_shell", "description": "run", "parameters": {
    "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]


def test_turn_schema_has_summary_and_one_branch_per_tool() -> None:
    js = turn_schema(SPECS)
    assert js["required"] == ["summary", "tool_calls", "done"]
    assert js["properties"]["summary"]["minLength"] == 1
    assert "max_index" not in js["properties"]
    assert "requests" not in js["properties"]
    branch = js["properties"]["tool_calls"]["items"]["anyOf"][1]
    assert branch["properties"]["tool"] == {"type": "string", "const": "exec_shell"}
    assert branch["properties"]["args"]["additionalProperties"] is False
    assert branch["properties"]["args"]["required"] == ["command"]


def test_check_args() -> None:
    assert check_args(SPECS[0], {"command": "ls"}) is None
    assert check_args(SPECS[0], {"command": "ls", "tool": "exec_shell"}) == (
        "unknown argument(s) tool. exec_shell takes: command (required: command)")
    assert "missing argument(s) command" in (check_args(SPECS[0], {}) or "")


def test_schema_shaped_reply_parses() -> None:
    reply = json.dumps({"summary": "s", "done": None,
                        "tool_calls": [{"file": "a", "lines": None},
                                       {"tool": "exec_shell", "args": {}}]})
    assert parse(reply, TOOLS).tool_calls == [FileRequest("a", None), ToolRequest("exec_shell", {})]
