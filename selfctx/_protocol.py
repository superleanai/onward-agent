"""The structured turn protocol: summarize the previous turn and request tools."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


class ProtocolError(ValueError):
    """The model output does not follow the protocol; the message is shown to the model."""


@dataclass(frozen=True)
class FileRequest:
    path: str
    lines: tuple[int, int] | None  # 1-based, inclusive; None = agentknit's default read
    id: str | None = None


@dataclass(frozen=True)
class ToolRequest:
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    id: str | None = None


@dataclass(frozen=True)
class RetrieveRequest:
    ids: tuple[str, ...]


Request = FileRequest | ToolRequest | RetrieveRequest


@dataclass(frozen=True)
class Output:
    summary: str
    tool_calls: list[Request]
    done: str | None


SYSTEM_PROMPT = """\
You are a coding agent. You write durable summaries and act through structured tool calls.

Your context grows by appending only. Nothing is numbered, edited, removed, or reordered, and you
cannot backtrack. The task comes first. It is followed by earlier model replies, file excerpts,
tool results, and errors in chronological order.

Each reply of yours is exactly one JSON object, nothing else:
{{"summary": "<summary of the previous turn>",
 "tool_calls": [ {{"id": "source-v1", "file": "<path>", "lines": [<first>, <last>]}},
                 {{"id": "tests-before", "tool": "<name>", "args": {{...}}}},
                 {{"retrieve": "tests-before"}} ],
 "done": null}}

"summary" is long-term durable context. It is appended verbatim and remains in every later turn.
At turn 1 there is no previous turn, so explain your initial understanding and approach instead.
After that, summarize what the previous turn established or changed: your action or reasoning, the
tool outcome, important file contents or errors, and the resulting next step. Include exact facts
needed later; do not merely say that a tool ran. Never invent an outcome for a call not yet run.
The raw replies, observations, and all summaries remain visible, so do not repeat them at length.

"tool_calls" is an ordered list. The host executes every call and appends its result before the
next turn, in the order listed. Give every file or tool call a unique, descriptive "id" string of
your choice. The result is headed `<your-id>` and stored durably, including when the visible result
was shortened. Later, {{"retrieve": "your-id"}} appends that exact full stored result without
rerunning the tool. Use this when you need a past result again. Retrieval is not file editing or
backtracking: it only appends a copy and never removes context.

A file call appends an excerpt with 1-based, inclusive line numbers ("lines": null gives the first
100 lines). It also registers a read required before editing an existing file. A tool call runs the
named tool with "args" and appends its result. Do not rerun a call merely to see its old output:
retrieve it by id. Make a fresh call when you need to verify fresh state.

Your context has a cost: every entry is re-read each turn, and a long context dilutes attention.
The last message gives its size and soft budget. Keep summaries compact but complete. Earlier
messages are never rewritten and the changing instruction is last, preserving the prompt-cache
prefix across turns.

When the task is finished, set "done" to your final answer for the user and "tool_calls" to [].

Tools:
{tools}
"""


def system_prompt(tool_specs: list[dict[str, Any]]) -> str:
    """Constant across turns, so it never breaks the prompt-cache prefix."""
    tools = "\n".join(
        json.dumps({"name": t["function"]["name"], "description": t["function"]["description"],
                    "parameters": t["function"]["parameters"]})
        for t in tool_specs)
    return SYSTEM_PROMPT.format(tools=tools)


def _json_object(reply: str) -> dict[str, Any]:
    start, end = reply.find("{"), reply.rfind("}")
    if start < 0 or end < start:
        raise ProtocolError("no JSON object found in your reply")
    try:
        data = json.loads(reply[start:end + 1])
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ProtocolError("the reply must be a JSON object")
    return data


def _request(raw: Any, tools: set[str]) -> Request:
    if not isinstance(raw, dict):
        raise ProtocolError(f"tool call {raw!r} is not an object")
    if "retrieve" in raw:
        ids = raw["retrieve"]
        if isinstance(ids, str):
            ids = [ids]
        if (not isinstance(ids, list) or not ids
                or not all(isinstance(id_, str) and id_.strip() for id_ in ids)):
            raise ProtocolError("retrieve must be an id or a non-empty list of id strings")
        return RetrieveRequest(tuple(ids))
    id_ = raw.get("id")
    if id_ is not None and (not isinstance(id_, str) or not id_.strip()):
        raise ProtocolError("id must be a non-empty string")
    if "file" in raw:
        path, lines = raw["file"], raw.get("lines")
        if not isinstance(path, str):
            raise ProtocolError(f"file call {raw!r}: file must be a string")
        if lines is None:
            return FileRequest(path, None, id_)
        if (not isinstance(lines, list) or len(lines) != 2
                or not all(isinstance(n, int) and not isinstance(n, bool) for n in lines)
                or not 1 <= lines[0] <= lines[1]):
            raise ProtocolError(f"file call {raw!r}: lines must be [first, last] with 1 <= first <= last")
        return FileRequest(path, (lines[0], lines[1]), id_)
    if "tool" in raw:
        name, args = raw["tool"], raw.get("args", {})
        if name not in tools:
            raise ProtocolError(f"unknown tool {name!r}; available: {', '.join(sorted(tools))}")
        if not isinstance(args, dict):
            raise ProtocolError(f"tool call {raw!r}: args must be an object")
        return ToolRequest(name, args, id_)
    raise ProtocolError(f"tool call {raw!r} has neither 'file', 'tool', nor 'retrieve'")


def parse(reply: str, tools: set[str]) -> Output:
    """Parse and validate a reply against the available tools."""
    data = _json_object(reply)
    summary = data.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise ProtocolError("summary must be a non-empty string")
    raw_calls = data.get("tool_calls", [])
    if not isinstance(raw_calls, list):
        raise ProtocolError("tool_calls must be a list")
    done = data.get("done")
    if done is not None and not isinstance(done, str):
        raise ProtocolError("done must be null or a string")
    return Output(summary, [_request(call, tools) for call in raw_calls], done)


def _tool_branch(spec: dict[str, Any]) -> dict[str, Any]:
    """Schema of one tool call: the tool name as a constant and its own parameters."""
    params = spec["function"]["parameters"]
    args = {**params, "additionalProperties": False}
    return {"type": "object", "additionalProperties": False, "required": ["tool", "args"],
            "properties": {"id": {"type": "string", "minLength": 1},
                           "tool": {"type": "string", "const": spec["function"]["name"]}, "args": args}}


def turn_schema(tool_specs: list[dict[str, Any]]) -> dict[str, Any]:
    """JSON schema of one model output, for `response_format: json_schema`.

    Endpoints may only follow it as an instruction (Kimi does), so `parse` and `check_args` still validate.
    """
    file_call = {"type": "object", "additionalProperties": False, "required": ["file", "lines"],
                 "properties": {"id": {"type": "string", "minLength": 1},
                                "file": {"type": "string"},
                                "lines": {"type": ["array", "null"], "minItems": 2, "maxItems": 2,
                                          "items": {"type": "integer", "minimum": 1}}}}
    retrieve_call = {"type": "object", "additionalProperties": False, "required": ["retrieve"],
                     "properties": {"retrieve": {"anyOf": [
                         {"type": "string", "minLength": 1},
                         {"type": "array", "minItems": 1,
                          "items": {"type": "string", "minLength": 1}}]}}}
    return {"type": "object", "additionalProperties": False,
            "required": ["summary", "tool_calls", "done"],
            "properties": {"summary": {"type": "string", "minLength": 1},
                           "tool_calls": {"type": "array", "items": {
                               "anyOf": [file_call] + [_tool_branch(t) for t in tool_specs]
                               + [retrieve_call]}},
                           "done": {"type": ["string", "null"]}}}


def response_format(tool_specs: list[dict[str, Any]]) -> dict[str, Any]:
    """The `response_format` request field carrying `turn_schema`; constant across turns."""
    return {"type": "json_schema",
            "json_schema": {"name": "turn", "strict": True, "schema": turn_schema(tool_specs)}}


def check_args(spec: dict[str, Any], args: dict[str, Any]) -> str | None:
    """Why `args` do not fit the tool's parameters (unknown or missing keys), or None if they do."""
    params = spec["function"]["parameters"]
    known, required = set(params.get("properties", {})), set(params.get("required", []))
    problems = []
    if unknown := sorted(set(args) - known):
        problems.append(f"unknown argument(s) {', '.join(unknown)}")
    if missing := sorted(required - set(args)):
        problems.append(f"missing argument(s) {', '.join(missing)}")
    if not problems:
        return None
    return f"{'; '.join(problems)}. {spec['function']['name']} takes: {', '.join(sorted(known))}" + (
        f" (required: {', '.join(sorted(required))})" if required else "")
