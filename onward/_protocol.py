"""The structured turn protocol for tool requests and final answers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


class ProtocolError(ValueError):
    """The model output does not follow the protocol; the message is shown to the model."""


@dataclass(frozen=True)
class FileRequest:
    path: str
    lines: tuple[int, int] | None  # 1-based, inclusive; None = agentknit's default read
    id: str


@dataclass(frozen=True)
class ToolRequest:
    tool: str
    args: dict[str, Any]
    id: str


@dataclass(frozen=True)
class RetrieveRequest:
    ids: tuple[str, ...]


Request = FileRequest | ToolRequest | RetrieveRequest


@dataclass(frozen=True)
class Output:
    tool_calls: list[Request]
    done: str | None


SYSTEM_PROMPT = """\
You are a coding agent. You act through structured tool calls.

The task comes first. Conversation history follows in chronological order. The host represents past
tool-call lists as `[tool calls: N]` without their arguments. Tool results and excerpts are visible
for one model request. In later requests, `[result "id"]` and `[error "id"]` mark where they appeared.
A placeholder can list several IDs if the call retrieved multiple results.{summary_note}

Each reply of yours is exactly one JSON object, nothing else:
{{"tool_calls": [ {{"type": "file", "id": "source-v1", "file": "<path>", "lines": [<first>, <last>]}},
                 {{"type": "tool", "id": "tests-before", "tool": "<name>", "args": {{...}}}},
                 {{"type": "retrieve", "retrieve": "tests-before"}} ],
 "done": null}}

"tool_calls" is an ordered list. Every entry has a "type": "file", "tool", or "retrieve". The
host executes every call and appends its result before the next turn, in the order listed. Give
every file or tool call a unique, descriptive "id" string of your choice; it is mandatory. The
result is headed `<your-id>` and stored on disk, including when the visible result was shortened.
Later, {{"type": "retrieve", "retrieve": "your-id"}} shows the full stored result for one request
without rerunning the tool. Use this when you need a past result again. You can supply a list of
IDs to see several past results in one request.

A file call appends an excerpt with 1-based, inclusive line numbers ("lines": null gives the first
100 lines). It also registers a read required before editing an existing file. A tool call runs the
named tool with "args" and appends its result. Do not rerun a call merely to see its old output:
retrieve it by id. Make a fresh call when you need to verify fresh state.

The last message gives the estimated context size and soft budget.

When the task is finished, set "done" to your final answer for the user and "tool_calls" to [].

Tools:
{tools}
"""


def system_prompt(tool_specs: list[dict[str, Any]], *, summarize_context: bool = False) -> str:
    """Constant across turns, so it never breaks the prompt-cache prefix."""
    tools = "\n".join(
        json.dumps({"name": t["function"]["name"], "description": t["function"]["description"],
                    "parameters": t["function"]["parameters"]})
        for t in tool_specs)
    summary_note = ("\n\nA separate model writes short notes under the tool-call and result markers. "
                    "Use those notes to carry forward findings. Retrieve by ID when you need the exact output."
                    if summarize_context else "")
    return SYSTEM_PROMPT.format(tools=tools, summary_note=summary_note)


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
        raise ProtocolError("each tool call must be an object")
    type_ = raw.get("type")
    if type_ not in ("file", "tool", "retrieve"):
        raise ProtocolError("type must be exactly 'file', 'tool', or 'retrieve'")
    if type_ == "retrieve":
        if "retrieve" not in raw:
            raise ProtocolError("retrieve call requires retrieve id(s)")
        ids = raw["retrieve"]
        if isinstance(ids, str):
            ids = [ids]
        if (not isinstance(ids, list) or not ids
                or not all(isinstance(id_, str) and id_.strip() for id_ in ids)):
            raise ProtocolError("retrieve must be an id or a non-empty list of id strings")
        return RetrieveRequest(tuple(ids))
    id_ = raw.get("id")
    if not isinstance(id_, str) or not id_.strip():
        raise ProtocolError("every file or tool call requires id, a non-empty string")
    if type_ == "file":
        if "file" not in raw:
            raise ProtocolError("file call requires file")
        path, lines = raw["file"], raw.get("lines")
        if not isinstance(path, str):
            raise ProtocolError("file call requires a string path")
        if lines is None:
            return FileRequest(path, None, id_)
        if (not isinstance(lines, list) or len(lines) != 2
                or not all(isinstance(n, int) and not isinstance(n, bool) for n in lines)
                or not 1 <= lines[0] <= lines[1]):
            raise ProtocolError("file call: lines must be [first, last] with 1 <= first <= last")
        return FileRequest(path, (lines[0], lines[1]), id_)
    if "tool" not in raw:
        raise ProtocolError("tool call requires tool")
    name, args = raw["tool"], raw.get("args", {})
    if not isinstance(name, str) or name not in tools:
        raise ProtocolError(f"unknown tool; available: {', '.join(sorted(tools))}")
    if not isinstance(args, dict):
        raise ProtocolError("tool call: args must be an object")
    return ToolRequest(name, args, id_)


def parse(reply: str, tools: set[str]) -> Output:
    """Parse and validate a reply against the available tools."""
    data = _json_object(reply)
    if "tool_calls" not in data or "done" not in data:
        raise ProtocolError("reply requires tool_calls and done")
    raw_calls = data["tool_calls"]
    if not isinstance(raw_calls, list):
        raise ProtocolError("tool_calls must be a list")
    done = data.get("done")
    if done is not None and not isinstance(done, str):
        raise ProtocolError("done must be null or a string")
    return Output([_request(call, tools) for call in raw_calls], done)


def _tool_branch(spec: dict[str, Any]) -> dict[str, Any]:
    """Schema of one tool call: the tool name as a constant and its own parameters."""
    params = spec["function"]["parameters"]
    args = {**params, "additionalProperties": False}
    return {"type": "object", "additionalProperties": False, "required": ["type", "id", "tool", "args"],
            "properties": {"type": {"const": "tool"}, "id": {"type": "string", "minLength": 1},
                           "tool": {"type": "string", "const": spec["function"]["name"]}, "args": args}}


def turn_schema(tool_specs: list[dict[str, Any]]) -> dict[str, Any]:
    """JSON schema of one model output, for `response_format: json_schema`.

    Endpoints may only follow it as an instruction (Kimi does), so `parse` and `check_args` still validate.
    """
    file_call = {"type": "object", "additionalProperties": False, "required": ["type", "id", "file", "lines"],
                 "properties": {"type": {"const": "file"}, "id": {"type": "string", "minLength": 1},
                                "file": {"type": "string"},
                                "lines": {"type": ["array", "null"], "minItems": 2, "maxItems": 2,
                                          "items": {"type": "integer", "minimum": 1}}}}
    retrieve_call = {"type": "object", "additionalProperties": False, "required": ["type", "retrieve"],
                     "properties": {"type": {"const": "retrieve"}, "retrieve": {"anyOf": [
                         {"type": "string", "minLength": 1},
                         {"type": "array", "minItems": 1,
                          "items": {"type": "string", "minLength": 1}}]}}}
    return {"type": "object", "additionalProperties": False,
            "required": ["tool_calls", "done"],
            "properties": {"tool_calls": {"type": "array", "items": {
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
