"""The monotonic agent loop: durable summaries plus structured tool calls."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from agentknit import dispatch

from ._items import Item, render, tokens
from ._protocol import FileRequest, ProtocolError, Request, check_args, parse, response_format, system_prompt

_ECHOED_TAGS = re.compile(r"\A\s*(?:\[context\] [^\n{]*\n)+")
_ENVELOPE = re.compile(r"\A<[0-9a-f]{8}[^>]*>\n?(.*?)</[0-9a-f]{8}>\s*\Z", re.DOTALL)


class Client(Protocol):
    """The subset of an OpenAI-compatible client the loop uses."""

    @property
    def chat(self) -> Any: ...


@dataclass
class Result:
    final: str | None
    items: list[Item]
    turns: int


def _cap(item: Item, limit: int) -> Item:
    """Keep head and tail of an over-long item, so one tool output cannot overflow the context window."""
    if len(item.content) <= limit:
        return item
    half = limit // 2
    dropped = len(item.content) - 2 * half
    note = f"\n[... {dropped} characters truncated by the host; narrow the command (grep, head, tail) ...]\n"
    return Item(item.kind, item.content[:half] + note + item.content[-half:])


def _read(req: FileRequest, tool_dispatch: dict[str, Any]) -> Item:
    args: dict[str, Any] = {"path": req.path}
    if req.lines:
        args |= {"offset": req.lines[0], "limit": req.lines[1] - req.lines[0] + 1}
    text, meta = dispatch("read_file", args, tool_dispatch)
    if meta.get("ok") is False:
        return Item("error", f"file {req.path}: {text}")
    match = _ENVELOPE.match(text)
    body = match.group(1) if match else text
    first = req.lines[0] if req.lines else 1
    numbered = "".join(f"{n}\t{line}" for n, line in enumerate(body.splitlines(keepends=True), first))
    last = first + max(0, len(body.splitlines()) - 1)
    return Item("file", f"{req.path} lines {first}-{last}\n{numbered}")


def _execute(req: Request, tool_dispatch: dict[str, Any], specs: dict[str, dict[str, Any]]) -> Item:
    if isinstance(req, FileRequest):
        return _read(req, tool_dispatch)
    header = f"{req.tool} {json.dumps(req.args)[:200]}"
    if problem := check_args(specs[req.tool], req.args):
        return Item("error", f"{header}\nnot run: {problem}")
    text, meta = dispatch(req.tool, req.args, tool_dispatch)
    return Item("error" if meta.get("ok") is False else "tool_result", f"{header}\n{text}")


def _usage(response: Any) -> dict[str, Any] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    # OpenAI puts cache hits in prompt_tokens_details; agentknit's client (e.g. Kimi) at top level.
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", None) if details else None
    if cached is None:
        cached = getattr(usage, "cached_tokens", None)
    return {"prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "cached_tokens": cached}


def run(task: str, client: Client, model: str, tool_specs: list[dict[str, Any]],
        tool_dispatch: dict[str, Any], *, request_params: dict[str, Any] | None = None,
        max_turns: int = 50, log_path: Path | None = None, schema: bool = True,
        max_item_chars: int = 30_000, context_budget: int | None = None) -> Result:
    """Run `task` until the model sets `done` or `max_turns` is reached.

    Every valid model reply is appended verbatim, making its `summary` durable.
    `schema` sends the turn schema as `response_format` (the endpoint must accept `json_schema`).
    """
    specs = {t["function"]["name"]: t for t in tool_specs}
    tools = set(specs)
    system = system_prompt(tool_specs)
    items = [Item("task", task)]
    params = dict(request_params or {})
    if schema:
        params["extra_body"] = {**params.get("extra_body", {}), "response_format": response_format(tool_specs)}
    log = log_path.open("a") if log_path else None
    try:
        for turn in range(1, max_turns + 1):
            response = client.chat.completions.create(
                model=model, messages=render(items, system, context_budget), **params)
            reply = _ECHOED_TAGS.sub("", response.choices[0].message.content or "")
            record: dict[str, Any] = {"turn": turn, "items_before": len(items),
                                      "usage": _usage(response),
                                      "finish_reason": getattr(response.choices[0], "finish_reason", None),
                                      "reply": reply}
            try:
                out = parse(reply, tools)
            except ProtocolError as exc:
                items = items + [Item("model_output", reply), Item("error", f"protocol error: {exc}")]
                record["error"] = str(exc)
            else:
                record["summary"] = out.summary
                if out.done is not None:
                    items = items + [Item("model_output", reply)]
                    record["items_after"] = len(items)
                    if log:
                        log.write(json.dumps(record) + "\n")
                    return Result(out.done, items, turn)
                new = [_cap(_execute(call, tool_dispatch, specs), max_item_chars) for call in out.tool_calls]
                items = items + [Item("model_output", reply)] + new
                record["new_items"] = [[item.kind, tokens(item.content)] for item in new]
            record["items_after"] = len(items)
            if log:
                log.write(json.dumps(record) + "\n")
                log.flush()
        return Result(None, items, max_turns)
    finally:
        if log:
            log.close()
