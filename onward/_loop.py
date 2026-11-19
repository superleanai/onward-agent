"""Execute structured tool calls and replace observations after one model request."""

from __future__ import annotations

import json
import re
import tempfile
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from agentknit import dispatch

from ._items import Item, render, tokens
from ._protocol import (
    FileRequest,
    ProtocolError,
    Request,
    RetrieveRequest,
    check_args,
    parse,
    response_format,
    system_prompt,
)
from ._results import ResultStore

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


Summarize = Callable[[str, list[Item]], list[str]]


def _compact(items: list[Item], task: str, summarize: Summarize | None) -> list[Item]:
    pending = [item for item in items if item.placeholder is not None]
    if not pending:
        return items
    summaries = iter(summarize(task, pending)) if summarize is not None else None
    return [
        Item(item.kind, item.placeholder if summaries is None else f"{item.placeholder}\n{next(summaries)}")
        if item.placeholder is not None else item
        for item in items
    ]


def _cap(item: Item, limit: int) -> Item:
    """Keep the head and tail of a long observation."""
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


def _execute(req: Request, tool_dispatch: dict[str, Any], specs: dict[str, dict[str, Any]],
             results: ResultStore) -> Item:
    if isinstance(req, RetrieveRequest):
        retrieved = [(id_, results.get(id_)) for id_ in req.ids]
        content = "\n\n".join(
            item.content if item is not None else f"unknown tool-call id {id_!r}"
            for id_, item in retrieved)
        kind: Literal["error", "tool_result"] = (
            "error" if any(item is None for _, item in retrieved) else "tool_result")
        return Item(kind, content)
    if req.id in results.paths:
        return Item("error", f"<{req.id}>\nnot run: duplicate tool-call id {req.id!r}")
    result = _read(req, tool_dispatch) if isinstance(req, FileRequest) else _run_tool(req, tool_dispatch, specs)
    result = Item(result.kind, f"<{req.id}>\n{result.content}")
    results.put(req.id, result)
    return result


def _run_tool(req: Request, tool_dispatch: dict[str, Any], specs: dict[str, dict[str, Any]]) -> Item:
    assert not isinstance(req, FileRequest | RetrieveRequest)
    if problem := check_args(specs[req.tool], req.args):
        return Item("error", f"{req.tool}\nnot run: {problem}")
    text, meta = dispatch(req.tool, req.args, tool_dispatch)
    return Item("error" if meta.get("ok") is False else "tool_result", f"{req.tool}\n{text}")


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
        max_item_chars: int = 30_000, context_budget: int | None = None,
        trace_context: bool = False, summarize: Summarize | None = None,
        results_dir: Path | None = None, items: list[Item] | None = None,
        results: ResultStore | None = None, start_turn: int = 1) -> Result:
    """Run `task` until the model sets `done` or `max_turns` is reached.

    Calls are compacted immediately. Results are compacted after the acting model sees them once.
    `schema` sends the turn schema as `response_format` (the endpoint must accept `json_schema`).
    Existing `items`, `results`, and `start_turn` let a REPL continue one conversation.
    """
    specs = {t["function"]["name"]: t for t in tool_specs}
    tools = set(specs)
    system = system_prompt(tool_specs, summarize_context=summarize is not None)
    items = (items or []) + [Item("task", task)]
    if results is None:
        if results_dir is None:
            results_dir = Path(tempfile.gettempdir()) / f"onward-results-{uuid.uuid4().hex}"
        results = ResultStore(results_dir)  # full results on disk; only ID/path index in RAM
    params = dict(request_params or {})
    if schema:
        params["extra_body"] = {**params.get("extra_body", {}), "response_format": response_format(tool_specs)}
    log = log_path.open("a") if log_path else None
    try:
        for turn in range(start_turn, start_turn + max_turns):
            messages = render(items, system, context_budget)
            response = client.chat.completions.create(model=model, messages=messages, **params)
            reply = _ECHOED_TAGS.sub("", response.choices[0].message.content or "")
            record: dict[str, Any] = {"turn": turn, "items_before": len(items),
                                      "usage": _usage(response),
                                      "finish_reason": getattr(response.choices[0], "finish_reason", None),
                                      "reply": reply}
            if trace_context:
                record["messages"] = messages
            try:
                out = parse(reply, tools)
            except ProtocolError as exc:
                out = None
                record["error"] = str(exc)
            if out is not None and out.done is None:
                marker = f"[tool calls: {len(out.tool_calls)}]"
                items.append(Item("model_output", reply, marker) if out.tool_calls else Item("model_output", marker))
            items = _compact(items, task, summarize)
            if out is None:
                items.extend([Item("model_output", "[invalid reply]"),
                              Item("error", f"protocol error: {record['error']}")])
            else:
                if out.done is not None:
                    items.append(Item("model_output", json.dumps({"tool_calls": [], "done": out.done})))
                    record["items_after"] = len(items)
                    if log:
                        log.write(json.dumps(record) + "\n")
                    return Result(out.done, items, turn)
                new_items: list[list[str | int]] = []
                for call in out.tool_calls:
                    item = _execute(call, tool_dispatch, specs, results)
                    if isinstance(call, RetrieveRequest):
                        ids = call.ids
                    else:
                        ids = (call.id,)
                        item = _cap(item, max_item_chars)
                        new_items.append([item.kind, tokens(item.content), call.id])
                    label = "error" if item.kind == "error" else "result"
                    references = ", ".join(json.dumps(id_) for id_ in ids)
                    items.append(Item(item.kind, item.content, f"[{label} {references}]"))
                record["new_items"] = new_items
            record["items_after"] = len(items)
            if log:
                log.write(json.dumps(record) + "\n")
                log.flush()
        return Result(None, items, start_turn + max_turns - 1)
    finally:
        if log:
            log.close()
