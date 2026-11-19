"""Render tasks, tool-call history, and recent observations as chat messages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Kind = Literal["task", "model_output", "file", "tool_result", "error"]


@dataclass(frozen=True)
class Item:
    """One context entry, with an optional replacement after the next model request."""

    kind: Kind
    content: str
    placeholder: str | None = None


def tokens(text: str) -> int:
    """Rough token estimate (4 characters per token): a relative signal, not a tokenizer."""
    return (len(text) + 3) // 4


def render(items: list[Item], system: str, budget: int | None = None) -> list[dict[str, str]]:
    """Render the current history in order, with the changing instruction last."""
    messages = [{"role": "system", "content": system}]
    for item in items:
        role = "assistant" if item.kind == "model_output" else "user"
        messages.append({"role": role, "content": item.content})
    total = sum(tokens(item.content) for item in items)
    status = f"[context] {len(items)} entries, ~{total} tokens"
    if budget is not None:
        status += f" (budget ~{budget} tokens)"
    status += (
        "\nReply with one JSON object. List the next calls in `tool_calls`. "
        "Give file and tool calls an `id`. Use `retrieve` to request a past result by ID. "
        "Set `done` to the final answer when finished."
    )
    messages.append({"role": "user", "content": status})
    return messages
