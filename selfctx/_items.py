"""Append-only context: one fixed header, a durable summary log, and recent history."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Kind = Literal["task", "model_output", "file", "tool_result", "error"]


@dataclass(frozen=True)
class Item:
    """One append-only piece of context."""

    kind: Kind
    content: str


def tokens(text: str) -> int:
    """Rough token estimate (4 characters per token): a relative signal, not a tokenizer."""
    return (len(text) + 3) // 4


def render(items: list[Item], system: str, budget: int | None = None) -> list[dict[str, str]]:
    """Render history without rewriting any earlier message.

    The task is the first message after the system prompt. Summaries and
    observations stay in their original order; only the final instruction
    changes between turns.
    """
    messages = [{"role": "system", "content": system}]
    for item in items:
        role = "assistant" if item.kind == "model_output" else "user"
        messages.append({"role": role, "content": item.content})
    total = sum(tokens(item.content) for item in items)
    status = f"[context] {len(items)} entries, ~{total} tokens"
    if budget is not None:
        status += f" (budget ~{budget} tokens)"
    status += (
        "\nReply with one JSON object. Summarize the previous turn in "
        "`summary`; this summary is durable context. List the next tool calls "
        "in `tool_calls`, or set `done` to the final answer."
    )
    messages.append({"role": "user", "content": status})
    return messages
