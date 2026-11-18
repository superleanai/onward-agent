"""Interactive REPL for a persistent, append-only Onward conversation."""

from __future__ import annotations

import contextlib
import json
import select
import sys
from collections.abc import Callable
from pathlib import Path

from ._loop import Result

RunTask = Callable[[str], tuple[Result, Path]]


def _read_task() -> str:
    """Read one task, preserving pasted lines and supporting backslash continuation."""
    print("onward> ", end="", file=sys.stderr, flush=True)
    try:
        text = sys.stdin.readline()
        if not text:
            raise EOFError
        text = text.rstrip("\n")
        while text.endswith("\\"):
            text = text[:-1]
            print("......> ", end="", file=sys.stderr, flush=True)
            continuation = sys.stdin.readline()
            if not continuation:
                break
            text += "\n" + continuation.rstrip("\n")
        # A terminal paste may contain several lines. Drain only immediately available
        # input; normal interactive input still submits one line at a time.
        if sys.stdin.isatty():
            while select.select([sys.stdin], [], [], 0.01)[0]:
                line = sys.stdin.readline()
                if not line:
                    break
                text += "\n" + line.rstrip("\n")
        return text
    except (EOFError, KeyboardInterrupt):
        print(file=sys.stderr)
        raise


def repl(run_task: RunTask, *, json_output: bool = False) -> int:
    """Read tasks until EOF or `exit`, preserving context and tool-result IDs between tasks."""
    print("Onward REPL. Enter a task; 'exit' or Ctrl-D quits.", file=sys.stderr)
    status = 0
    while True:
        try:
            task = _read_task()
        except (EOFError, KeyboardInterrupt):
            break
        if not task.strip():
            continue
        if task.strip().lower() in {"exit", "quit", "q"}:
            break
        with contextlib.redirect_stdout(sys.stderr):  # tool output streams; stdout is for answers
            result, log = run_task(task)
        if json_output:
            print(json.dumps({"final": result.final, "turns": result.turns,
                              "items": len(result.items), "log": str(log)}))
        elif result.final is not None:
            print(result.final)
        else:
            print(f"onward: no answer after {result.turns} turns; log: {log}", file=sys.stderr)
            status = 1
    return status
