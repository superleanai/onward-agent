"""selfctx: run a coding task with durable summaries and structured tool calls."""

from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import sys
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="selfctx", description=__doc__)
    parser.add_argument("task", nargs="?", help="the task (read from stdin when omitted)")
    parser.add_argument("--model", default="k3.1-pro", help="agentknit model name (default: %(default)s)")
    parser.add_argument("--inference-db", default="kimi-coding",
                        help="agentknit inference-db entry (default: %(default)s)")
    parser.add_argument("--max-turns", type=int, default=50, help="default: %(default)s")
    parser.add_argument("--log-dir", type=Path, default=Path("runs"),
                        help="one JSONL log per run, one record per turn (default: %(default)s)")
    parser.add_argument("--context-budget", type=int, default=20_000,
                        help="soft context budget in tokens, shown to the model each turn (default: %(default)s)")
    parser.add_argument("--no-schema", action="store_true",
                        help="do not send the turn schema as response_format (for endpoints without json_schema)")
    parser.add_argument("--json", action="store_true",
                        help="print {final, turns, items, log} as JSON instead of the final answer")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    task = args.task if args.task is not None else sys.stdin.read()
    if not task.strip():
        print("selfctx: empty task", file=sys.stderr)
        return 2

    import agentknit

    from ._loop import run

    schema = agentknit.load_specification(args.model, inference_db=args.inference_db)
    client = agentknit.create_client(schema)
    specs, tool_dispatch = agentknit.default_tool_spec()
    # file calls cover reads
    tool_specs: list[dict[str, Any]] = [s for s in specs if s["function"]["name"] != "read_file"]  # type: ignore[index]
    args.log_dir.mkdir(parents=True, exist_ok=True)
    log = args.log_dir / f"{datetime.datetime.now():%Y%m%d-%H%M%S}.jsonl"
    with contextlib.redirect_stdout(sys.stderr):  # tool output streams; stdout is for the answer
        result = run(task, client, schema.get("model", args.model), tool_specs, tool_dispatch,
                     request_params=dict(schema.get("request_params") or {}),
                     max_turns=args.max_turns, log_path=log, schema=not args.no_schema,
                     context_budget=args.context_budget)
    if args.json:
        print(json.dumps({"final": result.final, "turns": result.turns,
                          "items": len(result.items), "log": str(log)}))
    elif result.final is not None:
        print(result.final)
    else:
        print(f"selfctx: no answer after {result.turns} turns; log: {log}", file=sys.stderr)
    return 0 if result.final is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
