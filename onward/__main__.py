"""onward: run a coding task with structured tools and retrievable tool results."""

from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._loop import Result

RunTask = Callable[[str], "tuple[Result, Path]"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="onward", description=__doc__)
    parser.add_argument("task", nargs="?", help="one-shot task (default: REPL on a terminal, otherwise read stdin)")
    parser.add_argument("--model", default="k3.1-pro", help="agentknit model name (default: %(default)s)")
    parser.add_argument("--inference-db", default="kimi-coding",
                        help="agentknit inference-db entry (default: %(default)s)")
    parser.add_argument("--max-turns", type=int, default=50, help="default: %(default)s")
    parser.add_argument("--log-dir", type=Path, default=Path("runs"),
                        help="one JSONL log per run, one record per turn (default: %(default)s)")
    parser.add_argument("--trace-context", action="store_true",
                        help="include the exact prompt messages in each turn's run log")
    parser.add_argument("--summary-model", help="use this separate model to summarize compacted context entries")
    parser.add_argument("--summary-inference-db",
                        help="inference-db entry for the summary model (defaults to --inference-db)")
    parser.add_argument("--cache-stable-prefix", action="store_true",
                        help="write only stable acting context to the cache and disable summary cache writes")
    parser.add_argument("--context-budget", type=int, default=20_000,
                        help="soft context budget in tokens, shown to the model each turn (default: %(default)s)")
    parser.add_argument("--no-schema", action="store_true",
                        help="do not send the turn schema as response_format (for endpoints without json_schema)")
    parser.add_argument("--json", action="store_true",
                        help="print {final, turns, items, log} as JSON instead of the final answer")
    return parser


def _create_runner(args: argparse.Namespace) -> RunTask:
    """Create one conversation runner; successive REPL tasks share context and result IDs."""
    import agentknit

    from ._cache_client import create_cache_client
    from ._items import Item
    from ._loop import run
    from ._results import ResultStore
    from ._summary import ContextSummarizer

    schema = agentknit.load_specification(args.model, inference_db=args.inference_db)
    create_client = create_cache_client if args.cache_stable_prefix else agentknit.create_client
    client = create_client(schema)
    summarize = None
    if args.summary_model:
        summary_spec = agentknit.load_specification(
            args.summary_model, inference_db=args.summary_inference_db or args.inference_db)
        summarize = ContextSummarizer(
            create_client(summary_spec), summary_spec.get("model", args.summary_model),
            dict(summary_spec.get("request_params") or {}), cache_stable_prefix=args.cache_stable_prefix,
        )
    specs, tool_dispatch = agentknit.default_tool_spec()
    # file calls cover reads
    tool_specs: list[dict[str, Any]] = [s for s in specs if s["function"]["name"] != "read_file"]  # type: ignore[index]
    args.log_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{datetime.datetime.now():%Y%m%d-%H%M%S}"
    log = args.log_dir / f"{run_id}.jsonl"
    results = ResultStore(args.log_dir / f"{run_id}.results")
    items: list[Item] = []
    next_turn = 1

    def run_task(task: str) -> tuple[Result, Path]:
        nonlocal items, next_turn
        result = run(task, client, schema.get("model", args.model), tool_specs, tool_dispatch,
                     request_params=dict(schema.get("request_params") or {}),
                     max_turns=args.max_turns, log_path=log, schema=not args.no_schema,
                     context_budget=args.context_budget, trace_context=args.trace_context, summarize=summarize,
                     cache_stable_prefix=args.cache_stable_prefix, results=results, items=items, start_turn=next_turn)
        items = result.items
        next_turn = result.turns + 1
        return result, log

    return run_task


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cache_stable_prefix and not args.summary_model:
        parser.error("--cache-stable-prefix requires --summary-model")
    interactive = args.task is None and sys.stdin.isatty()
    task = args.task if args.task is not None else "" if interactive else sys.stdin.read()
    if not interactive and not task.strip():
        print("onward: empty task", file=sys.stderr)
        return 2

    run_task = _create_runner(args)
    if interactive:
        from ._repl import repl

        return repl(run_task, json_output=args.json)
    with contextlib.redirect_stdout(sys.stderr):  # tool output streams; stdout is for the answer
        result, log = run_task(task)
    if args.json:
        print(json.dumps({"final": result.final, "turns": result.turns,
                          "items": len(result.items), "log": str(log)}))
    elif result.final is not None:
        print(result.final)
    else:
        print(f"onward: no answer after {result.turns} turns; log: {log}", file=sys.stderr)
    return 0 if result.final is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
