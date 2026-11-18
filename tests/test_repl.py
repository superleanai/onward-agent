from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

from onward._items import Item
from onward._loop import Result
from onward._repl import repl
from onward._results import ResultStore


def test_repl_accepts_multiple_tasks_in_one_conversation(monkeypatch: Any, capsys: Any) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("first task\nsecond task\nexit\n"))
    tasks: list[str] = []
    items: list[Item] = []

    def run_task(task: str) -> tuple[Result, Path]:
        tasks.append(task)
        items.append(Item("task", task))
        return Result(f"answer {len(tasks)}", items.copy(), len(tasks)), Path("conversation.jsonl")

    assert repl(run_task) == 0
    assert tasks == ["first task", "second task"]
    out = capsys.readouterr().out
    assert "answer 1\n" in out and "answer 2\n" in out


def test_repl_eof_is_clean_and_blank_tasks_are_ignored(monkeypatch: Any) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("\n"))
    assert repl(lambda task: (_ for _ in ()).throw(AssertionError)) == 0


def test_repl_json_has_one_document_per_answer(monkeypatch: Any, capsys: Any) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("task\nexit\n"))

    def run_task(task: str) -> tuple[Result, Path]:
        return Result("answer", [Item("task", task)], 1), Path("conversation.jsonl")

    assert repl(run_task, json_output=True) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["final"] == "answer" and output["items"] == 1


def test_results_remain_addressable_across_repl_tasks(tmp_path: Path) -> None:
    store = ResultStore(tmp_path)
    store.put("first", Item("tool_result", "full result"))
    result, _ = run_task("second", store, [Item("task", "first")], tmp_path)
    assert result.final == "second answer"
    assert store.get("first") == Item("tool_result", "full result")


def run_task(task: str, results: ResultStore, items: list[Item], tmp_path: Path) -> tuple[Result, Path]:
    """Exercise the same persistent-loop API used by the CLI runner."""
    from onward._loop import run

    log = tmp_path / "conversation.jsonl"
    result = run(task, _OneAnswerClient(task), "model", [], {}, results=results,
                 items=items, log_path=log, schema=False)
    return result, log


class _OneAnswerClient:
    def __init__(self, task: str) -> None:
        from types import SimpleNamespace

        reply = json.dumps({"summary": "done", "tool_calls": [], "done": f"{task} answer"})
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))], usage=None)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response))
