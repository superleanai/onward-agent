from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from onward import __main__ as cli
from onward._loop import Result


class TtyInput(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_no_arguments_on_a_terminal_defaults_to_repl(monkeypatch: Any) -> None:
    stdin = TtyInput()
    monkeypatch.setattr(cli.sys, "stdin", stdin)
    monkeypatch.setattr(cli, "_create_runner", lambda args: lambda task: (_ for _ in ()).throw(AssertionError))
    calls: list[dict[str, Any]] = []

    def fake_repl(run_task: Any, *, json_output: bool) -> int:
        calls.append({"run_task": run_task, "json_output": json_output})
        return 0

    monkeypatch.setattr("onward._repl.repl", fake_repl)
    assert cli.main([]) == 0
    assert len(calls) == 1 and calls[0]["json_output"] is False
    assert stdin.tell() == 0


def test_piped_stdin_remains_a_one_shot_task(monkeypatch: Any, tmp_path: Path, capsys: Any) -> None:
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO("piped task\n"))
    monkeypatch.setattr("onward._repl.repl", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError))
    tasks: list[str] = []

    def run_task(task: str) -> tuple[Result, Path]:
        tasks.append(task)
        return Result("piped answer", [], 1), tmp_path / "run.jsonl"

    monkeypatch.setattr(cli, "_create_runner", lambda args: run_task)
    assert cli.main([]) == 0
    assert tasks == ["piped task\n"]
    assert capsys.readouterr().out == "piped answer\n"


def test_empty_piped_task_is_rejected_before_creating_a_client(monkeypatch: Any) -> None:
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO(" \n"))
    monkeypatch.setattr(cli, "_create_runner", lambda args: (_ for _ in ()).throw(AssertionError))
    assert cli.main([]) == 2
