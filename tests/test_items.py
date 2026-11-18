from __future__ import annotations

from selfctx._items import Item, render


def _ctx() -> list[Item]:
    return [Item("task", "fix it"), Item("model_output", '{"summary": "start"}'),
            Item("tool_result", "ls output")]


def test_render_has_no_item_numbers() -> None:
    messages = render(_ctx(), "SYS")
    assert messages[0] == {"role": "system", "content": "SYS"}
    assert [m["content"] for m in messages[1:-1]] == [
        "fix it", '{"summary": "start"}', "ls output"]
    assert all(not m["content"].startswith("[0]") for m in messages)


def test_model_output_is_assistant_everything_else_user() -> None:
    assert [m["role"] for m in render(_ctx(), "S")[1:-1]] == ["user", "assistant", "user"]


def test_appending_preserves_every_previous_message_byte_for_byte() -> None:
    before = render(_ctx(), "S")
    after = render(_ctx() + [Item("file", "contents")], "S")
    assert after[:-1] == before[:-1] + [{"role": "user", "content": "contents"}]


def test_status_is_last_and_reports_size_budget_and_required_fields() -> None:
    messages = render(_ctx(), "S", budget=20000)
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"].startswith("[context] 3 entries, ~10 tokens (budget ~20000 tokens)")
    assert "`summary`" in messages[-1]["content"]
    assert "`tool_calls`" in messages[-1]["content"]
    assert "`done`" in messages[-1]["content"]


def test_render_does_not_mutate_input() -> None:
    items = _ctx()
    render(items, "S")
    assert items == _ctx()
