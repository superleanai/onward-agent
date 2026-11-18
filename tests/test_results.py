from __future__ import annotations

from pathlib import Path

from onward._items import Item
from onward._results import ResultStore


def test_results_are_on_disk_and_retrievable(tmp_path: Path) -> None:
    store = ResultStore(tmp_path / "store")
    store.put("../unsafe/id", Item("tool_result", "x" * 100000))
    assert store.get("../unsafe/id") == Item("tool_result", "x" * 100000)
    files = list((tmp_path / "store").glob("*.json"))
    assert len(files) == 1 and files[0].name != "../unsafe/id"
    assert len(store.paths["../unsafe/id"].read_text()) > 100000
    # Index, not result bodies, stays in RAM.
    assert len(store.paths) == 1
