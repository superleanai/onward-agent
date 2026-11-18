"""Disk-backed, content-addressed storage for full tool-call results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ._items import Item


class ResultStore:
    """Persist results as files; retain only a tiny ID-to-path index in RAM."""

    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self.paths: dict[str, Path] = {}
        self.index_path = directory / "index.jsonl"

    def put(self, id_: str, item: Item) -> None:
        if id_ in self.paths:
            raise ValueError(f"duplicate tool-call id {id_!r}")
        # A hash avoids trusting a model-chosen ID as a filesystem path.
        digest = hashlib.sha256(id_.encode()).hexdigest()
        path = self.directory / f"{digest}.json"
        # Exclusive creation prevents accidental overwrite even after a hash collision.
        with path.open("x") as stream:
            json.dump({"id": id_, "kind": item.kind, "content": item.content}, stream)
        with self.index_path.open("a") as stream:
            stream.write(json.dumps({"id": id_, "file": path.name,
                                     "kind": item.kind, "characters": len(item.content)}) + "\n")
        self.paths[id_] = path

    def get(self, id_: str) -> Item | None:
        path = self.paths.get(id_)
        if path is None:
            return None
        data = json.loads(path.read_text())
        return Item(data["kind"], data["content"])
