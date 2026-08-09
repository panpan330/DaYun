from __future__ import annotations

import json
from pathlib import Path
from threading import Lock

from app.evaluation.eval_platform import EvalRunSnapshot


DEFAULT_MAX_SNAPSHOTS = 50

_store_lock = Lock()


class SnapshotStore:
    """Append-only JSON store of :class:`EvalRunSnapshot` objects.

    The underlying file holds a JSON array of snapshots ordered oldest first.
    Writes are atomic: the updated array is written to a ``.tmp`` sibling and
    then ``replace``-d over the real path, so a failed write never corrupts an
    existing snapshot file. Only the most recent ``max_snapshots`` entries are
    kept on disk.
    """

    def __init__(self, path: Path, *, max_snapshots: int = DEFAULT_MAX_SNAPSHOTS):
        self._path = path
        self._max_snapshots = max_snapshots

    @property
    def path(self) -> Path:
        return self._path

    def load_latest(self) -> EvalRunSnapshot | None:
        """Return the most recently saved snapshot, or None if none exists."""
        snapshots = self.load_all()
        return snapshots[-1] if snapshots else None

    def save(self, snapshot: EvalRunSnapshot) -> None:
        """Atomically append ``snapshot``, trimming to the newest ``max_snapshots``."""
        with _store_lock:
            snapshots = self.load_all()
            snapshots.append(snapshot)
            trimmed = snapshots[-self._max_snapshots :]
            self._path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._path.with_suffix(f"{self._path.suffix}.tmp")
            temporary.write_text(
                json.dumps(
                    [item.model_dump(mode="json") for item in trimmed],
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            temporary.replace(self._path)

    def load_all(self) -> list[EvalRunSnapshot]:
        """Return all saved snapshots in saved order (oldest first), or [] if none exist."""
        if not self._path.exists():
            return []
        raw_items = json.loads(self._path.read_text(encoding="utf-8"))
        return [EvalRunSnapshot.model_validate(item) for item in raw_items]
