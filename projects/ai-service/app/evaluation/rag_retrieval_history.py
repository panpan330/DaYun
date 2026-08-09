"""Persistent history of RAG retrieval evaluation runs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Literal

from pydantic import BaseModel, Field


_history_lock = Lock()
MAX_STORED_RUNS = 30


class RagRetrievalRun(BaseModel):
    run_id: str = Field(min_length=1, max_length=128)
    retriever_mode: Literal["keyword", "vector"]
    started_at: datetime
    completed_at: datetime
    top_k: int = Field(ge=1)
    hit_rate: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    precision: float = Field(ge=0, le=1)
    mrr: float = Field(ge=0, le=1)
    case_count: int = Field(ge=0)
    results: list[dict] = Field(default_factory=list)


class RagRetrievalRunHistory(BaseModel):
    schema_version: str = "stage12.rag_retrieval_runs.v1"
    runs: list[RagRetrievalRun] = Field(default_factory=list)


def load_rag_retrieval_runs(path: Path) -> list[RagRetrievalRun]:
    """Return all stored RAG retrieval runs in saved order (oldest first), or [] if none exist."""
    if not path.exists():
        return []
    history = RagRetrievalRunHistory.model_validate_json(path.read_text(encoding="utf-8"))
    return history.runs


def load_latest_rag_retrieval_run(path: Path) -> RagRetrievalRun | None:
    runs = load_rag_retrieval_runs(path)
    return runs[-1] if runs else None


def append_rag_retrieval_run(path: Path, run: RagRetrievalRun) -> RagRetrievalRun:
    with _history_lock:
        history = (
            RagRetrievalRunHistory.model_validate_json(path.read_text(encoding="utf-8"))
            if path.exists()
            else RagRetrievalRunHistory()
        )
        updated = history.model_copy(update={"runs": [*history.runs[-(MAX_STORED_RUNS - 1):], run]})
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        temporary.write_text(updated.model_dump_json(indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    return run
