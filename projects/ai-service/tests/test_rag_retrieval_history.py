"""Tests for RAG retrieval eval run history persistence."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.evaluation.rag_retrieval_history import (
    RagRetrievalRun,
    append_rag_retrieval_run,
    load_rag_retrieval_runs,
)


def _make_run(run_id: str, hit_rate: float = 0.8) -> RagRetrievalRun:
    return RagRetrievalRun(
        run_id=run_id,
        retriever_mode="keyword",
        started_at=datetime(2026, 8, 7, 12, 0, tzinfo=timezone.utc),
        completed_at=datetime(2026, 8, 7, 12, 1, tzinfo=timezone.utc),
        top_k=3,
        hit_rate=hit_rate,
        recall=0.7,
        precision=0.3,
        mrr=0.65,
        case_count=12,
        results=[],
    )


def test_append_and_load_rag_retrieval_runs(tmp_path: Path) -> None:
    path = tmp_path / "rag_retrieval_runs.json"
    append_rag_retrieval_run(path, _make_run("run-001"))
    append_rag_retrieval_run(path, _make_run("run-002", hit_rate=0.9))

    runs = load_rag_retrieval_runs(path)
    assert [r.run_id for r in runs] == ["run-001", "run-002"]
    assert runs[-1].hit_rate == 0.9


def test_load_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_rag_retrieval_runs(tmp_path / "nope.json") == []


def test_append_is_atomic_and_keeps_max(tmp_path: Path) -> None:
    path = tmp_path / "rag_retrieval_runs.json"
    for i in range(35):
        append_rag_retrieval_run(path, _make_run(f"run-{i:03d}"))
    runs = load_rag_retrieval_runs(path)
    assert len(runs) == 30
    assert runs[0].run_id == "run-005"
    assert runs[-1].run_id == "run-034"
