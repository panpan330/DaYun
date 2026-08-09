from __future__ import annotations

from pathlib import Path
from threading import Lock

from pydantic import BaseModel, Field

from app.evaluation.production_regression import ProductionRegressionRun


_history_lock = Lock()
MAX_STORED_RUNS = 30


class ProductionRegressionRunHistory(BaseModel):
    schema_version: str = "stage11.production_regression_runs.v1"
    runs: list[ProductionRegressionRun] = Field(default_factory=list)


def load_production_regression_runs(path: Path) -> list[ProductionRegressionRun]:
    """Return all stored production regression runs in saved order (oldest first), or [] if none exist."""
    if not path.exists():
        return []
    history = ProductionRegressionRunHistory.model_validate_json(path.read_text(encoding="utf-8"))
    return history.runs


def load_latest_production_regression_run(path: Path) -> ProductionRegressionRun | None:
    runs = load_production_regression_runs(path)
    return runs[-1] if runs else None


def append_production_regression_run(path: Path, run: ProductionRegressionRun) -> ProductionRegressionRun:
    with _history_lock:
        history = (
            ProductionRegressionRunHistory.model_validate_json(path.read_text(encoding="utf-8"))
            if path.exists()
            else ProductionRegressionRunHistory()
        )
        updated = history.model_copy(update={"runs": [*history.runs[-(MAX_STORED_RUNS - 1):], run]})
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        temporary.write_text(updated.model_dump_json(indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    return run
