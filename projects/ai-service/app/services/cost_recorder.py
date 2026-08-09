"""In-process LLM cost aggregation with periodic flush."""

from __future__ import annotations

import threading
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any


class CostRecorder:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buckets: dict[tuple[str, str], dict[str, Any]] = {}

    def record(
        self,
        model: str,
        intent: str,
        input_tokens: int,
        output_tokens: int,
        estimated_cost: float,
    ) -> None:
        key = (model, intent)
        with self._lock:
            bucket = self._buckets.setdefault(
                key,
                {
                    "call_count": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "estimated_cost": 0.0,
                },
            )
            bucket["call_count"] += 1
            bucket["input_tokens"] += input_tokens
            bucket["output_tokens"] += output_tokens
            bucket["estimated_cost"] += estimated_cost

    def flush(self) -> list[dict[str, Any]]:
        now = datetime.now(timezone.utc)
        with self._lock:
            items: list[dict[str, Any]] = []
            for (model, intent), bucket in self._buckets.items():
                items.append(
                    {
                        "model": model,
                        "intent": intent,
                        "call_count": bucket["call_count"],
                        "input_tokens": bucket["input_tokens"],
                        "output_tokens": bucket["output_tokens"],
                        "total_tokens": bucket["input_tokens"] + bucket["output_tokens"],
                        "estimated_cost": round(bucket["estimated_cost"], 6),
                        "window_start": now.isoformat(),
                        "window_end": now.isoformat(),
                    }
                )
            self._buckets.clear()
        return items

    def reset_for_test(self) -> None:
        with self._lock:
            self._buckets.clear()


_COST_RECORDER = CostRecorder()
_COST_INTENT: ContextVar[str] = ContextVar("cost_intent", default="general")


def get_cost_recorder() -> CostRecorder:
    return _COST_RECORDER


def set_cost_intent(intent: str) -> None:
    _COST_INTENT.set(intent)


def current_cost_intent() -> str:
    return _COST_INTENT.get()
