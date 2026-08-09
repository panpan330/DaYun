"""Background batch sync of in-process LLM cost aggregation to Java (MySQL)."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from app.services.java_cost_client import JavaCostClient

logger = logging.getLogger(__name__)


class PendingCostSync:
    """Periodically flushes the in-process CostRecorder to Java (MySQL).

    Records are merged idempotently on the Java side (model + intent + window
    unique key), so a failed batch is restored into the recorder and retried
    on the next cycle without data loss.
    """

    def __init__(
        self,
        *,
        client: JavaCostClient,
        interval_seconds: float = 30.0,
    ) -> None:
        self.client = client
        self.interval_seconds = interval_seconds

    def sync_once(self) -> int:
        from app.services.cost_recorder import get_cost_recorder

        records = get_cost_recorder().flush()
        if not records:
            return 0
        try:
            self.client.sync_cost_records(records)
            return len(records)
        except Exception as exc:
            # Restore the batch so the next cycle retries without data loss.
            logger.warning(
                "cost_sync_item_failed batch_size=%d error_type=%s",
                len(records),
                type(exc).__name__,
            )
            self._restore(records)
            return 0

    def _restore(self, records: list[dict[str, Any]]) -> None:
        from app.services.cost_recorder import get_cost_recorder

        recorder = get_cost_recorder()
        for item in records:
            recorder.record(
                model=item["model"],
                intent=item["intent"],
                input_tokens=item["input_tokens"],
                output_tokens=item["output_tokens"],
                estimated_cost=item["estimated_cost"],
            )

    def run_forever(self, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            started = time.monotonic()
            self.sync_once()
            elapsed = time.monotonic() - started
            stop_event.wait(max(0.0, self.interval_seconds - elapsed))
