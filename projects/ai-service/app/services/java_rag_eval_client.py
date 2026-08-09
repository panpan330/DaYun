"""Internal adapter for persisting RAG retrieval evaluation runs to Java (MySQL)."""

from __future__ import annotations

import json
import logging
from time import perf_counter
from typing import Any
from uuid import uuid4

import httpx

from app.core.business_context import build_java_internal_headers
from app.core.config import Settings
from app.core.trace import build_trace_headers
from app.evaluation.rag_retrieval_history import RagRetrievalRun

logger = logging.getLogger(__name__)


class JavaRagEvalClientError(Exception):
    """Raised when the Java rag-eval-runs service rejects or is unreachable."""

    def __init__(self, *, status_code: int, code: str, message: str) -> None:
        super().__init__(f"JavaRagEvalClientError(status_code={status_code}, code={code}, message={message})")
        self.status_code = status_code
        self.code = code
        self.message = message


def _run_to_payload(run: RagRetrievalRun) -> dict[str, Any]:
    return {
        "run_id": run.run_id,
        "retriever": run.retriever_mode,
        "started_at": run.started_at.isoformat(),
        "completed_at": run.completed_at.isoformat(),
        "top_k": run.top_k,
        "hit_rate": run.hit_rate,
        "recall": run.recall,
        "precision": run.precision,
        "mrr": run.mrr,
        "case_count": run.case_count,
        "details_json": json.dumps(run.results, ensure_ascii=False),
    }


class JavaRagEvalClient:
    """Persist RAG retrieval evaluation runs to the Java business service."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float,
        settings: Settings | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.strip().rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.settings = settings
        self.transport = transport

    @classmethod
    def from_settings(cls, settings: Settings) -> "JavaRagEvalClient":
        return cls(
            base_url=settings.resolved_java_business_service_base_url,
            timeout_seconds=settings.resolved_java_business_service_timeout_seconds,
            settings=settings,
        )

    def save_run(self, run: RagRetrievalRun) -> None:
        path = "/internal/rag-eval-runs"
        started_at = perf_counter()
        try:
            with httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout_seconds,
                transport=self.transport,
            ) as client:
                response = client.post(path, json=_run_to_payload(run), headers=self._build_headers())
        except httpx.RequestError as exc:
            logger.warning(
                "java_rag_eval_save_failed path=%s run_id=%s error_type=%s elapsed_ms=%.2f",
                path,
                run.run_id,
                type(exc).__name__,
                (perf_counter() - started_at) * 1000,
            )
            raise JavaRagEvalClientError(
                status_code=0,
                code="RAG_EVAL_SERVICE_UNAVAILABLE",
                message="RAG eval service could not be reached.",
            ) from exc

        if response.status_code != 200:
            logger.warning(
                "java_rag_eval_save_rejected path=%s run_id=%s status_code=%s elapsed_ms=%.2f",
                path,
                run.run_id,
                response.status_code,
                (perf_counter() - started_at) * 1000,
            )
            code = "RAG_EVAL_SERVICE_REJECTED"
            try:
                code = response.json().get("code") or code
            except ValueError:
                pass
            raise JavaRagEvalClientError(
                status_code=response.status_code,
                code=code,
                message="RAG eval run could not be saved.",
            )

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        return self._get_data("/api/rag-eval-runs", params={"limit": limit})

    def latest_run(self, retriever: str) -> dict[str, Any] | None:
        return self._get_data("/api/rag-eval-runs/latest", params={"retriever": retriever})

    def _get_data(self, path: str, *, params: dict[str, Any]) -> Any:
        try:
            with httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout_seconds,
                transport=self.transport,
            ) as client:
                response = client.get(path, params=params, headers=self._build_headers())
        except httpx.RequestError as exc:
            raise JavaRagEvalClientError(
                status_code=0,
                code="RAG_EVAL_SERVICE_UNAVAILABLE",
                message="RAG eval service could not be reached.",
            ) from exc
        if response.status_code != 200:
            raise JavaRagEvalClientError(
                status_code=response.status_code,
                code="RAG_EVAL_SERVICE_REJECTED",
                message="RAG eval history could not be loaded.",
            )
        try:
            return response.json()["data"]
        except (KeyError, TypeError, ValueError) as exc:
            raise JavaRagEvalClientError(
                status_code=response.status_code,
                code="RAG_EVAL_SERVICE_RESPONSE_INVALID",
                message="RAG eval service returned an invalid response.",
            ) from exc

    def _build_headers(self) -> dict[str, str]:
        headers = build_trace_headers()
        if self.settings is not None:
            headers.update(build_java_internal_headers(self.settings))
        if "X-Trace-Id" not in headers:
            # CLI / background callers have no request trace context; Java
            # requires X-Trace-Id (8-128 chars) for internal auth.
            headers["X-Trace-Id"] = f"sync-{uuid4().hex[:24]}"
        return headers
