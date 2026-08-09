import logging
from dataclasses import dataclass
from time import perf_counter

import httpx

from app.core.business_context import build_java_internal_headers
from app.core.config import Settings
from app.core.exceptions import AppException
from app.core.trace import build_trace_headers


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JavaFeedbackReceipt:
    feedback_id: int
    rating: str
    reason: str | None


@dataclass(frozen=True)
class JavaFeedbackContext:
    feedback_id: int
    conversation_id: str
    trace_id: str
    reason: str | None
    agent_route: str
    citation_count: int
    human_handoff_suggested: bool
    user_message_excerpt: str | None
    assistant_answer_excerpt: str | None
    citation_summary_json: str | None
    review_status: str
    bad_case_id: str | None
    review_note: str | None


class JavaFeedbackClient:
    """Internal adapter for durable Agent-response feedback owned by Java."""

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
    def from_settings(cls, settings: Settings) -> "JavaFeedbackClient":
        return cls(
            base_url=settings.resolved_java_business_service_base_url,
            timeout_seconds=settings.resolved_java_business_service_timeout_seconds,
            settings=settings,
        )

    def submit(
        self,
        *,
        conversation_id: str,
        trace_id: str,
        rating: str,
        reason: str | None,
        agent_route: str,
        citation_count: int,
        human_handoff_suggested: bool,
        user_message_excerpt: str | None,
        assistant_answer_excerpt: str | None,
        citation_summary_json: str | None,
    ) -> JavaFeedbackReceipt:
        path = "/internal/ai-response-feedback"
        started_at = perf_counter()
        try:
            with httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout_seconds,
                transport=self.transport,
            ) as client:
                response = client.post(
                    path,
                    json={
                        "conversation_id": conversation_id,
                        "trace_id": trace_id,
                        "rating": rating,
                        "reason": reason,
                        "agent_route": agent_route,
                        "citation_count": citation_count,
                        "human_handoff_suggested": human_handoff_suggested,
                        "user_message_excerpt": user_message_excerpt,
                        "assistant_answer_excerpt": assistant_answer_excerpt,
                        "citation_summary_json": citation_summary_json,
                    },
                    headers=self._build_headers(),
                )
        except httpx.RequestError as exc:
            logger.warning(
                "java_feedback_submit_failed path=%s error_type=%s elapsed_ms=%.2f",
                path,
                type(exc).__name__,
                (perf_counter() - started_at) * 1000,
            )
            raise AppException(
                code="FEEDBACK_SERVICE_UNAVAILABLE",
                message="Feedback could not be saved. Please try again later.",
                status_code=502,
            ) from exc

        if response.status_code != 200:
            logger.warning(
                "java_feedback_submit_rejected path=%s status_code=%s elapsed_ms=%.2f",
                path,
                response.status_code,
                (perf_counter() - started_at) * 1000,
            )
            raise AppException(
                code="FEEDBACK_SERVICE_REJECTED",
                message="Feedback could not be saved. Please try again later.",
                status_code=502,
            )
        try:
            data = response.json()["data"]
            return JavaFeedbackReceipt(
                feedback_id=int(data["feedback_id"]),
                rating=str(data["rating"]),
                reason=str(data["reason"]) if data.get("reason") is not None else None,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AppException(
                code="FEEDBACK_SERVICE_RESPONSE_INVALID",
                message="Feedback could not be saved. Please try again later.",
                status_code=502,
            ) from exc

    def _build_headers(self) -> dict[str, str]:
        headers = build_trace_headers()
        if self.settings is not None:
            headers.update(build_java_internal_headers(self.settings))
        return headers

    def get_context(self, feedback_id: int) -> JavaFeedbackContext:
        return self._request_context("get", f"/internal/ai-response-feedback/{feedback_id}")

    def mark_promoted(self, feedback_id: int, *, bad_case_id: str) -> JavaFeedbackContext:
        return self._request_context(
            "post",
            f"/internal/ai-response-feedback/{feedback_id}/promote",
            json={"bad_case_id": bad_case_id},
        )

    def mark_reviewed(
        self,
        feedback_id: int,
        *,
        review_status: str,
        review_note: str,
    ) -> JavaFeedbackContext:
        return self._request_context(
            "post",
            f"/internal/ai-response-feedback/{feedback_id}/review",
            json={"review_status": review_status, "review_note": review_note},
        )

    def _request_context(
        self,
        method: str,
        path: str,
        json: dict[str, object] | None = None,
    ) -> JavaFeedbackContext:
        try:
            with httpx.Client(base_url=self.base_url, timeout=self.timeout_seconds, transport=self.transport) as client:
                response = client.request(method, path, json=json, headers=self._build_headers())
        except httpx.RequestError as exc:
            raise AppException(
                code="FEEDBACK_SERVICE_UNAVAILABLE",
                message="Feedback review service is temporarily unavailable.",
                status_code=502,
            ) from exc
        if response.status_code == 404:
            raise AppException(
                code="FEEDBACK_NOT_FOUND",
                message="The feedback candidate no longer exists.",
                status_code=404,
            )
        if response.status_code != 200:
            raise AppException(
                code="FEEDBACK_SERVICE_REJECTED",
                message="Feedback review could not be completed.",
                status_code=502,
            )
        try:
            data = response.json()["data"]
            return JavaFeedbackContext(
                feedback_id=int(data["feedback_id"]),
                conversation_id=str(data["conversation_id"]),
                trace_id=str(data["trace_id"]),
                reason=str(data["reason"]) if data.get("reason") is not None else None,
                agent_route=str(data["agent_route"]),
                citation_count=int(data["citation_count"]),
                human_handoff_suggested=bool(data["human_handoff_suggested"]),
                user_message_excerpt=_optional_text(data.get("user_message_excerpt")),
                assistant_answer_excerpt=_optional_text(data.get("assistant_answer_excerpt")),
                citation_summary_json=_optional_text(data.get("citation_summary_json")),
                review_status=str(data["review_status"]),
                bad_case_id=_optional_text(data.get("bad_case_id")),
                review_note=_optional_text(data.get("review_note")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AppException(
                code="FEEDBACK_SERVICE_RESPONSE_INVALID",
                message="Feedback review service returned an invalid response.",
                status_code=502,
            ) from exc


def _optional_text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None
