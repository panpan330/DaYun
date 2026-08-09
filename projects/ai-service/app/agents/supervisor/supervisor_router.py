"""Supervisor routing: rule-based and LLM-based intent routing for the multi-agent system."""

from enum import StrEnum
import logging
from typing import Any, Protocol

from app.agents.ticket_agent import (
    LLMTicketIntentClassifier,
    TICKET_INTENT_CLASSIFICATION_PROMPT,
    classify_ticket_intent,
)
from app.core.config import Settings, get_settings
from app.core.exceptions import AppException


logger = logging.getLogger(__name__)


class SupervisorRoute(StrEnum):
    KNOWLEDGE_QUESTION = "knowledge_question"
    ORDER_QUERY = "order_query"
    TICKET_REQUEST = "ticket_request"
    REFUND_REQUEST = "refund_request"
    CANCEL_REQUEST = "cancel_request"
    SMALLTALK = "smalltalk"
    UNSUPPORTED = "unsupported"
    UNCLEAR = "unclear"


TICKET_INTENT_TO_SUPERVISOR_ROUTE: dict[str, SupervisorRoute] = {
    "policy_question": SupervisorRoute.KNOWLEDGE_QUESTION,
    "order_query": SupervisorRoute.ORDER_QUERY,
    "ticket_request": SupervisorRoute.TICKET_REQUEST,
    "refund_request": SupervisorRoute.REFUND_REQUEST,
    "cancel_request": SupervisorRoute.CANCEL_REQUEST,
    "smalltalk": SupervisorRoute.SMALLTALK,
    "unsupported": SupervisorRoute.UNSUPPORTED,
    "unclear": SupervisorRoute.UNCLEAR,
}


class SupervisorRouter(Protocol):
    def route(self, message: str) -> SupervisorRoute:
        """Route a user message to a supervisor route."""
        ...


class RuleSupervisorRouter:
    def route(self, message: str) -> SupervisorRoute:
        # 直接映射 classify_ticket_intent 的结果：安全边界词（如"直接退款到账"）
        # 判为 unsupported 时保留 SupervisorRoute.UNSUPPORTED（安全拒绝语义），
        # 不做 UNCLEAR 降级；与 LLMSupervisorRouter 路径行为一致。
        classification = classify_ticket_intent(message)
        return TICKET_INTENT_TO_SUPERVISOR_ROUTE[classification["intent"]]


class LLMSupervisorRouter:
    """LLM-based supervisor router. Reuses LLMTicketIntentClassifier for the
    model call, mapping its intent output to a SupervisorRoute."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: Any | None = None,
        prompt_spec: Any = None,
    ) -> None:
        self._settings = settings
        self._classifier = LLMTicketIntentClassifier(
            settings,
            client=client,
            prompt_spec=(
                prompt_spec
                if prompt_spec is not None
                else TICKET_INTENT_CLASSIFICATION_PROMPT
            ),
        )

    def route(self, message: str) -> SupervisorRoute:
        classification = self._classifier.classify_intent(message)
        return TICKET_INTENT_TO_SUPERVISOR_ROUTE[classification["intent"]]

    def route_with_fallback(self, message: str) -> tuple[SupervisorRoute, str]:
        """Return (route, source) where source is 'llm' or 'rule_fallback'."""
        try:
            return self.route(message), "llm"
        except AppException as exc:
            logger.warning(
                "supervisor_llm_route_failed code=%s falling_back_to_rule",
                exc.code,
            )
            return RuleSupervisorRouter().route(message), "rule_fallback"


class FakeLLMSupervisorRouter:
    def __init__(self, route: SupervisorRoute) -> None:
        self._route = route

    def route(self, message: str) -> SupervisorRoute:
        return self._route


def create_supervisor_router(
    settings: Settings | None = None,
    *,
    client: Any | None = None,
) -> SupervisorRouter:
    resolved_settings = settings or get_settings()
    if resolved_settings.resolved_supervisor_router_mode == "llm":
        return LLMSupervisorRouter(resolved_settings, client=client)
    return RuleSupervisorRouter()
