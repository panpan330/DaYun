from dataclasses import dataclass

import pytest

from app.core.config import Settings
from app.core.exceptions import AppException
from app.schemas.console_agent import (
    ConsoleAgentConversation,
    ConsoleAgentConversationMessage,
    ConsoleAgentFeedbackRequest,
)
from app.services.console_agent_service import ConsoleAgentActor, ConsoleAgentService
from app.services.java_feedback_client import JavaFeedbackReceipt


@dataclass
class FakeConversationStore:
    conversation: ConsoleAgentConversation
    feedback_update: dict[str, object] | None = None

    def get(self, *, actor: ConsoleAgentActor, conversation_id: str) -> ConsoleAgentConversation | None:
        return self.conversation if conversation_id == self.conversation.conversation_id else None

    def set_assistant_feedback(self, **kwargs: object) -> bool:
        self.feedback_update = kwargs
        return True


class FakeFeedbackClient:
    def __init__(self) -> None:
        self.arguments: dict[str, object] | None = None

    def submit(self, **kwargs: object) -> JavaFeedbackReceipt:
        self.arguments = kwargs
        return JavaFeedbackReceipt(
            feedback_id=42,
            rating=str(kwargs["rating"]),
            reason=kwargs["reason"] if isinstance(kwargs["reason"], str) else None,
        )


def _service() -> tuple[ConsoleAgentService, FakeConversationStore, FakeFeedbackClient]:
    store = FakeConversationStore(
        ConsoleAgentConversation.model_validate(
            {
                "conversation_id": "conversation-001",
                "title": "Order question",
                "updated_at": "2026-08-05T02:00:00Z",
                "messages": [
                    {
                        "id": "message-001",
                        "role": "assistant",
                        "content": "The policy allows a refund within seven days.",
                        "created_at": "2026-08-05T02:00:00Z",
                        "trace_id": "trace-feedback-001",
                        "route": "policy_query",
                        "citations": [],
                    }
                ],
            }
        )
    )
    client = FakeFeedbackClient()
    return (
        ConsoleAgentService(
            Settings(_env_file=None),
            graph=object(),
            conversation_store=store,  # type: ignore[arg-type]
            feedback_client=client,  # type: ignore[arg-type]
        ),
        store,
        client,
    )


def test_feedback_uses_only_response_metadata_stored_by_service() -> None:
    service, store, client = _service()
    actor = ConsoleAgentActor(user_id="U1001", tenant_id="default", roles=("customer",))

    response = service.submit_feedback(
        actor=actor,
        conversation_id="conversation-001",
        request=ConsoleAgentFeedbackRequest(
            trace_id="trace-feedback-001",
            rating="unhelpful",
            reason="citation_irrelevant",
        ),
    )

    assert response.feedback_id == 42
    assert client.arguments == {
        "conversation_id": "conversation-001",
        "trace_id": "trace-feedback-001",
        "rating": "unhelpful",
        "reason": "citation_irrelevant",
        "agent_route": "policy_query",
        "citation_count": 0,
        "human_handoff_suggested": False,
        "user_message_excerpt": None,
        "assistant_answer_excerpt": "The policy allows a refund within seven days.",
        "citation_summary_json": "[]",
    }
    assert store.feedback_update is not None
    assert store.feedback_update["trace_id"] == "trace-feedback-001"


def test_feedback_rejects_trace_id_not_owned_by_the_conversation() -> None:
    service, _, client = _service()
    actor = ConsoleAgentActor(user_id="U1001", tenant_id="default", roles=("customer",))

    with pytest.raises(AppException, match="selected AI response") as error:
        service.submit_feedback(
            actor=actor,
            conversation_id="conversation-001",
            request=ConsoleAgentFeedbackRequest(trace_id="trace-forged-999", rating="helpful"),
        )

    assert error.value.code == "AGENT_RESPONSE_NOT_FOUND"
    assert client.arguments is None
