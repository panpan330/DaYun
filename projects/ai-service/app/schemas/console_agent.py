from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.rag.generator import RagCitation
from app.schemas.chat import ConsoleChatRequest
from app.schemas.ticket import CreatedTicket


class ConsoleAgentTicketFields(BaseModel):
    model_config = ConfigDict(extra="forbid")

    issue_type: Literal["cancel", "refund", "logistics", "complaint", "policy_gap"]
    order_id: str | None = Field(default=None, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    description: str = Field(min_length=1, max_length=1000)
    user_request: str = Field(min_length=1, max_length=200)
    urgency: Literal["low", "normal", "high"]
    need_human_review: bool


class ConsoleAgentTicketConfirmation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirmation_id: str = Field(min_length=8, max_length=128)
    title: str = Field(min_length=1, max_length=200)
    summary: str = Field(min_length=1, max_length=1000)
    # True only when this confirmation belongs to a refund *execution* flow
    # (graph state refund_request_active).  A refund-typed ticket created
    # through the ordinary ticket flow leaves this False; the browser uses the
    # flag (not the draft fields alone) to pick refund-vs-ticket copy.
    is_refund_execution: bool = False
    # True only when this confirmation belongs to a cancel *execution* flow
    # (graph state cancel_request_active).  A cancel-typed ticket created
    # through the ordinary ticket flow leaves this False; the browser uses the
    # flag (not the draft fields alone) to pick cancel-vs-ticket copy.
    is_cancel_execution: bool = False
    ticket_fields: ConsoleAgentTicketFields


class ConsoleAgentHumanHandoff(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str = Field(min_length=1, max_length=300)
    related_order_id: str | None = Field(default=None, max_length=64)


class ConsoleAgentConversationMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=128)
    role: Literal["user", "assistant", "human_agent"]
    content: str = Field(min_length=1, max_length=4000)
    created_at: datetime
    trace_id: str | None = Field(default=None, min_length=1, max_length=128)
    route: str | None = Field(default=None, min_length=1, max_length=64)
    citations: list[RagCitation] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    pending_ticket_confirmation: ConsoleAgentTicketConfirmation | None = None
    created_ticket: CreatedTicket | None = None
    human_handoff: ConsoleAgentHumanHandoff | None = None
    feedback_rating: Literal["helpful", "unhelpful"] | None = None
    feedback_reason: Literal[
        "answer_incorrect",
        "intent_misunderstood",
        "citation_irrelevant",
        "should_handoff",
        "ticket_flow_incorrect",
        "other",
    ] | None = None


class ConsoleAgentConversationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation_id: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=120)
    updated_at: datetime


class ConsoleAgentConversation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation_id: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=120)
    updated_at: datetime
    messages: list[ConsoleAgentConversationMessage] = Field(default_factory=list)


class ConsoleAgentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reply: str = Field(min_length=1)
    conversation_id: str = Field(min_length=1, max_length=128)
    trace_id: str = Field(min_length=1)
    route: str = Field(min_length=1, max_length=64)
    citations: list[RagCitation] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    pending_ticket_confirmation: ConsoleAgentTicketConfirmation | None = None
    created_ticket: CreatedTicket | None = None
    human_handoff: ConsoleAgentHumanHandoff | None = None
    emotion: str | None = None


class HumanReplyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1, max_length=4000)


class ConversationSummaryView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    source: Literal["llm", "rule"]


class ConsoleAgentConfirmationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approved: bool


class ConsoleAgentTicketCorrectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticket_fields: ConsoleAgentTicketFields


class ConsoleAgentFeedbackRequest(BaseModel):
    """The browser may only express an opinion about a known Agent reply."""

    model_config = ConfigDict(extra="forbid")

    trace_id: str = Field(min_length=8, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")
    rating: Literal["helpful", "unhelpful"]
    reason: Literal[
        "answer_incorrect",
        "intent_misunderstood",
        "citation_irrelevant",
        "should_handoff",
        "ticket_flow_incorrect",
        "other",
    ] | None = None


class ConsoleAgentFeedbackResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feedback_id: int
    rating: Literal["helpful", "unhelpful"]
    reason: str | None = None


class ConsoleAgentMessageRequest(ConsoleChatRequest):
    """Customer-facing Agent request. Identity is derived from the access token."""
