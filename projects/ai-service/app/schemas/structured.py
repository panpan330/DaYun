from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class TicketIntent(StrEnum):
    CANCEL = "cancel"
    REFUND = "refund"
    ORDER_QUERY = "order_query"
    LOGISTICS = "logistics"
    COMPLAINT = "complaint"
    UNKNOWN = "unknown"


class TicketUrgency(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class StructuredOutputRequest(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=4000,
        description="User message to extract a ticket from.",
    )


class TicketExtraction(BaseModel):
    intent: TicketIntent = Field(
        description="Business intent extracted from the user message.",
    )
    order_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        description="Order id mentioned by the user, or null when absent.",
    )
    summary: str = Field(
        min_length=1,
        max_length=200,
        description="Short summary of the user issue.",
    )
    urgency: TicketUrgency = Field(
        default=TicketUrgency.NORMAL,
        description="Estimated urgency of the ticket.",
    )
    need_human_review: bool = Field(
        default=False,
        description="Whether a human support agent should review this ticket.",
    )

    @field_validator("order_id", mode="before")
    @classmethod
    def empty_order_id_to_none(cls, value: object) -> object:
        if isinstance(value, str):
            stripped_value = value.strip()
            return stripped_value or None
        return value


class StructuredOutputResponse(BaseModel):
    extraction: TicketExtraction = Field(
        description="Validated structured ticket extraction result.",
    )


def get_ticket_extraction_json_schema() -> dict[str, Any]:
    return TicketExtraction.model_json_schema()
