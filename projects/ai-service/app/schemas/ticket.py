from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.structured import TicketExtraction
from app.schemas.tool_confirmation import ToolConfirmationResponse


class TicketCategory(StrEnum):
    REFUND = "refund"
    ORDER_QUERY = "order_query"
    LOGISTICS = "logistics"
    COMPLAINT = "complaint"
    POLICY_GAP = "policy_gap"


class TicketPriority(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class CreateTicketArgs(BaseModel):
    """The backend-owned contract sent to the Java business service."""

    model_config = ConfigDict(extra="forbid")

    requester_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
    )
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=1000)
    category: TicketCategory
    priority: TicketPriority = TicketPriority.NORMAL
    related_order_id: str | None = Field(default=None, max_length=64)

    @field_validator(
        "requester_id",
        "title",
        "description",
        "related_order_id",
        mode="before",
    )
    @classmethod
    def strip_string_fields(cls, value: object) -> object:
        if isinstance(value, str):
            stripped_value = value.strip()
            return stripped_value or None
        return value


class CreatedTicket(BaseModel):
    """Validated response returned by the Java business service."""

    model_config = ConfigDict(extra="forbid")

    ticket_id: str = Field(pattern=r"^T[-A-Za-z0-9]{4,80}$")
    requester_id: str
    title: str
    description: str
    category: TicketCategory
    priority: TicketPriority
    related_order_id: str | None
    created_at: datetime


class TicketPlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actor_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="Temporary demo actor id. Production code derives this from authentication.",
    )
    message: str = Field(
        min_length=1,
        max_length=1000,
        description="The user's natural-language request for a support ticket.",
    )

    @field_validator("actor_id", "message", mode="before")
    @classmethod
    def strip_request_strings(cls, value: object) -> object:
        if isinstance(value, str):
            stripped_value = value.strip()
            return stripped_value or None
        return value


class TicketPlanResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    extraction: TicketExtraction
    confirmation: ToolConfirmationResponse


class ExecuteTicketConfirmationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actor_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="Actor who owns the confirmed plan.",
    )

    @field_validator("actor_id", mode="before")
    @classmethod
    def strip_actor_id(cls, value: object) -> object:
        if isinstance(value, str):
            stripped_value = value.strip()
            return stripped_value or None
        return value


class TicketExecutionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirmation_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    ticket: CreatedTicket


def get_create_ticket_args_json_schema() -> dict[str, object]:
    return CreateTicketArgs.model_json_schema()
