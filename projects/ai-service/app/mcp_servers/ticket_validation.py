"""Ticket draft validation helpers for MCP tool parameter learning."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


TicketCategory = Literal["refund", "logistics", "order_issue", "other"]
TicketPriority = Literal["low", "normal", "high"]
TicketTitle = Annotated[str, Field(min_length=5, max_length=80)]
TicketDescription = Annotated[str, Field(min_length=10, max_length=500)]


class TicketDraftValidationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: TicketTitle
    description: TicketDescription
    category: TicketCategory
    priority: TicketPriority = "normal"

    @field_validator("title", "description", mode="before")
    @classmethod
    def strip_text(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


def simplify_validation_errors(exc: ValidationError) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    for error in exc.errors():
        field = ".".join(str(part) for part in error["loc"])
        errors.append(
            {
                "field": field,
                "type": error["type"],
                "message": error["msg"],
            }
        )
    return errors


def validate_ticket_draft_arguments(
    *,
    title: str,
    description: str,
    category: TicketCategory,
    priority: TicketPriority = "normal",
) -> dict[str, Any]:
    try:
        draft = TicketDraftValidationRequest(
            title=title,
            description=description,
            category=category,
            priority=priority,
        )
    except ValidationError as exc:
        return {
            "ok": False,
            "error_code": "INVALID_TOOL_ARGUMENTS",
            "errors": simplify_validation_errors(exc),
            "draft": None,
        }

    return {
        "ok": True,
        "error_code": None,
        "errors": [],
        "draft": draft.model_dump(),
    }
