from collections.abc import Mapping
from datetime import datetime
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.core.exceptions import AppException


T = TypeVar("T", bound=BaseModel)


class JavaApiEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    success: bool
    code: str
    message: str
    data: dict[str, Any] | None
    trace_id: str


class JavaOrderToolView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_id: str
    order_status: str
    payment_status: str
    logistics_message: str
    latest_event: str
    can_create_ticket: bool
    user_visible_summary: str


class JavaTicketToolView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticket_id: str = Field(pattern=r"^T-[0-9a-fA-F-]{36}$")
    ticket_status: str
    title: str
    category: str
    priority: str
    related_order_id: str | None
    created_at: datetime
    user_visible_summary: str


def validate_java_success_envelope(
    payload: Mapping[str, Any],
    *,
    data_model: type[T],
) -> T:
    try:
        envelope = JavaApiEnvelope.model_validate(payload)
        if envelope.success is not True or envelope.code != "OK":
            raise ValueError("Java response is not a successful OK envelope.")
        if envelope.data is None:
            raise ValueError("Java successful response must contain data.")
        return data_model.model_validate(envelope.data)
    except (ValidationError, ValueError) as exc:
        raise AppException(
            code="JAVA_CONTRACT_VALIDATION_FAILED",
            message="Java 业务服务响应不符合 Python AI 服务契约。",
            status_code=502,
        ) from exc
