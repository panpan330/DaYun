from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class OrderStatus(StrEnum):
    WAITING_SHIPMENT = "waiting_shipment"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELED = "canceled"


class PaymentStatus(StrEnum):
    UNPAID = "unpaid"
    PAID = "paid"
    REFUNDED = "refunded"


class ToolAccessLevel(StrEnum):
    READ = "read"
    WRITE = "write"
    SENSITIVE = "sensitive"


class ToolDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Internal tool name that the backend is willing to recognize.",
    )
    description: str = Field(
        min_length=1,
        max_length=300,
        description="Short backend-owned description of what this tool can do.",
    )
    access_level: ToolAccessLevel = Field(
        description="How risky this tool is from a permission perspective.",
    )
    requires_confirmation: bool = Field(
        default=False,
        description="Whether this tool requires explicit user confirmation.",
    )
    enabled: bool = Field(
        default=True,
        description="Whether this tool can currently be used by the AI service.",
    )
    argument_schema: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON Schema for tool arguments.",
    )


class QueryOrderArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="Order id to query, for example A1001.",
    )

    @field_validator("order_id", mode="before")
    @classmethod
    def strip_order_id(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


class QueryOrderResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_id: str = Field(description="Order id returned by the business system.")
    order_status: OrderStatus = Field(description="Current order fulfillment status.")
    payment_status: PaymentStatus = Field(description="Current payment status.")
    logistics_message: str = Field(
        min_length=1,
        max_length=200,
        description="Short logistics or order status message.",
    )
    latest_event: str = Field(
        min_length=1,
        max_length=200,
        description="Latest simulated business event for this order.",
    )
    can_create_ticket: bool = Field(
        description="Whether this order is suitable for creating a support ticket.",
    )
    source: str = Field(
        default="fake_order_tool",
        description="Where this result comes from.",
    )


class QueryOrderResponse(BaseModel):
    result: QueryOrderResult = Field(description="Validated fake query order result.")


class LangChainToolInfo(BaseModel):
    name: str = Field(description="LangChain tool name.")
    description: str = Field(description="LangChain tool description.")
    args_schema: dict[str, Any] = Field(description="LangChain tool input schema.")


class LangChainToolListResponse(BaseModel):
    tools: list[LangChainToolInfo] = Field(description="Available LangChain tools.")


def get_query_order_args_json_schema() -> dict[str, Any]:
    return QueryOrderArgs.model_json_schema()


def get_query_order_result_json_schema() -> dict[str, Any]:
    return QueryOrderResult.model_json_schema()
