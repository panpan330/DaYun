from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RefundOrderArgs(BaseModel):
    """Arguments for the refund_order tool, owned by the AI service backend."""

    model_config = ConfigDict(extra="forbid")

    order_id: str = Field(..., description="要退款的订单号")
    reason: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="退款原因",
    )
    requester_id: str = Field(
        min_length=1,
        max_length=64,
        description="发起退款的操作者用户 ID",
    )


def get_refund_order_args_json_schema() -> dict[str, Any]:
    return RefundOrderArgs.model_json_schema()
