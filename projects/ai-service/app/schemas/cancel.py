from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CancelOrderArgs(BaseModel):
    """Arguments for the cancel_order tool, owned by the AI service backend."""

    model_config = ConfigDict(extra="forbid")

    order_id: str = Field(..., description="要取消的订单号")
    reason: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="取消原因",
    )
    requester_id: str = Field(
        min_length=1,
        max_length=64,
        description="发起取消的操作者用户 ID",
    )


def get_cancel_order_args_json_schema() -> dict[str, Any]:
    return CancelOrderArgs.model_json_schema()
