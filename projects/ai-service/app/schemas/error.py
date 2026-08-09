from typing import Any

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    code: str = Field(description="Stable error code for programmatic handling.")
    message: str = Field(description="Human-readable error message.")
    trace_id: str = Field(description="Request trace id for troubleshooting.")
    details: list[dict[str, Any]] | None = Field(
        default=None,
        description="Optional structured error details.",
    )
