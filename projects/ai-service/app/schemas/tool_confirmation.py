from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ToolConfirmationStatus(StrEnum):
    PENDING = "pending"
    CONFIRMED = "confirmed"


class ToolConfirmationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actor_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="Current actor id. Production code should derive this from authentication.",
    )
    tool_name: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Backend tool that the actor wants to confirm.",
    )
    arguments: dict[str, Any] = Field(
        min_length=1,
        description="Exact tool arguments shown to the actor before confirmation.",
    )

    @field_validator("actor_id", "tool_name", mode="before")
    @classmethod
    def strip_identifier(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


class ConfirmToolConfirmationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actor_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="Actor that confirms the already stored confirmation plan.",
    )

    @field_validator("actor_id", mode="before")
    @classmethod
    def strip_actor_id(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


class ToolConfirmationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirmation_id: str = Field(
        min_length=32,
        max_length=32,
        pattern=r"^[a-f0-9]{32}$",
        description="Opaque backend-generated id that binds this exact confirmation plan.",
    )
    status: ToolConfirmationStatus = Field(
        description="Whether the plan is still waiting for confirmation or was confirmed.",
    )
    actor_id: str = Field(description="Actor bound to the confirmation plan.")
    tool_name: str = Field(description="Tool covered by this confirmation plan.")
    arguments: dict[str, Any] = Field(
        description="Exact stored arguments covered by this confirmation plan.",
    )
    arguments_fingerprint: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[a-f0-9]{64}$",
        description="SHA-256 fingerprint of the stored tool name and arguments.",
    )
    created_at: datetime = Field(description="UTC time when the plan was created.")
    expires_at: datetime = Field(description="UTC time after which confirmation is rejected.")
    message: str = Field(min_length=1, description="User-facing explanation of the plan state.")
