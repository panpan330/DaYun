from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ToolDecisionType(StrEnum):
    ANSWER_DIRECTLY = "answer_directly"
    CALL_TOOL = "call_tool"


class ToolCallCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Backend tool name requested by the model.",
    )
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Validated tool arguments requested by the model.",
    )
    call_id: str | None = Field(
        default=None,
        description="Provider tool call id, if the model provider returned one.",
    )


class ToolDecisionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: ToolDecisionType = Field(
        description="Whether the model answered directly or requested a tool call.",
    )
    reply: str | None = Field(
        default=None,
        min_length=1,
        description="Direct model reply when no tool call is needed.",
    )
    tool_call: ToolCallCandidate | None = Field(
        default=None,
        description="Tool call candidate when the model requests a backend tool.",
    )

    @model_validator(mode="after")
    def validate_decision_shape(self) -> "ToolDecisionResponse":
        if self.decision == ToolDecisionType.ANSWER_DIRECTLY:
            if self.reply is None or self.tool_call is not None:
                raise ValueError(
                    "answer_directly requires reply and must not include tool_call"
                )
            return self

        if self.tool_call is None or self.reply is not None:
            raise ValueError("call_tool requires tool_call and must not include reply")
        return self
