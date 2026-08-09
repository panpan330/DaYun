"""Multiturn conversation evaluation case schemas."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from app.agents.intent_evaluation import _validate_unique_case_ids


class MultiturnTurn(BaseModel):
    message: str = Field(min_length=1)
    expected_intent: str = Field(min_length=1)
    expected_route: str = Field(min_length=1)
    requires_history: bool = False


class MultiturnFinalAnswer(BaseModel):
    expected_intent: str = Field(min_length=1)


class MultiturnEvalCase(BaseModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    conversation: list[MultiturnTurn] = Field(min_length=1)
    final_answer: MultiturnFinalAnswer | None = None

    @model_validator(mode="after")
    def require_multiple_turns(self) -> "MultiturnEvalCase":
        if len(self.conversation) < 2:
            raise ValueError("multiturn case requires at least 2 turns")
        return self


class MultiturnEvalDataset(BaseModel):
    schema_version: str = Field(min_length=1)
    description: str = ""
    cases: list[MultiturnEvalCase] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_case_ids(self) -> "MultiturnEvalDataset":
        _validate_unique_case_ids(self.cases)
        return self


def load_multiturn_cases(path: str | Path) -> list[MultiturnEvalCase]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    dataset = MultiturnEvalDataset.model_validate(data)
    return dataset.cases
