from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.evaluation.eval_platform import EvalRegressionReport


class EvaluationDatasetView(BaseModel):
    name: str
    version: str
    task_type: str
    description: str
    frozen: bool
    baseline_run_id: str | None = None
    tags: list[str] = Field(default_factory=list)


class EvaluationMetricView(BaseModel):
    name: str
    value: float
    direction: Literal["higher_is_better", "lower_is_better"]
    display_value: str


class EvaluationSuiteView(BaseModel):
    name: str
    title: str
    case_count: int
    failed_case_count: int
    passed: bool


class EvaluationRunOverview(BaseModel):
    run_id: str
    dataset_name: str
    dataset_version: str
    candidate_version: str
    model_name: str
    selected_case_count: int
    evaluated_check_count: int
    passed_check_count: int
    failed_check_count: int
    passed: bool
    metrics: list[EvaluationMetricView] = Field(default_factory=list)
    suites: list[EvaluationSuiteView] = Field(default_factory=list)


class ProductionRegressionCaseResultView(BaseModel):
    bad_case_id: str
    title: str
    outcome: Literal["passed", "failed", "not_ready", "error"]
    assertion: str | None = None
    expected: str | None = None
    actual: str | None = None
    detail: str


class ProductionRegressionRunView(BaseModel):
    run_id: str
    started_at: str
    completed_at: str
    total_case_count: int
    passed_case_count: int
    failed_case_count: int
    not_ready_case_count: int
    error_case_count: int
    passed: bool
    results: list[ProductionRegressionCaseResultView] = Field(default_factory=list)


class EvaluationHistoryPoint(BaseModel):
    started_at: str | None = None
    check_pass_rate: float | None = None
    passed: int | None = None
    total: int | None = None
    pass_rate: float | None = None
    hit_rate: float | None = None
    recall: float | None = None
    mrr: float | None = None


class EvaluationHistoryView(BaseModel):
    agent_eval: list[EvaluationHistoryPoint] = Field(default_factory=list)
    production_regression: list[EvaluationHistoryPoint] = Field(default_factory=list)
    rag_retrieval: list[EvaluationHistoryPoint] = Field(default_factory=list)


class EvaluationReportView(BaseModel):
    report: str
    type: str
    generated_at: datetime


class BadCaseSummaryView(BaseModel):
    record_count: int
    open_count: int
    regression_added_count: int
    severity_counts: dict[str, int] = Field(default_factory=dict)
    status_counts: dict[str, int] = Field(default_factory=dict)
    layer_counts: dict[str, int] = Field(default_factory=dict)


class BadCaseItemView(BaseModel):
    id: str
    title: str
    source: str
    task_type: str
    severity: str
    status: str
    failure_layer: str
    failure_category: str
    expected_behavior: str
    actual_behavior: str
    root_cause: str
    recommended_action: str
    regression_action: str
    evidence_summary: str
    tags: list[str] = Field(default_factory=list)


class EvaluationOverviewResponse(BaseModel):
    registry_version: str
    datasets: list[EvaluationDatasetView] = Field(default_factory=list)
    latest_run: EvaluationRunOverview
    bad_case_summary: BadCaseSummaryView
    bad_cases: list[BadCaseItemView] = Field(default_factory=list)
    generated_from_latest_run: bool
    latest_production_regression_run: ProductionRegressionRunView | None = None
    baseline_comparison: EvalRegressionReport | None = None
    trace_id: str


class ProductionFeedbackContextView(BaseModel):
    feedback_id: int
    conversation_id: str
    trace_id: str
    reason: str | None = None
    agent_route: str
    citation_count: int
    human_handoff_suggested: bool
    user_message_excerpt: str | None = None
    assistant_answer_excerpt: str | None = None
    citation_summary: list[dict[str, str | None]] = Field(default_factory=list)
    review_status: str
    bad_case_id: str | None = None
    review_note: str | None = None


class PromoteProductionFeedbackRequest(BaseModel):
    failure_layer: Literal[
        "intent", "field_extraction", "routing", "rag_retrieval", "rag_citation",
        "agent_decision", "tool_calling", "permission", "security", "model_output", "data", "unknown",
    ]
    severity: Literal["critical", "high", "medium", "low"]
    failure_category: str = Field(min_length=1, max_length=120)
    expected_behavior: str = Field(min_length=1, max_length=1000)
    recommended_action: str = Field(min_length=1, max_length=1000)
    regression_action: str = Field(min_length=1, max_length=1000)
    review_note: str = Field(default="", max_length=1000)
    regression_message: str = Field(min_length=1, max_length=4000)
    regression_assertion: Literal[
        "intent", "citation_present", "ticket_confirmation_required",
        "tool_called", "must_ask_for", "must_not_reveal",
    ]
    regression_expected_intent: Literal[
        "policy_question", "order_query", "ticket_request", "refund_request",
        "smalltalk", "unsupported", "unclear",
    ] | None = None
    regression_expected_tool: str | None = Field(
        default=None, max_length=200, description="tool_called 断言：期望被调用的工具名"
    )
    regression_must_ask_fields: list[str] = Field(
        default_factory=list, description="must_ask_for 断言：必须追问的字段"
    )
    regression_must_not_reveal_terms: list[str] = Field(
        default_factory=list, description="must_not_reveal 断言：不得泄露的 term"
    )

    @model_validator(mode="after")
    def validate_regression_assertion(self) -> "PromoteProductionFeedbackRequest":
        if self.regression_assertion == "intent" and self.regression_expected_intent is None:
            raise ValueError("regression_expected_intent is required for an intent assertion")
        if self.regression_assertion != "intent" and self.regression_expected_intent is not None:
            raise ValueError("regression_expected_intent is only supported by an intent assertion")
        if self.regression_assertion == "tool_called" and not self.regression_expected_tool:
            raise ValueError("regression_expected_tool is required for a tool_called assertion")
        if self.regression_assertion != "tool_called" and self.regression_expected_tool is not None:
            raise ValueError("regression_expected_tool is only supported by a tool_called assertion")
        if self.regression_assertion == "must_ask_for" and not self.regression_must_ask_fields:
            raise ValueError("regression_must_ask_fields is required for a must_ask_for assertion")
        if self.regression_assertion != "must_ask_for" and self.regression_must_ask_fields:
            raise ValueError("regression_must_ask_fields is only supported by a must_ask_for assertion")
        if self.regression_assertion == "must_not_reveal" and not self.regression_must_not_reveal_terms:
            raise ValueError(
                "regression_must_not_reveal_terms is required for a must_not_reveal assertion"
            )
        if self.regression_assertion != "must_not_reveal" and self.regression_must_not_reveal_terms:
            raise ValueError(
                "regression_must_not_reveal_terms is only supported by a must_not_reveal assertion"
            )
        return self


class PromoteProductionFeedbackResponse(BaseModel):
    bad_case: BadCaseItemView
    regression_draft: dict[str, object]
    written_case_id: str | None = Field(
        default=None,
        description="写回 agent_cases.json 的正式评测用例 id；幂等或写回失败时为 None",
    )


class ReviewProductionFeedbackRequest(BaseModel):
    review_status: Literal["triaged", "closed"]
    review_note: str = Field(default="", max_length=1000)
