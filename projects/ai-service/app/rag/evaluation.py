from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.rag.documents import RetrievedChunk
from app.rag.generator import RagAnswer, RagAnswerStatus


RetrievalMatchLevel = Literal["chunk_id", "section", "source", "none"]
RagEvalExpectedBehavior = Literal[
    "answer",
    "no_context",
    "access_denied",
    "security_block",
    "clarify",
]
RagEvalPriority = Literal["p0", "p1", "p2"]
RagEvalDifficulty = Literal[
    "basic",
    "paraphrase",
    "ambiguous",
    "permission",
    "adversarial",
    "no_context",
]
RagAnswerQualityDimension = Literal[
    "behavior",
    "answer_points",
    "citation",
    "refusal",
]
RagAnswerQualitySeverity = Literal["warning", "blocking"]
RagBadCaseLayer = Literal[
    "data",
    "retrieval",
    "ranking",
    "generation",
    "citation",
    "refusal",
    "access_control",
    "security",
    "evaluation",
    "unknown",
]


class RagEvalAccessContext(BaseModel):
    user_id: str | None = None
    tenant_id: str | None = None
    permission_groups: list[str] = Field(default_factory=list)
    business_domains: list[str] = Field(default_factory=list)
    doc_types: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)

    @field_validator("user_id", "tenant_id", mode="before")
    @classmethod
    def normalize_optional_string(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = value.strip()
            return normalized or None
        return value

    @field_validator(
        "permission_groups",
        "business_domains",
        "doc_types",
        "sources",
        mode="before",
    )
    @classmethod
    def normalize_context_lists(cls, value: object) -> object:
        return _normalize_string_list(value, field_name="access context")


class RagEvalExpectation(BaseModel):
    behavior: RagEvalExpectedBehavior = "answer"
    answer_points: list[str] = Field(default_factory=list)
    expected_sources: list[str] = Field(default_factory=list)
    expected_sections: list[str] = Field(default_factory=list)
    expected_chunk_ids: list[str] = Field(default_factory=list)
    forbidden_sources: list[str] = Field(default_factory=list)
    citation_required: bool = True
    refusal_reason_codes: list[str] = Field(default_factory=list)
    notes: str = ""

    @field_validator(
        "answer_points",
        "expected_sources",
        "expected_sections",
        "expected_chunk_ids",
        "forbidden_sources",
        "refusal_reason_codes",
        mode="before",
    )
    @classmethod
    def normalize_expectation_lists(cls, value: object) -> object:
        return _normalize_string_list(value, field_name="expectation")

    @field_validator("notes", mode="before")
    @classmethod
    def normalize_notes(cls, value: object) -> object:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        return value

    @model_validator(mode="after")
    def validate_behavior_expectation(self) -> "RagEvalExpectation":
        has_expected_evidence = any(
            (
                self.expected_sources,
                self.expected_sections,
                self.expected_chunk_ids,
            )
        )
        if set(self.expected_sources) & set(self.forbidden_sources):
            raise ValueError("expected sources and forbidden sources must not overlap")

        if self.behavior == "answer":
            if not self.answer_points:
                raise ValueError("answer cases must define expected answer points")
            if self.citation_required and not has_expected_evidence:
                raise ValueError(
                    "citation-required answer cases must define expected evidence"
                )
            return self

        if self.answer_points:
            raise ValueError("non-answer cases must not define answer points")
        if has_expected_evidence:
            raise ValueError("non-answer cases must not define expected evidence")
        if self.citation_required:
            raise ValueError("non-answer cases must not require citations")
        if not self.refusal_reason_codes:
            raise ValueError("non-answer cases must define refusal reason codes")
        return self


class RagEvalCase(BaseModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    query: str = Field(min_length=1)
    priority: RagEvalPriority = "p1"
    difficulty: RagEvalDifficulty = "basic"
    tags: list[str] = Field(default_factory=list)
    access_context: RagEvalAccessContext = Field(default_factory=RagEvalAccessContext)
    expectation: RagEvalExpectation
    notes: str = ""

    @field_validator("id", "name", "query", mode="before")
    @classmethod
    def normalize_required_string(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("tags", mode="before")
    @classmethod
    def normalize_tags(cls, value: object) -> object:
        return _normalize_string_list(value, field_name="tags")

    @field_validator("notes", mode="before")
    @classmethod
    def normalize_notes(cls, value: object) -> object:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        return value


class RagEvalDatasetReport(BaseModel):
    case_count: int = Field(ge=0)
    answer_case_count: int = Field(ge=0)
    refusal_case_count: int = Field(ge=0)
    p0_case_ids: list[str] = Field(default_factory=list)
    behavior_counts: dict[str, int] = Field(default_factory=dict)
    priority_counts: dict[str, int] = Field(default_factory=dict)
    difficulty_counts: dict[str, int] = Field(default_factory=dict)
    tag_counts: dict[str, int] = Field(default_factory=dict)
    source_counts: dict[str, int] = Field(default_factory=dict)
    missing_recommended_tags: list[str] = Field(default_factory=list)


class RagAnswerQualityFinding(BaseModel):
    code: str = Field(min_length=1)
    dimension: RagAnswerQualityDimension
    severity: RagAnswerQualitySeverity
    message: str = Field(min_length=1)
    evidence: str | None = None


class RagAnswerQualityResult(BaseModel):
    case_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    expected_behavior: RagEvalExpectedBehavior
    actual_behavior: str = Field(min_length=1)
    behavior_passed: bool
    answer_point_coverage: float = Field(ge=0, le=1)
    matched_answer_points: list[str] = Field(default_factory=list)
    missing_answer_points: list[str] = Field(default_factory=list)
    citation_passed: bool
    expected_sources: list[str] = Field(default_factory=list)
    actual_sources: list[str] = Field(default_factory=list)
    matched_sources: list[str] = Field(default_factory=list)
    missing_sources: list[str] = Field(default_factory=list)
    unexpected_sources: list[str] = Field(default_factory=list)
    forbidden_sources_used: list[str] = Field(default_factory=list)
    refusal_passed: bool
    expected_refusal_reason_codes: list[str] = Field(default_factory=list)
    actual_refusal_reason_codes: list[str] = Field(default_factory=list)
    findings: list[RagAnswerQualityFinding] = Field(default_factory=list)
    passed: bool


class RagAnswerQualitySummary(BaseModel):
    case_count: int = Field(ge=0)
    answer_case_count: int = Field(ge=0)
    refusal_case_count: int = Field(ge=0)
    passed_case_count: int = Field(ge=0)
    failed_case_count: int = Field(ge=0)
    pass_rate: float = Field(ge=0, le=1)
    average_answer_point_coverage: float = Field(ge=0, le=1)
    citation_pass_rate: float | None = Field(default=None, ge=0, le=1)
    refusal_pass_rate: float | None = Field(default=None, ge=0, le=1)
    results: list[RagAnswerQualityResult] = Field(default_factory=list)


RECOMMENDED_RAG_EVAL_TAGS = [
    "positive",
    "paraphrase",
    "ambiguous",
    "no_context",
    "permission",
    "security",
    "citation",
]


def load_rag_eval_cases(path: Path | str) -> list[RagEvalCase]:
    raw_text = Path(path).read_text(encoding="utf-8")
    raw_cases = json.loads(raw_text)
    if not isinstance(raw_cases, list):
        raise ValueError("RAG eval cases file must contain a JSON list")
    cases = [RagEvalCase.model_validate(raw_case) for raw_case in raw_cases]
    _validate_unique_rag_eval_case_ids(cases)
    return cases


def build_rag_eval_dataset_report(
    cases: Sequence[RagEvalCase],
) -> RagEvalDatasetReport:
    _validate_unique_rag_eval_case_ids(cases)
    behavior_counts = Counter(eval_case.expectation.behavior for eval_case in cases)
    priority_counts = Counter(eval_case.priority for eval_case in cases)
    difficulty_counts = Counter(eval_case.difficulty for eval_case in cases)
    tag_counts = Counter(tag for eval_case in cases for tag in eval_case.tags)
    source_counts = Counter(
        source
        for eval_case in cases
        for source in eval_case.expectation.expected_sources
    )
    present_tags = set(tag_counts)

    return RagEvalDatasetReport(
        case_count=len(cases),
        answer_case_count=behavior_counts["answer"],
        refusal_case_count=len(cases) - behavior_counts["answer"],
        p0_case_ids=[eval_case.id for eval_case in cases if eval_case.priority == "p0"],
        behavior_counts=dict(sorted(behavior_counts.items())),
        priority_counts=dict(sorted(priority_counts.items())),
        difficulty_counts=dict(sorted(difficulty_counts.items())),
        tag_counts=dict(sorted(tag_counts.items())),
        source_counts=dict(sorted(source_counts.items())),
        missing_recommended_tags=[
            tag for tag in RECOMMENDED_RAG_EVAL_TAGS if tag not in present_tags
        ],
    )


def format_rag_eval_dataset_report(report: RagEvalDatasetReport) -> list[str]:
    lines = [
        "RAG evaluation dataset report",
        f"cases: {report.case_count}",
        f"answer_cases: {report.answer_case_count}",
        f"refusal_cases: {report.refusal_case_count}",
        f"p0_cases: {', '.join(report.p0_case_ids) if report.p0_case_ids else '-'}",
        f"behaviors: {_format_counts(report.behavior_counts)}",
        f"priorities: {_format_counts(report.priority_counts)}",
        f"difficulties: {_format_counts(report.difficulty_counts)}",
        f"tags: {_format_counts(report.tag_counts)}",
        f"sources: {_format_counts(report.source_counts)}",
    ]
    if report.missing_recommended_tags:
        lines.append(
            "missing_recommended_tags: "
            + ", ".join(report.missing_recommended_tags)
        )
    return lines


def evaluate_rag_answer_quality(
    eval_case: RagEvalCase,
    rag_answer: RagAnswer,
    *,
    actual_refusal_reason_codes: Sequence[str] | None = None,
) -> RagAnswerQualityResult:
    expectation = eval_case.expectation
    actual_reason_codes = _normalize_string_list(
        list(actual_refusal_reason_codes or []),
        field_name="actual refusal reason codes",
    )
    actual_reason_codes = [code.upper() for code in actual_reason_codes]
    if rag_answer.status is RagAnswerStatus.NO_CONTEXT and "NO_CONTEXT" not in actual_reason_codes:
        actual_reason_codes.append("NO_CONTEXT")

    actual_behavior = _infer_actual_rag_behavior(rag_answer, actual_reason_codes)
    behavior_passed = actual_behavior == expectation.behavior
    actual_sources = _unique_strings(citation.source for citation in rag_answer.citations)
    expected_sources = expectation.expected_sources
    matched_sources = [source for source in expected_sources if source in actual_sources]
    missing_sources = [source for source in expected_sources if source not in actual_sources]
    unexpected_sources = [
        source for source in actual_sources if expected_sources and source not in expected_sources
    ]
    forbidden_sources_used = [
        source for source in actual_sources if source in set(expectation.forbidden_sources)
    ]
    matched_answer_points, missing_answer_points = _match_answer_points(
        rag_answer.answer,
        expectation.answer_points,
    )
    answer_point_coverage = _ratio(
        len(matched_answer_points),
        len(expectation.answer_points),
        default=1.0,
    )

    citation_passed = _is_citation_passed(
        expectation=expectation,
        rag_answer=rag_answer,
        missing_sources=missing_sources,
        forbidden_sources_used=forbidden_sources_used,
    )
    expected_refusal_codes = expectation.refusal_reason_codes
    refusal_passed = (
        True
        if expectation.behavior == "answer"
        else all(code in actual_reason_codes for code in expected_refusal_codes)
    )
    findings = _build_answer_quality_findings(
        expectation=expectation,
        actual_behavior=actual_behavior,
        behavior_passed=behavior_passed,
        missing_answer_points=missing_answer_points,
        rag_answer=rag_answer,
        missing_sources=missing_sources,
        unexpected_sources=unexpected_sources,
        forbidden_sources_used=forbidden_sources_used,
        expected_refusal_codes=expected_refusal_codes,
        actual_refusal_codes=actual_reason_codes,
        refusal_passed=refusal_passed,
    )
    passed = not any(finding.severity == "blocking" for finding in findings)

    return RagAnswerQualityResult(
        case_id=eval_case.id,
        query=eval_case.query,
        expected_behavior=expectation.behavior,
        actual_behavior=actual_behavior,
        behavior_passed=behavior_passed,
        answer_point_coverage=answer_point_coverage,
        matched_answer_points=matched_answer_points,
        missing_answer_points=missing_answer_points,
        citation_passed=citation_passed,
        expected_sources=expected_sources,
        actual_sources=actual_sources,
        matched_sources=matched_sources,
        missing_sources=missing_sources,
        unexpected_sources=unexpected_sources,
        forbidden_sources_used=forbidden_sources_used,
        refusal_passed=refusal_passed,
        expected_refusal_reason_codes=expected_refusal_codes,
        actual_refusal_reason_codes=actual_reason_codes,
        findings=findings,
        passed=passed,
    )


def evaluate_rag_answer_quality_results(
    cases: Sequence[RagEvalCase],
    answers_by_case_id: Mapping[str, RagAnswer],
    *,
    refusal_reasons_by_case_id: Mapping[str, Sequence[str]] | None = None,
) -> RagAnswerQualitySummary:
    _validate_unique_rag_eval_case_ids(cases)
    reason_map = refusal_reasons_by_case_id or {}
    results = [
        evaluate_rag_answer_quality(
            eval_case,
            answers_by_case_id[eval_case.id],
            actual_refusal_reason_codes=reason_map.get(eval_case.id),
        )
        for eval_case in cases
        if eval_case.id in answers_by_case_id
    ]
    answer_results = [
        result for result in results if result.expected_behavior == "answer"
    ]
    refusal_results = [
        result for result in results if result.expected_behavior != "answer"
    ]

    return RagAnswerQualitySummary(
        case_count=len(results),
        answer_case_count=len(answer_results),
        refusal_case_count=len(refusal_results),
        passed_case_count=sum(1 for result in results if result.passed),
        failed_case_count=sum(1 for result in results if not result.passed),
        pass_rate=_ratio(sum(1 for result in results if result.passed), len(results)),
        average_answer_point_coverage=_average(
            [result.answer_point_coverage for result in answer_results]
        ),
        citation_pass_rate=(
            _ratio(
                sum(1 for result in answer_results if result.citation_passed),
                len(answer_results),
            )
            if answer_results
            else None
        ),
        refusal_pass_rate=(
            _ratio(
                sum(1 for result in refusal_results if result.refusal_passed),
                len(refusal_results),
            )
            if refusal_results
            else None
        ),
        results=results,
    )


def format_rag_answer_quality_summary(summary: RagAnswerQualitySummary) -> list[str]:
    lines = [
        "RAG answer quality summary",
        f"cases: {summary.case_count}",
        f"answer_cases: {summary.answer_case_count}",
        f"refusal_cases: {summary.refusal_case_count}",
        f"passed_cases: {summary.passed_case_count}",
        f"failed_cases: {summary.failed_case_count}",
        f"pass_rate: {summary.pass_rate:.4f}",
        f"average_answer_point_coverage: {summary.average_answer_point_coverage:.4f}",
    ]
    if summary.citation_pass_rate is not None:
        lines.append(f"citation_pass_rate: {summary.citation_pass_rate:.4f}")
    if summary.refusal_pass_rate is not None:
        lines.append(f"refusal_pass_rate: {summary.refusal_pass_rate:.4f}")
    return lines


def format_rag_answer_quality_bad_cases(
    summary: RagAnswerQualitySummary,
) -> list[str]:
    bad_cases = [result for result in summary.results if not result.passed]
    if not bad_cases:
        return ["No bad cases."]

    lines = ["RAG answer quality bad cases:"]
    for result in bad_cases:
        lines.append(
            f"- {result.case_id}: expected={result.expected_behavior} "
            f"actual={result.actual_behavior} "
            f"answer_point_coverage={result.answer_point_coverage:.4f}"
        )
        for finding in result.findings:
            lines.append(
                f"  {finding.severity} {finding.dimension} code={finding.code} "
                f"evidence={finding.evidence or '-'}"
            )
    return lines


class RetrievalEvalCase(BaseModel):
    id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    expected_sources: list[str] = Field(default_factory=list)
    expected_sections: list[str] = Field(default_factory=list)
    expected_chunk_ids: list[str] = Field(default_factory=list)
    expect_no_results: bool = False
    permission_group: str | None = None
    business_domain: str | None = None
    doc_type: str | None = None
    source: str | None = None
    notes: str = ""

    @field_validator("id", "query", mode="before")
    @classmethod
    def normalize_required_string(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator(
        "expected_sources",
        "expected_sections",
        "expected_chunk_ids",
        mode="before",
    )
    @classmethod
    def normalize_expected_values(cls, value: object) -> object:
        if value is None:
            return []
        if isinstance(value, str) or not isinstance(value, Sequence):
            raise ValueError("expected values must be a list of strings")
        normalized_values: list[str] = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("expected values must contain non-blank strings")
            normalized = item.strip()
            if normalized not in normalized_values:
                normalized_values.append(normalized)
        return normalized_values

    @field_validator("permission_group", "business_domain", "doc_type", "source", mode="before")
    @classmethod
    def normalize_optional_filter(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = value.strip()
            return normalized or None
        return value

    @field_validator("notes", mode="before")
    @classmethod
    def normalize_notes(cls, value: object) -> object:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        return value

    @model_validator(mode="after")
    def validate_expectations(self) -> "RetrievalEvalCase":
        has_expected_targets = any(
            (
                self.expected_sources,
                self.expected_sections,
                self.expected_chunk_ids,
            )
        )
        if self.expect_no_results:
            if has_expected_targets:
                raise ValueError(
                    "no-result cases must not define expected retrieval targets"
                )
            return self
        if not has_expected_targets:
            raise ValueError("retrieval eval case must define expected targets")
        return self


class RetrievalEvalItem(BaseModel):
    rank: int = Field(ge=1)
    chunk_id: str = Field(min_length=1)
    source: str | None = None
    section: str | None = None
    score: float
    relevant: bool


class RetrievalEvalCaseResult(BaseModel):
    case_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    top_k: int = Field(gt=0)
    match_level: RetrievalMatchLevel
    metric_applicable: bool
    expected_count: int = Field(ge=0)
    retrieved_count: int = Field(ge=0)
    relevant_retrieved_count: int = Field(ge=0)
    matched_expected_count: int = Field(ge=0)
    hit: bool
    first_relevant_rank: int | None = None
    precision_at_k: float = Field(ge=0, le=1)
    recall_at_k: float = Field(ge=0, le=1)
    reciprocal_rank: float = Field(ge=0, le=1)
    passed: bool
    failed_reason: str | None = None
    retrieved_items: list[RetrievalEvalItem] = Field(default_factory=list)


class RetrievalEvalSummary(BaseModel):
    top_k: int = Field(gt=0)
    case_count: int = Field(ge=0)
    evaluated_case_count: int = Field(ge=0)
    no_result_case_count: int = Field(ge=0)
    passed_case_count: int = Field(ge=0)
    failed_case_count: int = Field(ge=0)
    hit_rate_at_k: float = Field(ge=0, le=1)
    recall_at_k: float = Field(ge=0, le=1)
    precision_at_k: float = Field(ge=0, le=1)
    mrr_at_k: float = Field(ge=0, le=1)
    no_result_success_rate: float | None = Field(default=None, ge=0, le=1)
    results: list[RetrievalEvalCaseResult] = Field(default_factory=list)


class RagBadCaseCause(BaseModel):
    layer: RagBadCaseLayer
    severity: RagAnswerQualitySeverity
    code: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    evidence: str | None = None
    suggested_action: str = Field(min_length=1)


class RagBadCaseAnalysis(BaseModel):
    case_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    failed: bool
    primary_layer: RagBadCaseLayer | None = None
    causes: list[RagBadCaseCause] = Field(default_factory=list)


class RagBadCaseReport(BaseModel):
    analyzed_case_count: int = Field(ge=0)
    failed_case_count: int = Field(ge=0)
    blocking_cause_count: int = Field(ge=0)
    layer_counts: dict[str, int] = Field(default_factory=dict)
    analyses: list[RagBadCaseAnalysis] = Field(default_factory=list)


def load_retrieval_eval_cases(path: Path | str) -> list[RetrievalEvalCase]:
    raw_text = Path(path).read_text(encoding="utf-8")
    raw_cases = json.loads(raw_text)
    if not isinstance(raw_cases, list):
        raise ValueError("retrieval eval cases file must contain a JSON list")
    cases = [RetrievalEvalCase.model_validate(raw_case) for raw_case in raw_cases]
    _validate_unique_case_ids(cases)
    return cases


def build_retrieval_eval_cases_from_rag_cases(
    cases: Sequence[RagEvalCase],
    *,
    include_no_context: bool = True,
) -> list[RetrievalEvalCase]:
    retrieval_cases: list[RetrievalEvalCase] = []
    for eval_case in cases:
        behavior = eval_case.expectation.behavior
        if behavior == "answer":
            if not _has_expected_evidence(eval_case.expectation):
                continue
            retrieval_cases.append(
                RetrievalEvalCase(
                    id=eval_case.id,
                    query=eval_case.query,
                    expected_sources=eval_case.expectation.expected_sources,
                    expected_sections=eval_case.expectation.expected_sections,
                    expected_chunk_ids=eval_case.expectation.expected_chunk_ids,
                    permission_group=_first_or_none(
                        eval_case.access_context.permission_groups
                    ),
                    business_domain=_first_or_none(
                        eval_case.access_context.business_domains
                    ),
                    doc_type=_first_or_none(eval_case.access_context.doc_types),
                    source=_first_or_none(eval_case.access_context.sources),
                    notes=eval_case.notes,
                )
            )
        elif behavior == "no_context" and include_no_context:
            retrieval_cases.append(
                RetrievalEvalCase(
                    id=eval_case.id,
                    query=eval_case.query,
                    expect_no_results=True,
                    permission_group=_first_or_none(
                        eval_case.access_context.permission_groups
                    ),
                    business_domain=_first_or_none(
                        eval_case.access_context.business_domains
                    ),
                    doc_type=_first_or_none(eval_case.access_context.doc_types),
                    source=_first_or_none(eval_case.access_context.sources),
                    notes=eval_case.notes,
                )
            )
    _validate_unique_case_ids(retrieval_cases)
    return retrieval_cases


def evaluate_retrieval_case(
    eval_case: RetrievalEvalCase,
    retrieved_chunks: Sequence[RetrievedChunk],
    *,
    top_k: int,
) -> RetrievalEvalCaseResult:
    _validate_top_k(top_k)
    top_chunks = list(retrieved_chunks)[:top_k]

    if eval_case.expect_no_results:
        passed = not top_chunks
        return RetrievalEvalCaseResult(
            case_id=eval_case.id,
            query=eval_case.query,
            top_k=top_k,
            match_level="none",
            metric_applicable=False,
            expected_count=0,
            retrieved_count=len(top_chunks),
            relevant_retrieved_count=0,
            matched_expected_count=0,
            hit=passed,
            precision_at_k=1.0 if passed else 0.0,
            recall_at_k=1.0 if passed else 0.0,
            reciprocal_rank=1.0 if passed else 0.0,
            passed=passed,
            failed_reason=None if passed else "expected no results but retrieved chunks",
            retrieved_items=[
                _build_eval_item(
                    rank=index,
                    chunk=chunk,
                    relevant=False,
                )
                for index, chunk in enumerate(top_chunks, start=1)
            ],
        )

    matcher = _ExpectedMatcher.from_case(eval_case)
    retrieved_items: list[RetrievalEvalItem] = []
    matched_expected_keys: set[str] = set()
    first_relevant_rank: int | None = None
    relevant_retrieved_count = 0

    for rank, chunk in enumerate(top_chunks, start=1):
        matched_key = matcher.match(chunk)
        relevant = matched_key is not None
        if relevant:
            relevant_retrieved_count += 1
            matched_expected_keys.add(matched_key)
            if first_relevant_rank is None:
                first_relevant_rank = rank
        retrieved_items.append(
            _build_eval_item(
                rank=rank,
                chunk=chunk,
                relevant=relevant,
            )
        )

    matched_expected_count = len(matched_expected_keys)
    expected_count = matcher.expected_count
    hit = first_relevant_rank is not None
    recall_at_k = matched_expected_count / expected_count
    precision_at_k = relevant_retrieved_count / top_k
    reciprocal_rank = 0.0 if first_relevant_rank is None else 1 / first_relevant_rank
    passed = matched_expected_count == expected_count

    return RetrievalEvalCaseResult(
        case_id=eval_case.id,
        query=eval_case.query,
        top_k=top_k,
        match_level=matcher.match_level,
        metric_applicable=True,
        expected_count=expected_count,
        retrieved_count=len(top_chunks),
        relevant_retrieved_count=relevant_retrieved_count,
        matched_expected_count=matched_expected_count,
        hit=hit,
        first_relevant_rank=first_relevant_rank,
        precision_at_k=round(precision_at_k, 6),
        recall_at_k=round(recall_at_k, 6),
        reciprocal_rank=round(reciprocal_rank, 6),
        passed=passed,
        failed_reason=None if passed else "missing expected retrieval targets",
        retrieved_items=retrieved_items,
    )


def evaluate_retrieval_results(
    cases: Sequence[RetrievalEvalCase],
    retrievals_by_case_id: Mapping[str, Sequence[RetrievedChunk]],
    *,
    top_k: int,
) -> RetrievalEvalSummary:
    _validate_top_k(top_k)
    _validate_unique_case_ids(cases)
    results = [
        evaluate_retrieval_case(
            eval_case,
            retrievals_by_case_id.get(eval_case.id, []),
            top_k=top_k,
        )
        for eval_case in cases
    ]
    evaluated_results = [result for result in results if result.metric_applicable]
    no_result_results = [result for result in results if not result.metric_applicable]
    no_result_success_rate = (
        _average([1.0 if result.passed else 0.0 for result in no_result_results])
        if no_result_results
        else None
    )

    return RetrievalEvalSummary(
        top_k=top_k,
        case_count=len(results),
        evaluated_case_count=len(evaluated_results),
        no_result_case_count=len(no_result_results),
        passed_case_count=sum(1 for result in results if result.passed),
        failed_case_count=sum(1 for result in results if not result.passed),
        hit_rate_at_k=_average([1.0 if result.hit else 0.0 for result in evaluated_results]),
        recall_at_k=_average([result.recall_at_k for result in evaluated_results]),
        precision_at_k=_average([result.precision_at_k for result in evaluated_results]),
        mrr_at_k=_average([result.reciprocal_rank for result in evaluated_results]),
        no_result_success_rate=no_result_success_rate,
        results=results,
    )


def format_retrieval_eval_summary(summary: RetrievalEvalSummary) -> list[str]:
    lines = [
        "RAG retrieval evaluation summary",
        f"top_k: {summary.top_k}",
        f"cases: {summary.case_count}",
        f"evaluated_cases: {summary.evaluated_case_count}",
        f"no_result_cases: {summary.no_result_case_count}",
        f"passed_cases: {summary.passed_case_count}",
        f"failed_cases: {summary.failed_case_count}",
        f"hit_rate@{summary.top_k}: {summary.hit_rate_at_k:.4f}",
        f"recall@{summary.top_k}: {summary.recall_at_k:.4f}",
        f"precision@{summary.top_k}: {summary.precision_at_k:.4f}",
        f"mrr@{summary.top_k}: {summary.mrr_at_k:.4f}",
    ]
    if summary.no_result_success_rate is not None:
        lines.append(
            f"no_result_success_rate: {summary.no_result_success_rate:.4f}"
        )
    return lines


def format_retrieval_case_metric_breakdown(
    result: RetrievalEvalCaseResult,
) -> list[str]:
    lines = [
        f"Retrieval metric breakdown: {result.case_id}",
        f"top_k: {result.top_k}",
        f"match_level: {result.match_level}",
    ]
    if not result.metric_applicable:
        lines.extend(
            [
                "metric_applicable: false",
                f"expected_no_results_passed: {str(result.passed).lower()}",
                f"retrieved_count: {result.retrieved_count}",
            ]
        )
        return lines

    lines.extend(
        [
            f"hit@{result.top_k}: {1 if result.hit else 0} "
            f"(first_relevant_rank={result.first_relevant_rank or '-'})",
            f"recall@{result.top_k}: {result.matched_expected_count}/"
            f"{result.expected_count} = {result.recall_at_k:.6f}",
            f"precision@{result.top_k}: {result.relevant_retrieved_count}/"
            f"{result.top_k} = {result.precision_at_k:.6f}",
            f"mrr@{result.top_k}: {result.reciprocal_rank:.6f}",
        ]
    )
    return lines


def format_retrieval_bad_cases(summary: RetrievalEvalSummary) -> list[str]:
    bad_cases = [result for result in summary.results if not result.passed]
    if not bad_cases:
        return ["No bad cases."]

    lines = ["Bad cases:"]
    for result in bad_cases:
        lines.append(
            f"- {result.case_id}: recall@{result.top_k}={result.recall_at_k:.4f} "
            f"mrr@{result.top_k}={result.reciprocal_rank:.4f} "
            f"reason={result.failed_reason}"
        )
        for item in result.retrieved_items:
            marker = "relevant" if item.relevant else "noise"
            lines.append(
                f"  {item.rank}. {marker} score={item.score:.4f} "
                f"source={item.source or '-'} section={item.section or '-'} "
                f"chunk_id={item.chunk_id}"
            )
    return lines


def analyze_rag_bad_case(
    *,
    retrieval_result: RetrievalEvalCaseResult | None = None,
    answer_quality_result: RagAnswerQualityResult | None = None,
) -> RagBadCaseAnalysis:
    if retrieval_result is None and answer_quality_result is None:
        raise ValueError("bad case analysis requires at least one evaluation result")

    case_id = (
        answer_quality_result.case_id
        if answer_quality_result is not None
        else retrieval_result.case_id
    )
    query = (
        answer_quality_result.query
        if answer_quality_result is not None
        else retrieval_result.query
    )
    causes: list[RagBadCaseCause] = []
    if retrieval_result is not None and (
        not retrieval_result.passed
        or _has_retrieval_quality_warning(retrieval_result)
    ):
        causes.extend(_analyze_retrieval_bad_case(retrieval_result))
    if answer_quality_result is not None and (
        not answer_quality_result.passed or answer_quality_result.findings
    ):
        causes.extend(_analyze_answer_quality_bad_case(answer_quality_result))

    return RagBadCaseAnalysis(
        case_id=case_id,
        query=query,
        failed=bool(causes),
        primary_layer=_primary_bad_case_layer(causes),
        causes=causes,
    )


def analyze_rag_bad_cases(
    *,
    retrieval_summary: RetrievalEvalSummary | None = None,
    answer_quality_summary: RagAnswerQualitySummary | None = None,
) -> RagBadCaseReport:
    retrieval_results = {
        result.case_id: result
        for result in (retrieval_summary.results if retrieval_summary else [])
    }
    answer_results = {
        result.case_id: result
        for result in (answer_quality_summary.results if answer_quality_summary else [])
    }
    case_ids = sorted(set(retrieval_results) | set(answer_results))
    analyses = [
        analyze_rag_bad_case(
            retrieval_result=retrieval_results.get(case_id),
            answer_quality_result=answer_results.get(case_id),
        )
        for case_id in case_ids
    ]
    failed_analyses = [analysis for analysis in analyses if analysis.failed]
    all_causes = [
        cause for analysis in failed_analyses for cause in analysis.causes
    ]
    layer_counts = Counter(cause.layer for cause in all_causes)

    return RagBadCaseReport(
        analyzed_case_count=len(analyses),
        failed_case_count=len(failed_analyses),
        blocking_cause_count=sum(
            1 for cause in all_causes if cause.severity == "blocking"
        ),
        layer_counts=dict(sorted(layer_counts.items())),
        analyses=analyses,
    )


def format_rag_bad_case_report(report: RagBadCaseReport) -> list[str]:
    lines = [
        "RAG bad case analysis report",
        f"analyzed_cases: {report.analyzed_case_count}",
        f"failed_cases: {report.failed_case_count}",
        f"blocking_causes: {report.blocking_cause_count}",
        f"layers: {_format_counts(report.layer_counts)}",
    ]
    for analysis in report.analyses:
        if not analysis.failed:
            continue
        lines.append(
            f"- {analysis.case_id}: primary_layer={analysis.primary_layer} "
            f"causes={len(analysis.causes)}"
        )
        for cause in analysis.causes:
            lines.append(
                f"  {cause.severity} {cause.layer} code={cause.code} "
                f"evidence={cause.evidence or '-'} action={cause.suggested_action}"
            )
    return lines


class _ExpectedMatcher:
    def __init__(
        self,
        *,
        match_level: RetrievalMatchLevel,
        expected_keys: set[str],
        expected_sources: set[str],
    ) -> None:
        self.match_level = match_level
        self.expected_keys = expected_keys
        self.expected_sources = expected_sources

    @classmethod
    def from_case(cls, eval_case: RetrievalEvalCase) -> "_ExpectedMatcher":
        if eval_case.expected_chunk_ids:
            return cls(
                match_level="chunk_id",
                expected_keys=set(eval_case.expected_chunk_ids),
                expected_sources=set(eval_case.expected_sources),
            )
        if eval_case.expected_sections:
            return cls(
                match_level="section",
                expected_keys=set(eval_case.expected_sections),
                expected_sources=set(eval_case.expected_sources),
            )
        return cls(
            match_level="source",
            expected_keys=set(eval_case.expected_sources),
            expected_sources=set(eval_case.expected_sources),
        )

    @property
    def expected_count(self) -> int:
        return len(self.expected_keys)

    def match(self, chunk: RetrievedChunk) -> str | None:
        source = _metadata_string(chunk.metadata, "source")
        if self.expected_sources and source not in self.expected_sources:
            return None

        if self.match_level == "chunk_id":
            return chunk.chunk_id if chunk.chunk_id in self.expected_keys else None
        if self.match_level == "section":
            section = _metadata_string(chunk.metadata, "section")
            return section if section in self.expected_keys else None
        if self.match_level == "source":
            return source if source in self.expected_keys else None
        return None


def _build_eval_item(
    *,
    rank: int,
    chunk: RetrievedChunk,
    relevant: bool,
) -> RetrievalEvalItem:
    return RetrievalEvalItem(
        rank=rank,
        chunk_id=chunk.chunk_id,
        source=_metadata_string(chunk.metadata, "source"),
        section=_metadata_string(chunk.metadata, "section"),
        score=chunk.score,
        relevant=relevant,
    )


def _metadata_string(metadata: Mapping[str, object], key: str) -> str | None:
    value = metadata.get(key)
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _validate_top_k(top_k: int) -> None:
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")


def _validate_unique_case_ids(cases: Sequence[RetrievalEvalCase]) -> None:
    seen: set[str] = set()
    for eval_case in cases:
        if eval_case.id in seen:
            raise ValueError("retrieval eval case ids must be unique")
        seen.add(eval_case.id)


def _validate_unique_rag_eval_case_ids(cases: Sequence[RagEvalCase]) -> None:
    seen: set[str] = set()
    for eval_case in cases:
        if eval_case.id in seen:
            raise ValueError("RAG eval case ids must be unique")
        seen.add(eval_case.id)


def _first_or_none(values: Sequence[str]) -> str | None:
    return values[0] if values else None


def _has_expected_evidence(expectation: RagEvalExpectation) -> bool:
    return any(
        (
            expectation.expected_sources,
            expectation.expected_sections,
            expectation.expected_chunk_ids,
        )
    )


def _infer_actual_rag_behavior(
    rag_answer: RagAnswer,
    actual_refusal_reason_codes: Sequence[str],
) -> str:
    reason_codes = set(actual_refusal_reason_codes)
    if "PROMPT_INJECTION" in reason_codes:
        return "security_block"
    if "ACCESS_DENIED" in reason_codes:
        return "access_denied"
    if "CLARIFY" in reason_codes or "NEED_CLARIFICATION" in reason_codes:
        return "clarify"
    if "NO_CONTEXT" in reason_codes or rag_answer.status is RagAnswerStatus.NO_CONTEXT:
        return "no_context"
    return "answer"


def _match_answer_points(
    answer: str,
    expected_points: Sequence[str],
) -> tuple[list[str], list[str]]:
    normalized_answer = _normalize_text_for_quality_match(answer)
    matched_points: list[str] = []
    missing_points: list[str] = []
    for point in expected_points:
        normalized_point = _normalize_text_for_quality_match(point)
        if normalized_point and normalized_point in normalized_answer:
            matched_points.append(point)
        else:
            missing_points.append(point)
    return matched_points, missing_points


def _is_citation_passed(
    *,
    expectation: RagEvalExpectation,
    rag_answer: RagAnswer,
    missing_sources: Sequence[str],
    forbidden_sources_used: Sequence[str],
) -> bool:
    if forbidden_sources_used:
        return False
    if expectation.behavior != "answer":
        return not rag_answer.citations
    if expectation.citation_required and not rag_answer.citations:
        return False
    if missing_sources:
        return False
    return True


def _build_answer_quality_findings(
    *,
    expectation: RagEvalExpectation,
    actual_behavior: str,
    behavior_passed: bool,
    missing_answer_points: Sequence[str],
    rag_answer: RagAnswer,
    missing_sources: Sequence[str],
    unexpected_sources: Sequence[str],
    forbidden_sources_used: Sequence[str],
    expected_refusal_codes: Sequence[str],
    actual_refusal_codes: Sequence[str],
    refusal_passed: bool,
) -> list[RagAnswerQualityFinding]:
    findings: list[RagAnswerQualityFinding] = []

    if not behavior_passed:
        findings.append(
            RagAnswerQualityFinding(
                code="RAG_ANSWER_BEHAVIOR_MISMATCH",
                dimension="behavior",
                severity="blocking",
                message="Actual RAG behavior does not match expected behavior.",
                evidence=f"expected={expectation.behavior} actual={actual_behavior}",
            )
        )

    if expectation.behavior == "answer" and missing_answer_points:
        for point in missing_answer_points:
            findings.append(
                RagAnswerQualityFinding(
                    code="RAG_ANSWER_POINT_MISSING",
                    dimension="answer_points",
                    severity="blocking",
                    message="The answer missed an expected answer point.",
                    evidence=point,
                )
            )

    if expectation.citation_required and expectation.behavior == "answer" and not rag_answer.citations:
        findings.append(
            RagAnswerQualityFinding(
                code="RAG_ANSWER_CITATION_REQUIRED_BUT_MISSING",
                dimension="citation",
                severity="blocking",
                message="The answer should include citations but returned none.",
            )
        )

    for source in missing_sources:
        findings.append(
            RagAnswerQualityFinding(
                code="RAG_ANSWER_EXPECTED_SOURCE_MISSING",
                dimension="citation",
                severity="blocking",
                message="The answer citations missed an expected source.",
                evidence=source,
            )
        )

    for source in unexpected_sources:
        findings.append(
            RagAnswerQualityFinding(
                code="RAG_ANSWER_UNEXPECTED_SOURCE",
                dimension="citation",
                severity="warning",
                message="The answer cited a source outside the expected source list.",
                evidence=source,
            )
        )

    for source in forbidden_sources_used:
        findings.append(
            RagAnswerQualityFinding(
                code="RAG_ANSWER_FORBIDDEN_SOURCE_USED",
                dimension="citation",
                severity="blocking",
                message="The answer cited a forbidden source.",
                evidence=source,
            )
        )

    if expectation.behavior != "answer" and rag_answer.citations:
        findings.append(
            RagAnswerQualityFinding(
                code="RAG_REFUSAL_HAS_CITATIONS",
                dimension="refusal",
                severity="blocking",
                message="A refusal response should not return knowledge citations.",
            )
        )

    if expectation.behavior != "answer" and not refusal_passed:
        missing_codes = [
            code for code in expected_refusal_codes if code not in set(actual_refusal_codes)
        ]
        findings.append(
            RagAnswerQualityFinding(
                code="RAG_REFUSAL_REASON_MISSING",
                dimension="refusal",
                severity="blocking",
                message="The refusal response missed expected reason codes.",
                evidence=", ".join(missing_codes),
            )
        )

    return findings


def _analyze_retrieval_bad_case(
    result: RetrievalEvalCaseResult,
) -> list[RagBadCaseCause]:
    if not result.metric_applicable:
        return [
            _bad_case_cause(
                layer="retrieval",
                severity="blocking",
                code="RAG_BAD_CASE_NO_CONTEXT_RETRIEVED_RESULTS",
                reason="A no-context case still returned retrieved chunks.",
                evidence=f"retrieved_count={result.retrieved_count}",
                suggested_action=(
                    "Check score_threshold, intent routing, metadata filters, and "
                    "no-context decision rules."
                ),
            )
        ]

    causes: list[RagBadCaseCause] = []
    if result.recall_at_k == 0:
        causes.append(
            _bad_case_cause(
                layer="retrieval",
                severity="blocking",
                code="RAG_BAD_CASE_RECALL_ZERO",
                reason="No expected retrieval target appeared in the evaluated top_k.",
                evidence=(
                    f"top_k={result.top_k} expected={result.expected_count} "
                    f"matched={result.matched_expected_count}"
                ),
                suggested_action=(
                    "Check source data, chunking, embeddings, query rewrite, "
                    "multi-query expansion, and metadata filters."
                ),
            )
        )
    elif result.recall_at_k < 1:
        causes.append(
            _bad_case_cause(
                layer="retrieval",
                severity="blocking",
                code="RAG_BAD_CASE_PARTIAL_RECALL",
                reason="Only part of the expected retrieval targets appeared.",
                evidence=f"recall={result.recall_at_k:.4f}",
                suggested_action=(
                    "Check whether top_k is too small, chunking split related facts, "
                    "or filters removed some expected evidence."
                ),
            )
        )

    if result.first_relevant_rank and result.first_relevant_rank > 1:
        causes.append(
            _bad_case_cause(
                layer="ranking",
                severity="warning",
                code="RAG_BAD_CASE_RELEVANT_RESULT_NOT_TOP1",
                reason="The first relevant chunk appeared after rank 1.",
                evidence=f"first_relevant_rank={result.first_relevant_rank}",
                suggested_action=(
                    "Check rerank, hybrid fusion weights, score normalization, "
                    "and query rewrite quality."
                ),
            )
        )
    if result.precision_at_k < 0.5 and result.retrieved_count:
        causes.append(
            _bad_case_cause(
                layer="retrieval",
                severity="warning",
                code="RAG_BAD_CASE_LOW_PRECISION",
                reason="The evaluated top_k contains more noise than useful evidence.",
                evidence=f"precision={result.precision_at_k:.4f}",
                suggested_action=(
                    "Check score_threshold, top_k, hybrid weights, metadata filters, "
                    "and rerank behavior."
                ),
            )
        )
    return causes


def _has_retrieval_quality_warning(result: RetrievalEvalCaseResult) -> bool:
    if not result.metric_applicable:
        return not result.passed
    return bool(
        (result.first_relevant_rank and result.first_relevant_rank > 1)
        or (result.precision_at_k < 0.5 and result.retrieved_count)
    )


def _analyze_answer_quality_bad_case(
    result: RagAnswerQualityResult,
) -> list[RagBadCaseCause]:
    causes: list[RagBadCaseCause] = []
    for finding in result.findings:
        layer = _layer_from_answer_quality_finding(result, finding)
        causes.append(
            _bad_case_cause(
                layer=layer,
                severity=finding.severity,
                code=finding.code,
                reason=finding.message,
                evidence=finding.evidence,
                suggested_action=_suggest_action_for_answer_quality_layer(layer),
            )
        )
    return causes


def _layer_from_answer_quality_finding(
    result: RagAnswerQualityResult,
    finding: RagAnswerQualityFinding,
) -> RagBadCaseLayer:
    if finding.code == "RAG_ANSWER_FORBIDDEN_SOURCE_USED":
        return "access_control"
    if finding.dimension == "behavior":
        if result.expected_behavior == "security_block" or result.actual_behavior == "security_block":
            return "security"
        if result.expected_behavior == "access_denied" or result.actual_behavior == "access_denied":
            return "access_control"
        if result.expected_behavior in {"no_context", "clarify"}:
            return "refusal"
        return "generation"
    if finding.dimension == "answer_points":
        return "generation"
    if finding.dimension == "citation":
        return "citation"
    if finding.dimension == "refusal":
        if result.expected_behavior == "security_block":
            return "security"
        if result.expected_behavior == "access_denied":
            return "access_control"
        return "refusal"
    return "unknown"


def _suggest_action_for_answer_quality_layer(layer: RagBadCaseLayer) -> str:
    suggestions: dict[str, str] = {
        "generation": "Check prompt rules, context compression, answer format, and model output parsing.",
        "citation": "Check citation construction, retrieved context ordering, source metadata, and citation verification.",
        "refusal": "Check no-context, clarification, and refusal reason mapping logic.",
        "access_control": "Check access scope, metadata filters, forbidden sources, and permission propagation.",
        "security": "Check prompt-injection detection, risk levels, and security blocking policy.",
    }
    return suggestions.get(layer, "Check the upstream evaluation result and related RAG chain step.")


def _primary_bad_case_layer(
    causes: Sequence[RagBadCaseCause],
) -> RagBadCaseLayer | None:
    if not causes:
        return None
    blocking_causes = [cause for cause in causes if cause.severity == "blocking"]
    selected_causes = blocking_causes or list(causes)
    priority = [
        "security",
        "access_control",
        "retrieval",
        "ranking",
        "generation",
        "citation",
        "refusal",
        "data",
        "evaluation",
        "unknown",
    ]
    for layer in priority:
        if any(cause.layer == layer for cause in selected_causes):
            return layer
    return selected_causes[0].layer


def _bad_case_cause(
    *,
    layer: RagBadCaseLayer,
    severity: RagAnswerQualitySeverity,
    code: str,
    reason: str,
    evidence: str | None,
    suggested_action: str,
) -> RagBadCaseCause:
    return RagBadCaseCause(
        layer=layer,
        severity=severity,
        code=code,
        reason=reason,
        evidence=evidence,
        suggested_action=suggested_action,
    )


def _unique_strings(values: Iterable[object]) -> list[str]:
    unique_values: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        normalized = value.strip()
        if normalized and normalized not in unique_values:
            unique_values.append(normalized)
    return unique_values


def _normalize_text_for_quality_match(text: str) -> str:
    return "".join(char.lower() for char in text if char.isalnum())


def _average(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 6)


def _ratio(numerator: int, denominator: int, *, default: float = 0.0) -> float:
    if denominator <= 0:
        return default
    return round(numerator / denominator, 6)


def _normalize_string_list(value: object, *, field_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be a list of strings")
    normalized_values: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{field_name} must contain non-blank strings")
        normalized = item.strip()
        if normalized not in normalized_values:
            normalized_values.append(normalized)
    return normalized_values


def _format_counts(counts: Mapping[str, int]) -> str:
    if not counts:
        return "-"
    return ", ".join(f"{name}={count}" for name, count in counts.items())
