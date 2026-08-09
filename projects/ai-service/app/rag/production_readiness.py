from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field


RagProductionReadinessCategory = Literal[
    "quality",
    "security",
    "performance",
    "cost",
    "observability",
    "data",
    "agent_boundary",
]
RagProductionReadinessStatus = Literal[
    "passed",
    "warning",
    "failed",
    "not_checked",
]
RagProductionReleaseStatus = Literal["ready", "conditional", "blocked"]


class RagProductionReadinessCheck(BaseModel):
    check_id: str = Field(min_length=1)
    category: RagProductionReadinessCategory
    title: str = Field(min_length=1)
    requirement: str = Field(min_length=1)
    evidence_examples: list[str] = Field(default_factory=list)
    risk_if_missing: str = Field(min_length=1)
    release_blocker: bool = False


class RagProductionReadinessAnswer(BaseModel):
    check_id: str = Field(min_length=1)
    status: RagProductionReadinessStatus
    evidence: list[str] = Field(default_factory=list)
    notes: str | None = None


class RagProductionReadinessFinding(BaseModel):
    check_id: str = Field(min_length=1)
    category: RagProductionReadinessCategory
    title: str = Field(min_length=1)
    status: RagProductionReadinessStatus
    release_blocker: bool
    evidence: list[str] = Field(default_factory=list)
    recommendation: str = Field(min_length=1)
    risk_if_missing: str = Field(min_length=1)


class RagProductionReadinessReport(BaseModel):
    release_status: RagProductionReleaseStatus
    checklist_count: int = Field(ge=0)
    passed_count: int = Field(ge=0)
    warning_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    not_checked_count: int = Field(ge=0)
    blocker_count: int = Field(ge=0)
    category_status_counts: dict[str, dict[str, int]] = Field(default_factory=dict)
    blocker_check_ids: list[str] = Field(default_factory=list)
    findings: list[RagProductionReadinessFinding] = Field(default_factory=list)


def default_rag_production_readiness_checklist() -> list[RagProductionReadinessCheck]:
    return [
        _check(
            "quality.retrieval_metrics",
            "quality",
            "Retrieval metrics are measured",
            "Hit@K, Recall@K, Precision@K, and MRR@K have been measured on representative cases.",
            ["retrieval metric report", "per-case metric breakdown"],
            "RAG can miss relevant knowledge without anyone noticing.",
            blocker=True,
        ),
        _check(
            "quality.answer_quality",
            "quality",
            "Answer quality is evaluated",
            "Final answers are checked against expected behavior, answer points, citations, and refusal rules.",
            ["answer quality summary", "bad case list"],
            "A system can retrieve correctly but still answer incorrectly.",
            blocker=True,
        ),
        _check(
            "quality.citation_verification",
            "quality",
            "Citations are verified",
            "Answered responses have valid citations that point back to retrieved chunks.",
            ["citation verification report", "missing citation count"],
            "Users may trust unsupported or fake sources.",
            blocker=True,
        ),
        _check(
            "quality.bad_case_process",
            "quality",
            "Bad cases have owners and actions",
            "Retrieval, ranking, generation, citation, refusal, access-control, and security bad cases are classified.",
            ["bad case report", "tuning recommendations"],
            "Teams may tune the wrong layer and fail to improve quality.",
        ),
        _check(
            "security.permission_filter",
            "security",
            "Permission filters are enforced",
            "Tenant, permission group, business domain, doc type, visibility, and status filters are applied before retrieval.",
            ["payload filter report", "access scope tests"],
            "Users can receive knowledge they are not allowed to see.",
            blocker=True,
        ),
        _check(
            "security.prompt_injection",
            "security",
            "RAG prompt injection is checked",
            "Retrieved content and model-visible metadata are scanned for prompt-injection risks before generation.",
            ["security report", "blocked reason codes"],
            "Knowledge-base text can override instructions or induce unsafe tool usage.",
            blocker=True,
        ),
        _check(
            "security.safe_logging",
            "security",
            "Logs do not expose sensitive content",
            "Safe log payloads avoid raw chunk content and redact sensitive query previews.",
            ["safe observability payload", "logging field review"],
            "Logs can become a secondary sensitive data store.",
            blocker=True,
        ),
        _check(
            "security.tool_boundaries",
            "security",
            "Tool boundaries are enforced",
            "Read, write, and sensitive tools are separated, with write confirmation and disabled sensitive tools blocked.",
            ["tool registry", "agent boundary report"],
            "The model may execute risky business actions without proper authorization.",
            blocker=True,
        ),
        _check(
            "performance.timeouts",
            "performance",
            "Stage timeouts are defined",
            "Embedding, vector store, rerank, generation, and security stages have timeout budgets.",
            ["operation timing report", "near-timeout/timed-out stages"],
            "Slow dependencies can exhaust workers or make the whole service unstable.",
            blocker=True,
        ),
        _check(
            "performance.degradation",
            "performance",
            "Degradation behavior is explicit",
            "The service can choose safe cached retrieval, safe fallback, or no-context responses when dependencies fail.",
            ["degradation decision", "fallback tests"],
            "Failures may turn into long waits, blank responses, or unsupported model guesses.",
            blocker=True,
        ),
        _check(
            "performance.cache_safety",
            "performance",
            "Cache keys are scoped safely",
            "Retrieval cache keys include query hash, retrieval parameters, permission scope, model, and collection information.",
            ["cache key components", "cache hit-rate report"],
            "Cached chunks may be reused across tenants, permissions, or stale collection versions.",
            blocker=True,
        ),
        _check(
            "cost.token_budget",
            "cost",
            "Token and candidate budgets are controlled",
            "top_k, rerank candidates, context budget, and generation length are bounded.",
            ["context compression report", "rerank candidate count"],
            "Costs can grow quickly and latency can become unpredictable.",
        ),
        _check(
            "cost.provider_fallback",
            "cost",
            "Provider fallback and retry cost are reviewed",
            "Rerank and model fallback rules are measured so retries do not silently multiply cost.",
            ["rerank execution result", "provider fallback count"],
            "Fallbacks and retries can hide provider failures while increasing cost.",
        ),
        _check(
            "observability.rag_event",
            "observability",
            "RAG observability event exists",
            "query, retrieval, rerank, citation, timing, and warning snapshots are recorded for each important request.",
            ["RagObservabilityEvent", "safe log payload"],
            "Online failures cannot be diagnosed or turned into bad cases.",
            blocker=True,
        ),
        _check(
            "observability.warning_codes",
            "observability",
            "Warning codes are machine-readable",
            "No-context, fallback, citation invalid, near-timeout, and timed-out states have stable warning codes.",
            ["warning code list", "log query examples"],
            "Operational issues become hard to aggregate and alert on.",
        ),
        _check(
            "data.update_plan",
            "data",
            "Data update plan is defined",
            "New, modified, deleted, and unchanged sources map to ingest, refresh, delete, skip, or full reindex actions.",
            ["data update plan", "document manifest diff"],
            "Old chunks, duplicated chunks, and deleted documents can keep affecting answers.",
            blocker=True,
        ),
        _check(
            "data.metadata_validation",
            "data",
            "Document metadata is validated",
            "source, doc type, business domain, permission group, chunk id, chunk count, and chunk size are validated.",
            ["metadata validation tests", "payload schema review"],
            "Filters, citations, routing, and updates can break because metadata is missing or inconsistent.",
            blocker=True,
        ),
        _check(
            "data.reindex_policy",
            "data",
            "Full reindex policy is documented",
            "Embedding model, dimension, chunking, metadata schema, metric, or index changes trigger full reindex decisions.",
            ["full reindex checklist", "collection version policy"],
            "Global indexing changes may leave mixed old and new vector semantics in one collection.",
        ),
        _check(
            "agent_boundary.owner_decision",
            "agent_boundary",
            "RAG, Agent, Tool owners are explicit",
            "Policy/process questions, order lookup, ticket creation, safety, smalltalk, and clarification are routed to clear owners.",
            ["RagAgentBoundaryDecision", "boundary tests"],
            "Requests may be handled by the wrong subsystem, causing wrong answers or unsafe actions.",
            blocker=True,
        ),
        _check(
            "agent_boundary.write_confirmation",
            "agent_boundary",
            "Write operations require confirmation",
            "Agent workflows collect fields and request user confirmation before write tools execute.",
            ["create_ticket confirmation test", "tool registry review"],
            "The system may create or modify business records without user approval.",
            blocker=True,
        ),
        _check(
            "agent_boundary.rag_as_context",
            "agent_boundary",
            "RAG-as-Agent-context boundary is clear",
            "Agent workflows may use RAG as evidence, but RAG does not own workflow state or write execution.",
            ["agent boundary report", "LangGraph node design"],
            "RAG and Agent responsibilities can blur, making failures hard to reason about.",
        ),
    ]


def build_rag_production_readiness_report(
    answers: Sequence[RagProductionReadinessAnswer],
    *,
    checklist: Sequence[RagProductionReadinessCheck] | None = None,
) -> RagProductionReadinessReport:
    checks = list(checklist or default_rag_production_readiness_checklist())
    _validate_unique_check_ids(checks)
    answer_by_id = _answers_by_id(answers)
    unknown_ids = sorted(set(answer_by_id) - {check.check_id for check in checks})
    if unknown_ids:
        raise ValueError(f"unknown readiness check ids: {', '.join(unknown_ids)}")

    findings: list[RagProductionReadinessFinding] = []
    status_counts: Counter[str] = Counter()
    category_status_counts: dict[str, Counter[str]] = defaultdict(Counter)
    blocker_check_ids: list[str] = []

    for check in checks:
        answer = answer_by_id.get(check.check_id)
        status: RagProductionReadinessStatus = answer.status if answer else "not_checked"
        evidence = answer.evidence if answer else []
        status_counts[status] += 1
        category_status_counts[check.category][status] += 1
        if check.release_blocker and status in {"failed", "not_checked"}:
            blocker_check_ids.append(check.check_id)
        if status != "passed":
            findings.append(
                RagProductionReadinessFinding(
                    check_id=check.check_id,
                    category=check.category,
                    title=check.title,
                    status=status,
                    release_blocker=check.release_blocker,
                    evidence=evidence,
                    recommendation=_recommendation_for(check, status),
                    risk_if_missing=check.risk_if_missing,
                )
            )

    release_status = _release_status(
        blocker_count=len(blocker_check_ids),
        status_counts=status_counts,
    )
    return RagProductionReadinessReport(
        release_status=release_status,
        checklist_count=len(checks),
        passed_count=status_counts["passed"],
        warning_count=status_counts["warning"],
        failed_count=status_counts["failed"],
        not_checked_count=status_counts["not_checked"],
        blocker_count=len(blocker_check_ids),
        category_status_counts={
            category: dict(sorted(counts.items()))
            for category, counts in sorted(category_status_counts.items())
        },
        blocker_check_ids=blocker_check_ids,
        findings=findings,
    )


def format_rag_production_readiness_report(
    report: RagProductionReadinessReport,
) -> list[str]:
    lines = [
        "RAG production readiness report",
        f"release_status: {report.release_status}",
        f"checks: {report.checklist_count}",
        (
            "status_counts: "
            f"passed={report.passed_count} warning={report.warning_count} "
            f"failed={report.failed_count} not_checked={report.not_checked_count}"
        ),
        f"blockers: {report.blocker_count}",
    ]
    if report.blocker_check_ids:
        lines.append(f"blocker_check_ids: {', '.join(report.blocker_check_ids)}")
    for category, counts in report.category_status_counts.items():
        lines.append(f"category {category}: {counts}")
    for finding in report.findings:
        lines.append(
            (
                f"- {finding.status} {finding.check_id} "
                f"blocker={finding.release_blocker} "
                f"recommendation={finding.recommendation}"
            )
        )
    return lines


def _check(
    check_id: str,
    category: RagProductionReadinessCategory,
    title: str,
    requirement: str,
    evidence_examples: list[str],
    risk_if_missing: str,
    *,
    blocker: bool = False,
) -> RagProductionReadinessCheck:
    return RagProductionReadinessCheck(
        check_id=check_id,
        category=category,
        title=title,
        requirement=requirement,
        evidence_examples=evidence_examples,
        risk_if_missing=risk_if_missing,
        release_blocker=blocker,
    )


def _validate_unique_check_ids(
    checks: Sequence[RagProductionReadinessCheck],
) -> None:
    duplicated = [
        check_id
        for check_id, count in Counter(check.check_id for check in checks).items()
        if count > 1
    ]
    if duplicated:
        raise ValueError(f"duplicated readiness check ids: {', '.join(sorted(duplicated))}")


def _answers_by_id(
    answers: Sequence[RagProductionReadinessAnswer],
) -> dict[str, RagProductionReadinessAnswer]:
    answer_by_id: dict[str, RagProductionReadinessAnswer] = {}
    duplicated: list[str] = []
    for answer in answers:
        if answer.check_id in answer_by_id:
            duplicated.append(answer.check_id)
        answer_by_id[answer.check_id] = answer
    if duplicated:
        raise ValueError(f"duplicated readiness answers: {', '.join(sorted(duplicated))}")
    return answer_by_id


def _release_status(
    *,
    blocker_count: int,
    status_counts: Counter[str],
) -> RagProductionReleaseStatus:
    if blocker_count > 0:
        return "blocked"
    if status_counts["failed"] or status_counts["warning"] or status_counts["not_checked"]:
        return "conditional"
    return "ready"


def _recommendation_for(
    check: RagProductionReadinessCheck,
    status: RagProductionReadinessStatus,
) -> str:
    if status == "not_checked":
        return "Collect evidence before release."
    if status == "failed" and check.release_blocker:
        return "Fix this blocker before release."
    if status == "failed":
        return "Fix the issue or explicitly accept the residual risk."
    if status == "warning":
        return "Review the evidence and decide whether the residual risk is acceptable."
    return "No action required."
