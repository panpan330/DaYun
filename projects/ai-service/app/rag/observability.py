from collections import Counter
from collections.abc import Sequence
import hashlib
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.rag.citation_verification import (
    CitationFindingSeverity,
    CitationVerificationReport,
)
from app.rag.documents import RetrievedChunk
from app.rag.performance import RagOperationTiming
from app.rag.rerank import RerankExecutionResult, RerankReport, RerankedChunk


DEFAULT_QUERY_PREVIEW_CHARS = 120
DEFAULT_OBSERVED_CHUNK_LIMIT = 10

EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE_PATTERN = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
SECRET_PATTERN = re.compile(r"\b(?:sk|ak|api)[-_][A-Za-z0-9._-]{8,}\b")

RagObservedStageStatus = Literal["ok", "near_timeout", "timed_out"]


class RagObservedQuery(BaseModel):
    query_hash: str = Field(min_length=16)
    query_preview: str = Field(min_length=1)
    rewritten_query_hash: str | None = None
    rewritten_query_preview: str | None = None
    expanded_query_count: int = Field(ge=0)
    expanded_query_hashes: list[str] = Field(default_factory=list)


class RagObservedRetrievedChunk(BaseModel):
    rank: int = Field(ge=1)
    chunk_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    title: str | None = None
    section: str | None = None
    retrieval_score: float
    content_hash: str = Field(min_length=16)
    content_chars: int = Field(ge=0)


class RagObservedRetrieval(BaseModel):
    requested_top_k: int = Field(gt=0)
    returned_count: int = Field(ge=0)
    observed_count: int = Field(ge=0)
    top_chunk_id: str | None = None
    top_score: float | None = None
    source_counts: dict[str, int] = Field(default_factory=dict)
    chunks: list[RagObservedRetrievedChunk] = Field(default_factory=list)


class RagObservedRerankedChunk(BaseModel):
    chunk_id: str = Field(min_length=1)
    original_rank: int = Field(ge=1)
    rerank_rank: int = Field(ge=1)
    retrieval_score: float | None = None
    rerank_score: float = Field(ge=0, le=1)


class RagObservedRerank(BaseModel):
    candidate_count: int = Field(ge=0)
    returned_count: int = Field(ge=0)
    top_before_chunk_id: str | None = None
    top_after_chunk_id: str | None = None
    moved_count: int = Field(ge=0)
    used_fallback: bool = False
    fallback_reason: str | None = None
    elapsed_ms: float | None = Field(default=None, ge=0)
    chunks: list[RagObservedRerankedChunk] = Field(default_factory=list)


class RagObservedCitation(BaseModel):
    answer_status: str = Field(min_length=1)
    is_valid: bool
    retrieved_chunk_count: int = Field(ge=0)
    checked_citation_count: int = Field(ge=0)
    cited_chunk_count: int = Field(ge=0)
    missing_citation_count: int = Field(ge=0)
    answer_support_score: float = Field(ge=0, le=1)
    blocking_finding_count: int = Field(ge=0)
    warning_finding_count: int = Field(ge=0)
    finding_codes: list[str] = Field(default_factory=list)


class RagObservedTiming(BaseModel):
    stage: str = Field(min_length=1)
    elapsed_ms: float = Field(ge=0)
    timeout_seconds: float = Field(gt=0)
    status: RagObservedStageStatus


class RagObservabilityEvent(BaseModel):
    trace_id: str = Field(min_length=1)
    query: RagObservedQuery
    retrieval: RagObservedRetrieval
    rerank: RagObservedRerank | None = None
    citation: RagObservedCitation | None = None
    timings: list[RagObservedTiming] = Field(default_factory=list)
    total_elapsed_ms: float | None = Field(default=None, ge=0)
    warning_codes: list[str] = Field(default_factory=list)


def build_rag_observability_event(
    *,
    trace_id: str,
    user_query: str,
    retrieved_chunks: Sequence[RetrievedChunk],
    requested_top_k: int,
    rewritten_query: str | None = None,
    expanded_queries: Sequence[str] = (),
    rerank_report: RerankReport | None = None,
    rerank_execution: RerankExecutionResult | None = None,
    citation_report: CitationVerificationReport | None = None,
    timings: Sequence[RagOperationTiming] = (),
    total_elapsed_ms: float | None = None,
    observed_chunk_limit: int = DEFAULT_OBSERVED_CHUNK_LIMIT,
) -> RagObservabilityEvent:
    normalized_trace_id = trace_id.strip()
    if not normalized_trace_id:
        raise ValueError("trace_id must not be blank")
    normalized_query = user_query.strip()
    if not normalized_query:
        raise ValueError("user_query must not be blank")
    if requested_top_k <= 0:
        raise ValueError("requested_top_k must be greater than 0")
    if observed_chunk_limit <= 0:
        raise ValueError("observed_chunk_limit must be greater than 0")
    if total_elapsed_ms is not None and total_elapsed_ms < 0:
        raise ValueError("total_elapsed_ms must be greater than or equal to 0")

    event = RagObservabilityEvent(
        trace_id=normalized_trace_id,
        query=_observe_query(
            normalized_query,
            rewritten_query=rewritten_query,
            expanded_queries=expanded_queries,
        ),
        retrieval=_observe_retrieval(
            retrieved_chunks,
            requested_top_k=requested_top_k,
            observed_chunk_limit=observed_chunk_limit,
        ),
        rerank=_observe_rerank(
            rerank_report=rerank_report,
            rerank_execution=rerank_execution,
        ),
        citation=(
            _observe_citation(citation_report)
            if citation_report is not None
            else None
        ),
        timings=[_observe_timing(timing) for timing in timings],
        total_elapsed_ms=total_elapsed_ms,
    )
    return event.model_copy(update={"warning_codes": _build_warning_codes(event)})


def build_safe_rag_log_payload(event: RagObservabilityEvent) -> dict[str, Any]:
    payload = event.model_dump(mode="json", exclude_none=True)
    payload["query"].pop("expanded_query_hashes", None)
    payload["timed_out_stages"] = [
        timing.stage
        for timing in event.timings
        if timing.status == "timed_out"
    ]
    payload["near_timeout_stages"] = [
        timing.stage
        for timing in event.timings
        if timing.status == "near_timeout"
    ]
    return payload


def format_rag_observability_event(event: RagObservabilityEvent) -> list[str]:
    lines = [
        (
            "RAG observability event "
            f"trace_id={event.trace_id} "
            f"query_hash={event.query.query_hash[:12]} "
            f"retrieved={event.retrieval.returned_count} "
            f"observed_chunks={event.retrieval.observed_count}"
        ),
        (
            "retrieval "
            f"top_k={event.retrieval.requested_top_k} "
            f"top_chunk={event.retrieval.top_chunk_id or '-'} "
            f"top_score={_format_optional_score(event.retrieval.top_score)} "
            f"sources={event.retrieval.source_counts}"
        ),
    ]
    if event.rerank is not None:
        lines.append(
            (
                "rerank "
                f"candidates={event.rerank.candidate_count} "
                f"returned={event.rerank.returned_count} "
                f"top_before={event.rerank.top_before_chunk_id or '-'} "
                f"top_after={event.rerank.top_after_chunk_id or '-'} "
                f"fallback={event.rerank.used_fallback}"
            )
        )
    if event.citation is not None:
        lines.append(
            (
                "citation "
                f"valid={event.citation.is_valid} "
                f"citations={event.citation.checked_citation_count} "
                f"missing={event.citation.missing_citation_count} "
                f"blocking={event.citation.blocking_finding_count} "
                f"warning={event.citation.warning_finding_count}"
            )
        )
    if event.timings:
        lines.append(
            "timings "
            + ", ".join(
                (
                    f"{timing.stage}={timing.elapsed_ms:.2f}ms/"
                    f"{timing.status}"
                )
                for timing in event.timings
            )
        )
    if event.warning_codes:
        lines.append("warnings " + ", ".join(event.warning_codes))
    return lines


def _observe_query(
    query: str,
    *,
    rewritten_query: str | None,
    expanded_queries: Sequence[str],
) -> RagObservedQuery:
    normalized_expanded_queries = [
        expanded_query.strip()
        for expanded_query in expanded_queries
        if expanded_query.strip()
    ]
    normalized_rewritten_query = (
        rewritten_query.strip()
        if rewritten_query is not None and rewritten_query.strip()
        else None
    )
    return RagObservedQuery(
        query_hash=_hash_text(query),
        query_preview=_safe_text_preview(query),
        rewritten_query_hash=(
            _hash_text(normalized_rewritten_query)
            if normalized_rewritten_query is not None
            else None
        ),
        rewritten_query_preview=(
            _safe_text_preview(normalized_rewritten_query)
            if normalized_rewritten_query is not None
            else None
        ),
        expanded_query_count=len(normalized_expanded_queries),
        expanded_query_hashes=[
            _hash_text(expanded_query)
            for expanded_query in normalized_expanded_queries
        ],
    )


def _observe_retrieval(
    chunks: Sequence[RetrievedChunk],
    *,
    requested_top_k: int,
    observed_chunk_limit: int,
) -> RagObservedRetrieval:
    observed_chunks = [
        _observe_retrieved_chunk(rank, chunk)
        for rank, chunk in enumerate(chunks[:observed_chunk_limit], start=1)
    ]
    source_counts = Counter(
        _metadata_text(chunk.metadata.get("source"), fallback="unknown-source")
        for chunk in chunks
    )
    return RagObservedRetrieval(
        requested_top_k=requested_top_k,
        returned_count=len(chunks),
        observed_count=len(observed_chunks),
        top_chunk_id=chunks[0].chunk_id if chunks else None,
        top_score=chunks[0].score if chunks else None,
        source_counts=dict(sorted(source_counts.items())),
        chunks=observed_chunks,
    )


def _observe_retrieved_chunk(
    rank: int,
    chunk: RetrievedChunk,
) -> RagObservedRetrievedChunk:
    return RagObservedRetrievedChunk(
        rank=rank,
        chunk_id=chunk.chunk_id,
        source=_metadata_text(chunk.metadata.get("source"), fallback="unknown-source"),
        title=_optional_metadata_text(chunk.metadata.get("title")),
        section=_optional_metadata_text(chunk.metadata.get("section")),
        retrieval_score=chunk.score,
        content_hash=_hash_text(chunk.content),
        content_chars=len(chunk.content),
    )


def _observe_rerank(
    *,
    rerank_report: RerankReport | None,
    rerank_execution: RerankExecutionResult | None,
) -> RagObservedRerank | None:
    if rerank_report is None and rerank_execution is None:
        return None

    report_chunks = rerank_report.results if rerank_report is not None else []
    execution_chunks = rerank_execution.results if rerank_execution is not None else []
    chunks = report_chunks or execution_chunks
    return RagObservedRerank(
        candidate_count=(
            rerank_report.candidate_count
            if rerank_report is not None
            else len(execution_chunks)
        ),
        returned_count=(
            rerank_report.returned_count
            if rerank_report is not None
            else len(execution_chunks)
        ),
        top_before_chunk_id=(
            rerank_report.top_before_chunk_id
            if rerank_report is not None
            else None
        ),
        top_after_chunk_id=(
            rerank_report.top_after_chunk_id
            if rerank_report is not None
            else (execution_chunks[0].chunk_id if execution_chunks else None)
        ),
        moved_count=rerank_report.moved_count if rerank_report is not None else 0,
        used_fallback=(
            rerank_execution.used_fallback
            if rerank_execution is not None
            else False
        ),
        fallback_reason=(
            rerank_execution.fallback_reason
            if rerank_execution is not None
            else None
        ),
        elapsed_ms=(
            rerank_execution.elapsed_ms
            if rerank_execution is not None
            else None
        ),
        chunks=[_observe_reranked_chunk(chunk) for chunk in chunks],
    )


def _observe_reranked_chunk(chunk: RerankedChunk) -> RagObservedRerankedChunk:
    return RagObservedRerankedChunk(
        chunk_id=chunk.chunk_id,
        original_rank=chunk.original_rank,
        rerank_rank=chunk.rerank_rank,
        retrieval_score=chunk.retrieval_score,
        rerank_score=chunk.rerank_score,
    )


def _observe_citation(
    report: CitationVerificationReport,
) -> RagObservedCitation:
    severity_counts = Counter(finding.severity for finding in report.findings)
    return RagObservedCitation(
        answer_status=report.answer_status.value,
        is_valid=report.is_valid,
        retrieved_chunk_count=report.retrieved_chunk_count,
        checked_citation_count=report.checked_citation_count,
        cited_chunk_count=report.cited_chunk_count,
        missing_citation_count=report.missing_citation_count,
        answer_support_score=report.answer_support_score,
        blocking_finding_count=severity_counts[CitationFindingSeverity.BLOCKING],
        warning_finding_count=severity_counts[CitationFindingSeverity.WARNING],
        finding_codes=sorted({finding.code for finding in report.findings}),
    )


def _observe_timing(timing: RagOperationTiming) -> RagObservedTiming:
    return RagObservedTiming(
        stage=timing.stage.value,
        elapsed_ms=timing.elapsed_ms,
        timeout_seconds=timing.timeout_seconds,
        status=timing.status.value,
    )


def _build_warning_codes(event: RagObservabilityEvent) -> list[str]:
    warnings: list[str] = []
    if event.retrieval.returned_count == 0:
        warnings.append("RAG_OBS_NO_RETRIEVED_CHUNKS")
    if event.retrieval.returned_count < event.retrieval.requested_top_k:
        warnings.append("RAG_OBS_RETRIEVED_LESS_THAN_TOP_K")
    if event.rerank is not None and event.rerank.used_fallback:
        warnings.append("RAG_OBS_RERANK_USED_FALLBACK")
    if event.citation is not None and not event.citation.is_valid:
        warnings.append("RAG_OBS_CITATION_INVALID")
    if any(timing.status == "near_timeout" for timing in event.timings):
        warnings.append("RAG_OBS_NEAR_TIMEOUT")
    if any(timing.status == "timed_out" for timing in event.timings):
        warnings.append("RAG_OBS_TIMED_OUT")
    return warnings


def _safe_text_preview(
    text: str,
    *,
    max_chars: int = DEFAULT_QUERY_PREVIEW_CHARS,
) -> str:
    redacted = EMAIL_PATTERN.sub("[REDACTED_EMAIL]", text)
    redacted = PHONE_PATTERN.sub("[REDACTED_PHONE]", redacted)
    redacted = SECRET_PATTERN.sub("[REDACTED_SECRET]", redacted)
    normalized = " ".join(redacted.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _metadata_text(value: Any, *, fallback: str) -> str:
    text = _optional_metadata_text(value)
    return text if text is not None else fallback


def _optional_metadata_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _format_optional_score(score: float | None) -> str:
    if score is None:
        return "-"
    return f"{score:.4f}"
