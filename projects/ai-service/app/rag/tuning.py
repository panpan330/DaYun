from collections.abc import Iterable, Sequence
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.rag.documents import RagDocument, RetrievedChunk
from app.rag.embeddings import EmbeddingModel
from app.rag.evaluation import (
    RagAnswerQualitySummary,
    RagBadCaseReport,
    RetrievalEvalSummary,
)
from app.rag.retriever import (
    DEFAULT_TOP_K,
    VectorStoreReader,
    format_retrieved_chunks_for_debug,
    retrieve_top_k,
)
from app.rag.splitters import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    split_documents_into_chunks,
)


class ChunkTuningCase(BaseModel):
    chunk_size: int = Field(default=DEFAULT_CHUNK_SIZE, gt=0)
    chunk_overlap: int = Field(default=DEFAULT_CHUNK_OVERLAP, ge=0)

    @field_validator("chunk_size", "chunk_overlap", mode="before")
    @classmethod
    def reject_bool_ints(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("chunk tuning values must be integers")
        return value

    @model_validator(mode="after")
    def validate_overlap(self) -> "ChunkTuningCase":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class ChunkTuningReport(BaseModel):
    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)
    document_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    min_chunk_chars: int = Field(ge=0)
    max_chunk_chars: int = Field(ge=0)
    average_chunk_chars: float = Field(ge=0)
    source_count: int = Field(ge=0)


class RetrievalTuningCase(BaseModel):
    top_k: int = Field(default=DEFAULT_TOP_K, gt=0)
    score_threshold: float | None = None

    @field_validator("top_k", mode="before")
    @classmethod
    def reject_bool_top_k(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("top_k must be an integer")
        return value

    @field_validator("score_threshold", mode="before")
    @classmethod
    def validate_score_threshold(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("score_threshold must be a number")
        return value


class RetrievalTuningReport(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(gt=0)
    score_threshold: float | None = None
    result_count: int = Field(ge=0)
    source_count: int = Field(ge=0)
    top_score: float | None = None
    bottom_score: float | None = None
    sources: list[str] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)
    debug_lines: list[str] = Field(default_factory=list)


TuningParameter = Literal[
    "chunk_size",
    "chunk_overlap",
    "top_k",
    "score_threshold",
    "rerank",
    "metadata_filter",
    "prompt",
    "no_context_gate",
    "security_policy",
]
TuningDirection = Literal["increase", "decrease", "review", "keep"]
TuningPriority = Literal["high", "medium", "low"]


class RagParameterTuningRecommendation(BaseModel):
    parameter: TuningParameter
    direction: TuningDirection
    priority: TuningPriority
    reason: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    expected_benefit: str = Field(min_length=1)
    risk: str = Field(min_length=1)


class RagParameterTuningReport(BaseModel):
    recommendation_count: int = Field(ge=0)
    high_priority_count: int = Field(ge=0)
    metric_snapshot: list[str] = Field(default_factory=list)
    recommendations: list[RagParameterTuningRecommendation] = Field(
        default_factory=list
    )


def compare_chunk_tuning_cases(
    documents: Sequence[RagDocument],
    cases: Sequence[ChunkTuningCase],
) -> list[ChunkTuningReport]:
    return [
        build_chunk_tuning_report(
            documents,
            chunk_size=case.chunk_size,
            chunk_overlap=case.chunk_overlap,
        )
        for case in cases
    ]


def build_chunk_tuning_report(
    documents: Sequence[RagDocument],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> ChunkTuningReport:
    case = ChunkTuningCase(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = split_documents_into_chunks(
        list(documents),
        chunk_size=case.chunk_size,
        chunk_overlap=case.chunk_overlap,
    )
    chunk_lengths = [len(chunk.content) for chunk in chunks]
    sources = {
        source
        for chunk in chunks
        if isinstance((source := chunk.metadata.get("source")), str)
    }

    return ChunkTuningReport(
        chunk_size=case.chunk_size,
        chunk_overlap=case.chunk_overlap,
        document_count=len(documents),
        chunk_count=len(chunks),
        min_chunk_chars=min(chunk_lengths, default=0),
        max_chunk_chars=max(chunk_lengths, default=0),
        average_chunk_chars=_average(chunk_lengths),
        source_count=len(sources),
    )


def build_retrieval_tuning_cases(
    *,
    top_ks: Sequence[int],
    score_thresholds: Sequence[float | None],
) -> list[RetrievalTuningCase]:
    return [
        RetrievalTuningCase(top_k=top_k, score_threshold=score_threshold)
        for top_k in top_ks
        for score_threshold in score_thresholds
    ]


def compare_retrieval_tuning_cases(
    query: str,
    *,
    embedding_model: EmbeddingModel,
    vector_store: VectorStoreReader,
    cases: Sequence[RetrievalTuningCase],
    permission_group: str | None = None,
    business_domain: str | None = None,
    doc_type: str | None = None,
    source: str | None = None,
) -> list[RetrievalTuningReport]:
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query must not be blank")

    reports: list[RetrievalTuningReport] = []
    for case in cases:
        chunks = retrieve_top_k(
            normalized_query,
            embedding_model=embedding_model,
            vector_store=vector_store,
            top_k=case.top_k,
            permission_group=permission_group,
            business_domain=business_domain,
            doc_type=doc_type,
            source=source,
            score_threshold=case.score_threshold,
        )
        reports.append(
            build_retrieval_tuning_report(
                normalized_query,
                chunks,
                top_k=case.top_k,
                score_threshold=case.score_threshold,
            )
        )
    return reports


def build_rag_parameter_tuning_report(
    *,
    retrieval_summary: RetrievalEvalSummary | None = None,
    answer_quality_summary: RagAnswerQualitySummary | None = None,
    bad_case_report: RagBadCaseReport | None = None,
) -> RagParameterTuningReport:
    recommendations: list[RagParameterTuningRecommendation] = []
    metric_snapshot: list[str] = []

    if retrieval_summary is not None:
        metric_snapshot.extend(_retrieval_metric_snapshot(retrieval_summary))
        _append_retrieval_metric_recommendations(recommendations, retrieval_summary)

    if answer_quality_summary is not None:
        metric_snapshot.extend(_answer_quality_metric_snapshot(answer_quality_summary))
        _append_answer_quality_recommendations(
            recommendations,
            answer_quality_summary,
        )

    if bad_case_report is not None:
        metric_snapshot.append(
            f"bad_case_layers={_format_layer_counts(bad_case_report.layer_counts)}"
        )
        _append_bad_case_layer_recommendations(recommendations, bad_case_report)

    deduped_recommendations = _dedupe_recommendations(recommendations)
    return RagParameterTuningReport(
        recommendation_count=len(deduped_recommendations),
        high_priority_count=sum(
            1 for recommendation in deduped_recommendations if recommendation.priority == "high"
        ),
        metric_snapshot=metric_snapshot,
        recommendations=deduped_recommendations,
    )


def format_rag_parameter_tuning_report(
    report: RagParameterTuningReport,
) -> list[str]:
    lines = [
        "RAG parameter tuning report",
        f"recommendations: {report.recommendation_count}",
        f"high_priority: {report.high_priority_count}",
    ]
    lines.extend(f"metric: {line}" for line in report.metric_snapshot)
    for recommendation in report.recommendations:
        lines.append(
            f"- {recommendation.priority} {recommendation.parameter} "
            f"{recommendation.direction}: {recommendation.reason} "
            f"evidence={recommendation.evidence} risk={recommendation.risk}"
        )
    return lines


def build_retrieval_tuning_report(
    query: str,
    chunks: Sequence[RetrievedChunk],
    *,
    top_k: int,
    score_threshold: float | None = None,
) -> RetrievalTuningReport:
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query must not be blank")
    case = RetrievalTuningCase(top_k=top_k, score_threshold=score_threshold)
    scores = [chunk.score for chunk in chunks]
    sources = _unique_strings(
        chunk.metadata.get("source")
        for chunk in chunks
        if isinstance(chunk.metadata.get("source"), str)
    )

    return RetrievalTuningReport(
        query=normalized_query,
        top_k=case.top_k,
        score_threshold=case.score_threshold,
        result_count=len(chunks),
        source_count=len(sources),
        top_score=max(scores, default=None),
        bottom_score=min(scores, default=None),
        sources=sources,
        chunk_ids=[chunk.chunk_id for chunk in chunks],
        debug_lines=format_retrieved_chunks_for_debug(chunks),
    )


def _average(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 2)


def _unique_strings(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value in seen:
            continue
        unique.append(value)
        seen.add(value)
    return unique


def _append_retrieval_metric_recommendations(
    recommendations: list[RagParameterTuningRecommendation],
    summary: RetrievalEvalSummary,
) -> None:
    if summary.evaluated_case_count == 0 and summary.no_result_case_count == 0:
        return

    if summary.recall_at_k < 0.8 and summary.evaluated_case_count:
        recommendations.append(
            _recommendation(
                parameter="top_k",
                direction="increase",
                priority="high",
                reason="Recall@K is low, so expected evidence is often not entering the candidate set.",
                evidence=f"recall@{summary.top_k}={summary.recall_at_k:.4f}",
                expected_benefit="Give retrieval more room to include missing relevant chunks.",
                risk="Higher top_k can reduce precision, increase context size, and add more noise.",
            )
        )
        recommendations.append(
            _recommendation(
                parameter="score_threshold",
                direction="decrease",
                priority="medium",
                reason="Low recall can be caused by filtering out relevant but lower-scored chunks.",
                evidence=f"recall@{summary.top_k}={summary.recall_at_k:.4f}",
                expected_benefit="Allow borderline relevant chunks to remain available for rerank or generation.",
                risk="Lower threshold can introduce weakly related chunks and more prompt-injection surface.",
            )
        )
        recommendations.append(
            _recommendation(
                parameter="chunk_size",
                direction="review",
                priority="medium",
                reason="Low recall may mean key facts are split poorly or chunks lack enough surrounding context.",
                evidence=f"failed_cases={summary.failed_case_count}",
                expected_benefit="Find whether smaller or larger chunks make expected evidence easier to retrieve.",
                risk="Changing chunk_size requires re-indexing and can invalidate chunk_id-level expectations.",
            )
        )

    if (
        summary.precision_at_k < 0.5
        and summary.recall_at_k >= 0.8
        and summary.evaluated_case_count
    ):
        recommendations.append(
            _recommendation(
                parameter="score_threshold",
                direction="increase",
                priority="high" if summary.recall_at_k >= 0.8 else "medium",
                reason="Precision@K is low, so the retrieved candidates contain too much noise.",
                evidence=f"precision@{summary.top_k}={summary.precision_at_k:.4f}",
                expected_benefit="Filter out low-confidence candidates before context construction.",
                risk="Higher threshold can hurt recall, especially for paraphrased or ambiguous questions.",
            )
        )
        recommendations.append(
            _recommendation(
                parameter="top_k",
                direction="decrease",
                priority="medium" if summary.recall_at_k >= 0.8 else "low",
                reason="Low precision may mean top_k is allowing too many noisy chunks into the candidate set.",
                evidence=f"precision@{summary.top_k}={summary.precision_at_k:.4f}",
                expected_benefit="Reduce context noise and downstream generation confusion.",
                risk="Lower top_k can hide relevant evidence and reduce recall.",
            )
        )

    if (
        summary.hit_rate_at_k >= 0.8
        and summary.mrr_at_k < 0.6
        and summary.evaluated_case_count
    ):
        recommendations.append(
            _recommendation(
                parameter="rerank",
                direction="review",
                priority="medium",
                reason="Hit Rate is acceptable but MRR is low, so relevant chunks are found but not ranked early.",
                evidence=(
                    f"hit_rate@{summary.top_k}={summary.hit_rate_at_k:.4f}, "
                    f"mrr@{summary.top_k}={summary.mrr_at_k:.4f}"
                ),
                expected_benefit="Move useful evidence earlier before context compression and generation.",
                risk="Rerank adds latency and can fail over to weaker fallback behavior.",
            )
        )

    if (
        summary.no_result_success_rate is not None
        and summary.no_result_success_rate < 1.0
    ):
        recommendations.append(
            _recommendation(
                parameter="score_threshold",
                direction="increase",
                priority="high",
                reason="No-context cases are still returning retrieved chunks.",
                evidence=(
                    f"no_result_success_rate={summary.no_result_success_rate:.4f}"
                ),
                expected_benefit="Reduce false positives for questions outside the knowledge base.",
                risk="A stricter threshold can also remove valid but lower-scored evidence.",
            )
        )
        recommendations.append(
            _recommendation(
                parameter="no_context_gate",
                direction="review",
                priority="high",
                reason="No-context behavior needs a decision rule beyond ordinary top_k retrieval.",
                evidence=(
                    f"no_result_success_rate={summary.no_result_success_rate:.4f}"
                ),
                expected_benefit="Prevent the model from answering unsupported questions.",
                risk="Over-strict no-context gates can reject answerable user questions.",
            )
        )


def _append_answer_quality_recommendations(
    recommendations: list[RagParameterTuningRecommendation],
    summary: RagAnswerQualitySummary,
) -> None:
    if summary.answer_case_count and summary.average_answer_point_coverage < 0.8:
        recommendations.append(
            _recommendation(
                parameter="prompt",
                direction="review",
                priority="high",
                reason="Answer point coverage is low even after retrieval and answer evaluation.",
                evidence=(
                    f"average_answer_point_coverage={summary.average_answer_point_coverage:.4f}"
                ),
                expected_benefit="Make the model explicitly cover conditions, exceptions, and answer points.",
                risk="A stricter prompt can make answers longer or less natural.",
            )
        )
        recommendations.append(
            _recommendation(
                parameter="chunk_overlap",
                direction="increase",
                priority="medium",
                reason="Missing answer points can be caused by facts split across chunk boundaries.",
                evidence=(
                    f"average_answer_point_coverage={summary.average_answer_point_coverage:.4f}"
                ),
                expected_benefit="Preserve neighboring facts across chunks so generation sees complete context.",
                risk="More overlap increases index size, duplicate content, and retrieval noise.",
            )
        )

    if summary.citation_pass_rate is not None and summary.citation_pass_rate < 1.0:
        recommendations.append(
            _recommendation(
                parameter="metadata_filter",
                direction="review",
                priority="medium",
                reason="Citation pass rate is below 1.0, so sources may be missing or unexpected.",
                evidence=f"citation_pass_rate={summary.citation_pass_rate:.4f}",
                expected_benefit="Ensure retrieved and cited chunks carry correct source metadata.",
                risk="Metadata changes may require re-indexing and dataset expectation updates.",
            )
        )

    if summary.refusal_pass_rate is not None and summary.refusal_pass_rate < 1.0:
        recommendations.append(
            _recommendation(
                parameter="no_context_gate",
                direction="review",
                priority="high",
                reason="Refusal pass rate is below 1.0, so non-answer behavior is unreliable.",
                evidence=f"refusal_pass_rate={summary.refusal_pass_rate:.4f}",
                expected_benefit="Improve no-context, access-denied, security-block, and clarify decisions.",
                risk="Over-strict refusal logic can block legitimate answerable questions.",
            )
        )


def _append_bad_case_layer_recommendations(
    recommendations: list[RagParameterTuningRecommendation],
    report: RagBadCaseReport,
) -> None:
    layer_counts = report.layer_counts
    if layer_counts.get("retrieval", 0):
        recommendations.append(
            _recommendation(
                parameter="top_k",
                direction="increase",
                priority="high",
                reason="Bad cases include retrieval-layer failures.",
                evidence=f"retrieval_layer_cases={layer_counts['retrieval']}",
                expected_benefit="Improve the chance that expected evidence reaches downstream rerank and generation.",
                risk="More candidates increase noise, cost, and context pressure.",
            )
        )
    if layer_counts.get("ranking", 0):
        recommendations.append(
            _recommendation(
                parameter="rerank",
                direction="review",
                priority="medium",
                reason="Bad cases include ranking-layer warnings or failures.",
                evidence=f"ranking_layer_cases={layer_counts['ranking']}",
                expected_benefit="Promote relevant chunks earlier in the candidate list.",
                risk="Rerank adds latency and depends on model/provider quality.",
            )
        )
    if layer_counts.get("generation", 0):
        recommendations.append(
            _recommendation(
                parameter="prompt",
                direction="review",
                priority="medium",
                reason="Bad cases include generation-layer failures.",
                evidence=f"generation_layer_cases={layer_counts['generation']}",
                expected_benefit="Improve answer completeness and instruction following.",
                risk="Prompt changes can shift answer style and must be regression-tested.",
            )
        )
    if layer_counts.get("access_control", 0):
        recommendations.append(
            _recommendation(
                parameter="metadata_filter",
                direction="review",
                priority="high",
                reason="Bad cases include access-control failures, which should not be solved by top_k tuning.",
                evidence=f"access_control_layer_cases={layer_counts['access_control']}",
                expected_benefit="Prevent forbidden or cross-scope evidence from reaching the model.",
                risk="Incorrect filters can hide valid documents from allowed users.",
            )
        )
    if layer_counts.get("security", 0):
        recommendations.append(
            _recommendation(
                parameter="security_policy",
                direction="review",
                priority="high",
                reason="Bad cases include security failures, which must be handled before generation.",
                evidence=f"security_layer_cases={layer_counts['security']}",
                expected_benefit="Block prompt-injection and unsafe context before model use.",
                risk="Over-strict policies can block benign but unusual documents.",
            )
        )


def _retrieval_metric_snapshot(summary: RetrievalEvalSummary) -> list[str]:
    lines = [
        f"top_k={summary.top_k}",
        f"hit_rate@{summary.top_k}={summary.hit_rate_at_k:.4f}",
        f"recall@{summary.top_k}={summary.recall_at_k:.4f}",
        f"precision@{summary.top_k}={summary.precision_at_k:.4f}",
        f"mrr@{summary.top_k}={summary.mrr_at_k:.4f}",
    ]
    if summary.no_result_success_rate is not None:
        lines.append(
            f"no_result_success_rate={summary.no_result_success_rate:.4f}"
        )
    return lines


def _answer_quality_metric_snapshot(summary: RagAnswerQualitySummary) -> list[str]:
    lines = [
        f"answer_pass_rate={summary.pass_rate:.4f}",
        (
            "average_answer_point_coverage="
            f"{summary.average_answer_point_coverage:.4f}"
        ),
    ]
    if summary.citation_pass_rate is not None:
        lines.append(f"citation_pass_rate={summary.citation_pass_rate:.4f}")
    if summary.refusal_pass_rate is not None:
        lines.append(f"refusal_pass_rate={summary.refusal_pass_rate:.4f}")
    return lines


def _recommendation(
    *,
    parameter: TuningParameter,
    direction: TuningDirection,
    priority: TuningPriority,
    reason: str,
    evidence: str,
    expected_benefit: str,
    risk: str,
) -> RagParameterTuningRecommendation:
    return RagParameterTuningRecommendation(
        parameter=parameter,
        direction=direction,
        priority=priority,
        reason=reason,
        evidence=evidence,
        expected_benefit=expected_benefit,
        risk=risk,
    )


def _dedupe_recommendations(
    recommendations: Sequence[RagParameterTuningRecommendation],
) -> list[RagParameterTuningRecommendation]:
    deduped: list[RagParameterTuningRecommendation] = []
    seen: set[tuple[str, str]] = set()
    for recommendation in recommendations:
        key = (recommendation.parameter, recommendation.direction)
        if key in seen:
            continue
        deduped.append(recommendation)
        seen.add(key)
    return deduped


def _format_layer_counts(layer_counts: dict[str, int]) -> str:
    if not layer_counts:
        return "-"
    return ", ".join(
        f"{layer}={count}" for layer, count in sorted(layer_counts.items())
    )
