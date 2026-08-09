import pytest

from app.rag.citation_verification import (
    CitationFindingCategory,
    CitationFindingSeverity,
    CitationVerificationFinding,
    CitationVerificationReport,
)
from app.rag.generator import RagAnswerStatus
from app.rag.observability import (
    build_rag_observability_event,
    build_safe_rag_log_payload,
    format_rag_observability_event,
)
from app.rag.performance import RagOperationStage, assess_operation_timing
from app.rag.rerank import (
    RerankExecutionResult,
    RerankReport,
    RerankScoreBreakdown,
    RerankedChunk,
)
from tests.rag_fakes import make_retrieved_chunk


def test_build_rag_observability_event_records_safe_query_and_retrieval_snapshot() -> None:
    chunk = make_retrieved_chunk(
        content="退款通常会在 3 到 5 个工作日内原路退回，手机号 13812345678 不应进入日志。",
        metadata={
            "source": "refund-return-policy.md",
            "title": "退款退货规则",
            "section": "退款时效",
        },
        score=0.87,
    )

    event = build_rag_observability_event(
        trace_id="trace-rag-001",
        user_query="我的手机号 13812345678，退款多久到账？",
        rewritten_query="退款到账时效 13812345678",
        expanded_queries=["退款多久原路退回", "退款到账时间"],
        retrieved_chunks=[chunk],
        requested_top_k=5,
        total_elapsed_ms=93.5,
    )
    payload = build_safe_rag_log_payload(event)
    payload_text = str(payload)

    assert event.query.expanded_query_count == 2
    assert event.retrieval.returned_count == 1
    assert event.retrieval.top_chunk_id == chunk.chunk_id
    assert event.retrieval.source_counts == {"refund-return-policy.md": 1}
    assert "13812345678" not in payload_text
    assert "[REDACTED_PHONE]" in payload["query"]["query_preview"]
    assert "退款通常会在" not in payload_text
    assert event.retrieval.chunks[0].content_hash
    assert "RAG_OBS_RETRIEVED_LESS_THAN_TOP_K" in event.warning_codes


def test_build_rag_observability_event_records_timing_warnings() -> None:
    event = build_rag_observability_event(
        trace_id="trace-rag-002",
        user_query="物流一直没有更新怎么办？",
        retrieved_chunks=[],
        requested_top_k=3,
        timings=[
            assess_operation_timing(
                RagOperationStage.VECTOR_STORE,
                elapsed_ms=850,
                timeout_seconds=1,
            ),
            assess_operation_timing(
                RagOperationStage.GENERATION,
                elapsed_ms=1500,
                timeout_seconds=1,
            ),
        ],
    )
    payload = build_safe_rag_log_payload(event)
    lines = format_rag_observability_event(event)

    assert payload["near_timeout_stages"] == ["vector_store"]
    assert payload["timed_out_stages"] == ["generation"]
    assert "RAG_OBS_NO_RETRIEVED_CHUNKS" in event.warning_codes
    assert "RAG_OBS_NEAR_TIMEOUT" in event.warning_codes
    assert "RAG_OBS_TIMED_OUT" in event.warning_codes
    assert any("generation=1500.00ms/timed_out" in line for line in lines)


def test_build_rag_observability_event_records_rerank_and_citation_summary() -> None:
    reranked_chunk = RerankedChunk(
        chunk_id="refund_chunk_0002",
        content="退款时效说明。",
        metadata={"source": "refund-return-policy.md"},
        retrieval_score=0.72,
        rerank_score=0.93,
        original_rank=2,
        rerank_rank=1,
        score_breakdown=RerankScoreBreakdown(
            content_match_score=0.9,
            title_section_match_score=0.7,
            normalized_retrieval_score=0.8,
            source_agreement_score=0.0,
        ),
    )
    rerank_report = RerankReport(
        query="退款多久到账？",
        top_k=1,
        candidate_count=2,
        returned_count=1,
        top_before_chunk_id="refund_chunk_0001",
        top_after_chunk_id="refund_chunk_0002",
        moved_count=1,
        results=[reranked_chunk],
        retrieval_score_direction="higher_is_better",
    )
    rerank_execution = RerankExecutionResult(
        results=[reranked_chunk],
        used_fallback=True,
        fallback_reason="RerankModelError",
        elapsed_ms=12.5,
    )
    citation_report = CitationVerificationReport(
        answer_status=RagAnswerStatus.ANSWERED,
        is_valid=False,
        retrieved_chunk_count=2,
        checked_citation_count=1,
        cited_chunk_count=0,
        missing_citation_count=1,
        answer_support_score=0.03,
        findings=[
            CitationVerificationFinding(
                code="RAG_CITATION_CHUNK_NOT_FOUND",
                category=CitationFindingCategory.CITATION_REFERENCE,
                severity=CitationFindingSeverity.BLOCKING,
                message="citation chunk id was not retrieved",
                chunk_id="fake_chunk",
            )
        ],
    )

    event = build_rag_observability_event(
        trace_id="trace-rag-003",
        user_query="退款多久到账？",
        retrieved_chunks=[
            make_retrieved_chunk(chunk_id="refund_chunk_0001", score=0.8),
            make_retrieved_chunk(chunk_id="refund_chunk_0002", score=0.72),
        ],
        requested_top_k=2,
        rerank_report=rerank_report,
        rerank_execution=rerank_execution,
        citation_report=citation_report,
    )

    assert event.rerank is not None
    assert event.rerank.top_before_chunk_id == "refund_chunk_0001"
    assert event.rerank.top_after_chunk_id == "refund_chunk_0002"
    assert event.rerank.used_fallback is True
    assert event.rerank.fallback_reason == "RerankModelError"
    assert event.citation is not None
    assert event.citation.blocking_finding_count == 1
    assert event.citation.finding_codes == ["RAG_CITATION_CHUNK_NOT_FOUND"]
    assert "RAG_OBS_RERANK_USED_FALLBACK" in event.warning_codes
    assert "RAG_OBS_CITATION_INVALID" in event.warning_codes


def test_build_rag_observability_event_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="trace_id"):
        build_rag_observability_event(
            trace_id=" ",
            user_query="退款多久到账？",
            retrieved_chunks=[],
            requested_top_k=3,
        )

    with pytest.raises(ValueError, match="requested_top_k"):
        build_rag_observability_event(
            trace_id="trace-rag-004",
            user_query="退款多久到账？",
            retrieved_chunks=[],
            requested_top_k=0,
        )
