from app.rag.citation_verification import (
    CitationFindingSeverity,
    format_citation_verification_report_for_debug,
    verify_rag_answer_sources,
)
from app.rag.generator import (
    RagAnswer,
    RagAnswerStatus,
    RagCitation,
    build_grounded_rag_answer,
    build_no_context_rag_answer,
)
from tests.rag_fakes import make_retrieved_chunk


def make_policy_chunk(**overrides):
    payload = {
        "point_id": "point-refund-1",
        "chunk_id": "refund_policy_chunk_0001",
        "content": (
            "Refunds are returned to the original payment method within "
            "1 to 3 business days after approval."
        ),
        "metadata": {
            "source": "refund-policy.md",
            "title": "Refund Policy",
            "section": "Refund arrival time",
            "permission_group": "customer_service",
        },
        "score": 0.92,
    }
    payload.update(overrides)
    return make_retrieved_chunk(**payload)


def test_verify_rag_answer_sources_accepts_backend_generated_citations() -> None:
    chunk = make_policy_chunk()
    answer = build_grounded_rag_answer(
        "Refunds are returned to the original payment method within 1 to 3 business days.",
        [chunk],
    )

    report = verify_rag_answer_sources(answer, [chunk])

    assert report.is_valid is True
    assert report.retrieved_chunk_count == 1
    assert report.checked_citation_count == 1
    assert report.cited_chunk_count == 1
    assert report.missing_citation_count == 0
    assert report.answer_support_score > 0.8
    assert report.findings == []


def test_verify_rag_answer_sources_accepts_no_context_without_citations() -> None:
    answer = build_no_context_rag_answer()

    report = verify_rag_answer_sources(answer, [])

    assert report.is_valid is True
    assert report.answer_status is RagAnswerStatus.NO_CONTEXT
    assert report.checked_citation_count == 0
    assert report.cited_chunk_count == 0
    assert report.findings == []


def test_verify_rag_answer_sources_blocks_answer_without_citations() -> None:
    chunk = make_policy_chunk()
    answer = RagAnswer(
        answer="Refunds are returned within 1 to 3 business days.",
        status=RagAnswerStatus.ANSWERED,
        citations=[],
    )

    report = verify_rag_answer_sources(answer, [chunk])

    assert report.is_valid is False
    assert {finding.code for finding in report.findings} == {
        "RAG_ANSWERED_WITHOUT_CITATIONS"
    }


def test_verify_rag_answer_sources_blocks_out_of_range_source_index() -> None:
    chunk = make_policy_chunk()
    answer = RagAnswer(
        answer="Refunds are returned within 1 to 3 business days.",
        status=RagAnswerStatus.ANSWERED,
        citations=[
            RagCitation(
                source_index=2,
                source="refund-policy.md",
                title="Refund Policy",
                section="Refund arrival time",
                chunk_id="refund_policy_chunk_0001",
                score=0.92,
            )
        ],
    )

    report = verify_rag_answer_sources(answer, [chunk])

    assert report.is_valid is False
    assert "RAG_CITATION_SOURCE_INDEX_OUT_OF_RANGE" in {
        finding.code for finding in report.findings
    }
    assert report.missing_citation_count == 1


def test_verify_rag_answer_sources_blocks_unretrieved_chunk_id() -> None:
    chunk = make_policy_chunk()
    answer = RagAnswer(
        answer="Refunds are returned within 1 to 3 business days.",
        status=RagAnswerStatus.ANSWERED,
        citations=[
            RagCitation(
                source_index=1,
                source="refund-policy.md",
                title="Refund Policy",
                section="Refund arrival time",
                chunk_id="unknown_chunk",
                score=0.92,
            )
        ],
    )

    report = verify_rag_answer_sources(answer, [chunk])

    assert report.is_valid is False
    assert {"RAG_CITATION_CHUNK_NOT_RETRIEVED", "RAG_CITATION_SOURCE_INDEX_MISMATCH"} <= {
        finding.code for finding in report.findings
    }


def test_verify_rag_answer_sources_blocks_source_index_chunk_mismatch() -> None:
    refund_chunk = make_policy_chunk()
    shipping_chunk = make_policy_chunk(
        point_id="point-shipping-1",
        chunk_id="shipping_policy_chunk_0001",
        content="Paid orders are usually shipped within 24 hours.",
        metadata={
            "source": "shipping-policy.md",
            "title": "Shipping Policy",
            "section": "Normal shipping time",
            "permission_group": "customer_service",
        },
        score=0.81,
    )
    answer = RagAnswer(
        answer="Paid orders are usually shipped within 24 hours.",
        status=RagAnswerStatus.ANSWERED,
        citations=[
            RagCitation(
                source_index=1,
                source="shipping-policy.md",
                title="Shipping Policy",
                section="Normal shipping time",
                chunk_id="shipping_policy_chunk_0001",
                score=0.81,
            )
        ],
    )

    report = verify_rag_answer_sources(answer, [refund_chunk, shipping_chunk])

    assert report.is_valid is False
    assert "RAG_CITATION_SOURCE_INDEX_MISMATCH" in {
        finding.code for finding in report.findings
    }


def test_verify_rag_answer_sources_blocks_source_metadata_mismatch() -> None:
    chunk = make_policy_chunk()
    answer = RagAnswer(
        answer="Refunds are returned within 1 to 3 business days.",
        status=RagAnswerStatus.ANSWERED,
        citations=[
            RagCitation(
                source_index=1,
                source="wrong-source.md",
                title="Refund Policy",
                section="Refund arrival time",
                chunk_id="refund_policy_chunk_0001",
                score=0.92,
            )
        ],
    )

    report = verify_rag_answer_sources(answer, [chunk])

    assert report.is_valid is False
    assert "RAG_CITATION_SOURCE_MISMATCH" in {
        finding.code for finding in report.findings
    }


def test_verify_rag_answer_sources_warns_on_low_answer_overlap() -> None:
    chunk = make_policy_chunk()
    answer = build_grounded_rag_answer(
        "Customers can update account security settings from the profile page.",
        [chunk],
    )

    report = verify_rag_answer_sources(answer, [chunk])

    assert report.is_valid is True
    assert {
        "RAG_CITATION_LOW_TEXT_OVERLAP",
        "RAG_ANSWER_LOW_TEXT_OVERLAP",
    } <= {finding.code for finding in report.findings}
    assert {
        finding.severity for finding in report.findings
    } == {CitationFindingSeverity.WARNING}


def test_verify_rag_answer_sources_warns_on_duplicate_citations() -> None:
    chunk = make_policy_chunk()
    citation = RagCitation(
        source_index=1,
        source="refund-policy.md",
        title="Refund Policy",
        section="Refund arrival time",
        chunk_id="refund_policy_chunk_0001",
        score=0.92,
    )
    answer = RagAnswer(
        answer="Refunds are returned to the original payment method within 1 to 3 business days.",
        status=RagAnswerStatus.ANSWERED,
        citations=[citation, citation],
    )

    report = verify_rag_answer_sources(answer, [chunk])

    assert report.is_valid is True
    assert [finding.code for finding in report.findings] == [
        "RAG_CITATION_DUPLICATE_CHUNK",
        "RAG_CITATION_DUPLICATE_CHUNK",
    ]


def test_format_citation_verification_report_for_debug_includes_findings() -> None:
    chunk = make_policy_chunk()
    answer = RagAnswer(
        answer="Refunds are returned within 1 to 3 business days.",
        status=RagAnswerStatus.ANSWERED,
        citations=[],
    )
    report = verify_rag_answer_sources(answer, [chunk])

    lines = format_citation_verification_report_for_debug(report)

    assert lines[0].startswith("valid=False status=answered retrieved=1 citations=0")
    assert "code=RAG_ANSWERED_WITHOUT_CITATIONS" in lines[1]
