import pytest

from app.rag.context_compression import (
    ContextCompressionAction,
    ContextCompressionPolicy,
    compress_retrieved_context,
    estimate_context_chars,
    format_context_compression_report_for_debug,
)
from app.rag.generator import build_rag_context
from tests.rag_fakes import make_retrieved_chunk


def make_chunk(
    chunk_id: str,
    content: str,
    *,
    score: float = 0.9,
    source: str = "policy.md",
    section: str = "General",
):
    return make_retrieved_chunk(
        point_id=f"point-{chunk_id}",
        chunk_id=chunk_id,
        content=content,
        metadata={
            "source": source,
            "title": "Policy",
            "section": section,
            "permission_group": "customer_service",
        },
        score=score,
    )


def test_compress_retrieved_context_keeps_chunks_when_under_budget() -> None:
    chunks = [
        make_chunk("refund-1", "Refunds are returned within 1 to 3 business days."),
        make_chunk("shipping-1", "Paid orders are usually shipped within 24 hours."),
    ]

    report = compress_retrieved_context(
        "refund time",
        chunks,
        policy=ContextCompressionPolicy(max_total_chars=500, max_chunk_chars=250),
    )

    assert report.original_total_chars == estimate_context_chars(chunks)
    assert report.final_total_chars == report.original_total_chars
    assert report.compressed_chunk_count == 0
    assert report.dropped_chunk_count == 0
    assert [item.action for item in report.items] == [
        ContextCompressionAction.KEEP_FULL,
        ContextCompressionAction.KEEP_FULL,
    ]
    assert [chunk.content for chunk in report.compressed_chunks] == [
        chunk.content for chunk in chunks
    ]


def test_compress_retrieved_context_compresses_long_chunk_around_query_terms() -> None:
    long_content = (
        "Account security settings can be updated from the profile page. "
        "Marketing campaigns may include coupons and seasonal messages. "
        "Refunds are returned to the original payment method within 1 to 3 business days. "
        "Warehouse teams may pack orders in multiple waves during promotions."
    )
    chunk = make_chunk("refund-1", long_content)

    report = compress_retrieved_context(
        "refund original payment method",
        [chunk],
        policy=ContextCompressionPolicy(
            max_total_chars=110,
            max_chunk_chars=110,
            min_chunk_chars=40,
        ),
    )

    assert report.compressed_chunk_count == 1
    assert report.dropped_chunk_count == 0
    compressed = report.compressed_chunks[0]
    assert len(compressed.content) <= 110
    assert "Refunds are returned" in compressed.content
    assert "original payment method" in compressed.content
    assert compressed.metadata["context_compression_action"] == "compress"
    assert compressed.metadata["context_original_rank"] == 1
    assert compressed.metadata["context_original_chars"] == len(long_content)
    assert compressed.metadata["context_final_chars"] == len(compressed.content)


def test_compress_retrieved_context_drops_chunks_when_budget_is_exhausted() -> None:
    chunks = [
        make_chunk("chunk-1", "A" * 90, score=0.99),
        make_chunk("chunk-2", "B" * 90, score=0.88),
        make_chunk("chunk-3", "C" * 90, score=0.77),
    ]

    report = compress_retrieved_context(
        "refund",
        chunks,
        policy=ContextCompressionPolicy(
            max_total_chars=120,
            max_chunk_chars=90,
            min_chunk_chars=50,
            always_keep_top_n=1,
        ),
    )

    assert [chunk.chunk_id for chunk in report.compressed_chunks] == ["chunk-1"]
    assert report.dropped_chunk_count == 2
    assert [item.action for item in report.items] == [
        ContextCompressionAction.KEEP_FULL,
        ContextCompressionAction.DROP,
        ContextCompressionAction.DROP,
    ]


def test_compress_retrieved_context_can_compress_second_chunk_if_budget_remains() -> None:
    chunks = [
        make_chunk("chunk-1", "Refund policy summary is short."),
        make_chunk(
            "chunk-2",
            (
                "Shipping details are unrelated. "
                "Refund audit records explain refund approval timing. "
                "Extra operational text makes this chunk longer than the budget."
            ),
        ),
    ]

    report = compress_retrieved_context(
        "refund approval timing",
        chunks,
        policy=ContextCompressionPolicy(
            max_total_chars=115,
            max_chunk_chars=85,
            min_chunk_chars=40,
            always_keep_top_n=1,
        ),
    )

    assert [item.action for item in report.items] == [
        ContextCompressionAction.KEEP_FULL,
        ContextCompressionAction.COMPRESS,
    ]
    assert report.final_total_chars <= 115
    assert "refund approval timing" in report.compressed_chunks[1].content.lower()


def test_compress_retrieved_context_preserves_order_for_kept_chunks() -> None:
    chunks = [
        make_chunk("chunk-1", "First relevant refund policy."),
        make_chunk("chunk-2", "Second relevant refund policy."),
        make_chunk("chunk-3", "Third relevant refund policy."),
    ]

    report = compress_retrieved_context(
        "refund policy",
        chunks,
        policy=ContextCompressionPolicy(max_total_chars=500, max_chunk_chars=200),
    )

    assert [chunk.chunk_id for chunk in report.compressed_chunks] == [
        "chunk-1",
        "chunk-2",
        "chunk-3",
    ]
    assert [item.original_rank for item in report.items] == [1, 2, 3]


def test_compressed_chunks_can_feed_existing_rag_context_builder() -> None:
    chunk = make_chunk(
        "refund-1",
        (
            "Refunds are returned to the original payment method within 1 to 3 business days. "
            "This sentence adds enough extra text to require compression."
        ),
    )

    report = compress_retrieved_context(
        "refund payment method",
        [chunk],
        policy=ContextCompressionPolicy(
            max_total_chars=90,
            max_chunk_chars=90,
            min_chunk_chars=40,
        ),
    )
    context = build_rag_context(report.compressed_chunks)

    assert "source: policy.md" in context
    assert "chunk_id: refund-1" in context
    assert "content:" in context


def test_format_context_compression_report_for_debug_includes_actions() -> None:
    chunks = [
        make_chunk("chunk-1", "Refund policy summary is short."),
        make_chunk("chunk-2", "X" * 120),
    ]
    report = compress_retrieved_context(
        "refund",
        chunks,
        policy=ContextCompressionPolicy(
            max_total_chars=50,
            max_chunk_chars=60,
            min_chunk_chars=30,
        ),
    )

    lines = format_context_compression_report_for_debug(report)

    assert lines[0].startswith("budget=50 original=")
    assert "action=keep_full" in "\n".join(lines)
    assert "action=drop" in "\n".join(lines)


def test_compress_retrieved_context_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="query"):
        compress_retrieved_context("   ", [])

    with pytest.raises(ValueError, match="min_chunk_chars"):
        ContextCompressionPolicy(
            max_total_chars=100,
            max_chunk_chars=50,
            min_chunk_chars=80,
        )

    with pytest.raises(ValueError, match="min_chunk_chars"):
        ContextCompressionPolicy(
            max_total_chars=40,
            max_chunk_chars=80,
            min_chunk_chars=50,
        )
