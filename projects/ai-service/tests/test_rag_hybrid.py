import pytest

from app.rag.documents import RagChunk
from app.rag.hybrid import (
    HybridSearchWeights,
    KeywordSearchResult,
    SimpleKeywordRetriever,
    build_hybrid_fusion_report,
    extract_keyword_terms,
    format_hybrid_results_for_debug,
    fuse_hybrid_results,
    hybrid_retrieve,
)
from tests.rag_fakes import (
    FakeEmbeddingModel,
    FakeVectorStoreReader,
    make_retrieved_chunk,
)


def make_chunk(**overrides) -> RagChunk:
    payload = {
        "chunk_id": "refund_chunk_0001",
        "content": "退款到账时间通常为 1 到 3 个工作日。",
        "metadata": {
            "source": "refund-return-policy.md",
            "title": "退款退货规则",
            "section": "退款到账时间",
            "doc_type": "policy",
            "business_domain": "refund",
            "permission_group": "customer_service",
        },
    }
    payload.update(overrides)
    return RagChunk(**payload)


def test_extract_keyword_terms_builds_chinese_ngrams_and_ascii_terms() -> None:
    terms = extract_keyword_terms("订单 ABC123 退款多久到账？")

    assert "订单" in terms
    assert "abc123" in terms
    assert "退款" in terms
    assert "到账" in terms
    assert "多久" in terms


def test_simple_keyword_retriever_scores_and_sorts_chunks() -> None:
    retriever = SimpleKeywordRetriever(
        [
            make_chunk(),
            make_chunk(
                chunk_id="refund_chunk_0002",
                content="退款申请需要先完成订单状态校验。",
                metadata={
                    "source": "refund-return-policy.md",
                    "title": "退款退货规则",
                    "section": "退款申请",
                    "doc_type": "policy",
                    "business_domain": "refund",
                    "permission_group": "customer_service",
                },
            ),
        ]
    )

    results = retriever.search("退款多久到账？", top_k=2)

    assert results[0].chunk_id == "refund_chunk_0001"
    assert results[0].score > results[1].score
    assert "退款" in results[0].matched_terms
    assert "到账" in results[0].matched_terms


def test_simple_keyword_retriever_filters_by_metadata() -> None:
    retriever = SimpleKeywordRetriever(
        [
            make_chunk(),
            make_chunk(
                chunk_id="internal_refund_chunk_0001",
                metadata={
                    "source": "internal-refund.md",
                    "title": "内部退款规则",
                    "section": "退款到账时间",
                    "doc_type": "policy",
                    "business_domain": "refund",
                    "permission_group": "internal_staff",
                },
            ),
        ]
    )

    results = retriever.search(
        "退款到账",
        top_k=3,
        permission_group="customer_service",
    )

    assert [result.chunk_id for result in results] == ["refund_chunk_0001"]


def test_simple_keyword_retriever_respects_top_k_and_min_score() -> None:
    retriever = SimpleKeywordRetriever(
        [
            make_chunk(chunk_id="refund_chunk_0001", content="退款到账时间通常为 1 到 3 个工作日。"),
            make_chunk(chunk_id="refund_chunk_0002", content="退款申请需要先完成订单状态校验。"),
            make_chunk(chunk_id="shipping_chunk_0001", content="订单付款后 24 小时内发货。"),
        ]
    )

    results = retriever.search("退款到账", top_k=1, min_score=0.5)

    assert [result.chunk_id for result in results] == ["refund_chunk_0001"]


def test_simple_keyword_retriever_rejects_invalid_query_and_options() -> None:
    retriever = SimpleKeywordRetriever([make_chunk()])

    with pytest.raises(ValueError, match="query"):
        retriever.search("   ")

    with pytest.raises(ValueError, match="top_k"):
        retriever.search("退款", top_k=0)

    with pytest.raises(ValueError, match="min_score"):
        retriever.search("退款", min_score=True)


def test_fuse_hybrid_results_merges_by_chunk_id_and_scores_sources() -> None:
    vector_chunks = [
        make_retrieved_chunk(chunk_id="chunk-vector-only", score=0.9),
        make_retrieved_chunk(chunk_id="chunk-both", score=0.6),
    ]
    keyword_results = [
        KeywordSearchResult(
            chunk_id="chunk-both",
            content="退款到账时间通常为 1 到 3 个工作日。",
            metadata={"source": "refund.md"},
            score=1.0,
            matched_terms=["退款", "到账"],
        ),
        KeywordSearchResult(
            chunk_id="chunk-keyword-only",
            content="退款申请需要先完成订单状态校验。",
            metadata={"source": "refund.md"},
            score=0.8,
            matched_terms=["退款"],
        ),
    ]

    results = fuse_hybrid_results(
        vector_chunks,
        keyword_results,
        top_k=3,
        vector_weight=0.6,
        keyword_weight=0.4,
    )

    assert [result.chunk_id for result in results] == [
        "chunk-both",
        "chunk-vector-only",
        "chunk-keyword-only",
    ]
    assert results[0].retrieval_sources == ["vector", "keyword"]
    assert results[0].vector_score == 0.6
    assert results[0].keyword_score == 1.0
    assert results[0].matched_terms == ["退款", "到账"]


def test_build_hybrid_fusion_report_summarizes_source_mix() -> None:
    vector_chunks = [
        make_retrieved_chunk(chunk_id="chunk-vector-only", score=0.9),
        make_retrieved_chunk(chunk_id="chunk-both", score=0.6),
    ]
    keyword_results = [
        KeywordSearchResult(
            chunk_id="chunk-both",
            content="退款到账时间通常为 1 到 3 个工作日。",
            metadata={"source": "refund.md", "section": "退款到账"},
            score=1.0,
            matched_terms=["退款", "到账"],
        ),
        KeywordSearchResult(
            chunk_id="chunk-keyword-only",
            content="退款申请需要先完成订单状态校验。",
            metadata={"source": "refund.md", "section": "退款申请"},
            score=0.8,
            matched_terms=["退款"],
        ),
    ]

    report = build_hybrid_fusion_report(
        vector_chunks,
        keyword_results,
        top_k=3,
        vector_weight=0.6,
        keyword_weight=0.4,
    )

    assert report.vector_result_count == 2
    assert report.keyword_result_count == 2
    assert report.fused_result_count == 3
    assert report.vector_only_count == 1
    assert report.keyword_only_count == 1
    assert report.both_count == 1
    assert report.overlap_chunk_ids == ["chunk-both"]
    assert report.top_chunk_id == "chunk-both"
    assert report.debug_lines[0].startswith("1. hybrid_score=")
    assert "sources=vector,keyword" in report.debug_lines[0]


def test_format_hybrid_results_for_debug_includes_scores_and_terms() -> None:
    results = fuse_hybrid_results(
        [make_retrieved_chunk(chunk_id="chunk-vector-only", score=0.9)],
        [
            KeywordSearchResult(
                chunk_id="chunk-keyword-only",
                content="退款到账时间通常为 1 到 3 个工作日。",
                metadata={"source": "refund.md", "section": "退款到账"},
                score=1.0,
                matched_terms=["退款", "到账"],
            )
        ],
        top_k=2,
    )

    lines = format_hybrid_results_for_debug(results)

    assert lines[0].startswith("1. hybrid_score=")
    assert "vector_score=" in lines[0]
    assert "keyword_score=" in lines[0]
    assert "chunk_id=" in lines[0]


def test_hybrid_weights_reject_zero_total_and_bool() -> None:
    with pytest.raises(ValueError, match="at least one"):
        HybridSearchWeights(vector_weight=0, keyword_weight=0)

    with pytest.raises(ValueError, match="numbers"):
        HybridSearchWeights(vector_weight=True, keyword_weight=0.3)


def test_hybrid_retrieve_runs_vector_and_keyword_retrieval() -> None:
    embedding_model = FakeEmbeddingModel(dimension=4)
    vector_store = FakeVectorStoreReader(
        chunks=[
            make_retrieved_chunk(
                chunk_id="shipping_chunk_0001",
                content="订单付款后 24 小时内发货。",
                metadata={
                    "source": "order-shipping-policy.md",
                    "title": "订单发货规则",
                    "section": "正常发货时效",
                    "business_domain": "order",
                    "permission_group": "customer_service",
                },
                score=0.88,
            )
        ]
    )
    keyword_retriever = SimpleKeywordRetriever([make_chunk()])

    results = hybrid_retrieve(
        "退款多久到账？",
        embedding_model=embedding_model,
        vector_store=vector_store,
        keyword_retriever=keyword_retriever,
        vector_top_k=1,
        keyword_top_k=1,
        final_top_k=2,
        permission_group="customer_service",
    )

    assert {result.chunk_id for result in results} == {
        "shipping_chunk_0001",
        "refund_chunk_0001",
    }
    assert vector_store.last_call["top_k"] == 1
    assert vector_store.last_call["payload_filter"] == {
        "must": [
            {"key": "permission_group", "match": {"value": "customer_service"}},
        ]
    }
    assert embedding_model.last_texts == ["退款多久到账？"]


def test_hybrid_retrieve_rejects_blank_query() -> None:
    with pytest.raises(ValueError, match="query"):
        hybrid_retrieve(
            "   ",
            embedding_model=FakeEmbeddingModel(dimension=4),
            vector_store=FakeVectorStoreReader(),
            keyword_retriever=SimpleKeywordRetriever([make_chunk()]),
        )
