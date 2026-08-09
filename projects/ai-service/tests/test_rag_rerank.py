import httpx
import pytest

from app.core.config import Settings
from app.rag.hybrid import HybridSearchResult, KeywordSearchResult
from app.rag.rerank import (
    HttpReranker,
    RerankCandidate,
    RerankModelError,
    RuleBasedReranker,
    build_rerank_report,
    format_reranked_chunks_for_debug,
    make_rerank_candidates_from_hybrid_results,
    make_rerank_candidates_from_keyword_results,
    make_rerank_candidates_from_retrieved_chunks,
    rerank_candidates,
    rerank_with_fallback,
    reranked_chunks_to_retrieved_chunks,
)
from app.rag.score_interpretation import describe_milvus_score
from tests.rag_fakes import make_retrieved_chunk


def make_candidate(**overrides) -> RerankCandidate:
    payload = {
        "chunk_id": "refund_arrival_chunk",
        "content": "退货商品入库并审核通过后，退款通常会在 1 到 3 个工作日内原路退回。",
        "metadata": {
            "source": "refund-return-policy.md",
            "title": "退款退货规则",
            "section": "退款到账时间",
            "permission_group": "customer_service",
        },
        "retrieval_score": 0.7,
        "retrieval_sources": ["vector"],
    }
    payload.update(overrides)
    return RerankCandidate(**payload)


def test_rerank_candidates_moves_more_relevant_chunk_to_top() -> None:
    candidates = [
        make_candidate(
            chunk_id="logistics_refund_chunk",
            content="物流异常不能直接退款，需要先确认订单状态和异常原因。",
            metadata={"source": "logistics.txt", "section": "物流异常可以直接退款吗"},
            retrieval_score=0.99,
        ),
        make_candidate(
            chunk_id="refund_arrival_chunk",
            content="退货商品入库并审核通过后，退款通常会在 1 到 3 个工作日内原路退回。如果超过 3 个工作日仍未到账，客服需要核查退款流水状态。",
            metadata={
                "source": "refund-return-policy.md",
                "title": "退款退货规则",
                "section": "退款到账时间",
            },
            retrieval_score=0.72,
        ),
    ]

    results = rerank_candidates("退款多久到账？", candidates, top_k=2)

    assert [result.chunk_id for result in results] == [
        "refund_arrival_chunk",
        "logistics_refund_chunk",
    ]
    assert results[0].original_rank == 2
    assert results[0].rerank_rank == 1
    assert "到账" in results[0].matched_terms


def test_rerank_candidates_records_score_breakdown() -> None:
    results = rerank_candidates(
        "退款到账",
        [
            make_candidate(
                retrieval_score=0.5,
                retrieval_sources=["vector", "keyword"],
                matched_terms=["退款"],
            )
        ],
        top_k=1,
    )

    result = results[0]

    assert result.score_breakdown.content_match_score > 0
    assert result.score_breakdown.title_section_match_score > 0
    assert result.score_breakdown.normalized_retrieval_score == 1
    assert result.score_breakdown.source_agreement_score == 1
    assert result.retrieval_sources == ["vector", "keyword"]
    assert "退款" in result.matched_terms


def test_rerank_candidates_can_normalize_lower_is_better_retrieval_scores() -> None:
    candidates = [
        RerankCandidate(
            chunk_id="far_l2_distance",
            content="generic placeholder text",
            retrieval_score=0.9,
            retrieval_sources=["vector"],
        ),
        RerankCandidate(
            chunk_id="near_l2_distance",
            content="generic placeholder text",
            retrieval_score=0.2,
            retrieval_sources=["vector"],
        ),
    ]

    results = rerank_candidates(
        "refund arrival",
        candidates,
        top_k=2,
        retrieval_score_meaning=describe_milvus_score("L2"),
    )

    assert [result.chunk_id for result in results] == [
        "near_l2_distance",
        "far_l2_distance",
    ]
    assert results[0].score_breakdown.normalized_retrieval_score == 1
    assert results[1].score_breakdown.normalized_retrieval_score == 0


def test_build_rerank_report_summarizes_rank_changes() -> None:
    candidates = [
        make_candidate(
            chunk_id="weak_first",
            content="物流异常不能直接退款，需要先确认订单状态。",
            retrieval_score=0.99,
        ),
        make_candidate(
            chunk_id="strong_second",
            content="退款到账时间通常为 1 到 3 个工作日。",
            metadata={"source": "refund.md", "section": "退款到账时间"},
            retrieval_score=0.72,
        ),
        make_candidate(
            chunk_id="dropped_third",
            content="售后工单会在 24 小时内处理。",
            retrieval_score=0.4,
        ),
    ]

    report = build_rerank_report("退款多久到账", candidates, top_k=2)

    assert report.candidate_count == 3
    assert report.returned_count == 2
    assert report.top_before_chunk_id == "weak_first"
    assert report.top_after_chunk_id == "strong_second"
    assert report.moved_count == 2
    assert report.promoted_chunk_ids == ["strong_second"]
    assert report.dropped_chunk_ids == ["dropped_third"]
    assert report.retrieval_score_direction == "higher_is_better"
    assert report.debug_lines[0].startswith("1. rerank_score=")


def test_rerank_candidates_limits_top_k_and_uses_stable_tie_breaker() -> None:
    candidates = [
        make_candidate(chunk_id="chunk-b", retrieval_score=0.5),
        make_candidate(chunk_id="chunk-a", retrieval_score=0.5),
    ]

    results = rerank_candidates("退款到账", candidates, top_k=1)

    assert len(results) == 1
    assert results[0].chunk_id == "chunk-b"


def test_rerank_candidates_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="query"):
        rerank_candidates("   ", [make_candidate()])

    with pytest.raises(ValueError, match="top_k"):
        rerank_candidates("退款", [make_candidate()], top_k=0)

    with pytest.raises(ValueError, match="retrieval_score"):
        RerankCandidate(
            chunk_id="bad",
            content="退款到账",
            retrieval_score=True,
        )


def test_make_rerank_candidates_from_retrieved_chunks() -> None:
    chunks = [
        make_retrieved_chunk(
            chunk_id="chunk-1",
            content="订单通常 48 小时内发货。",
            score=0.88,
            metadata={"source": "shipping.md", "point_id": "point-1"},
        )
    ]

    candidates = make_rerank_candidates_from_retrieved_chunks(chunks)

    assert candidates[0].chunk_id == "chunk-1"
    assert candidates[0].retrieval_score == 0.88
    assert candidates[0].retrieval_sources == ["vector"]


def test_make_rerank_candidates_from_keyword_results() -> None:
    results = [
        KeywordSearchResult(
            chunk_id="chunk-keyword",
            content="退款通常 1 到 3 个工作日到账。",
            metadata={"source": "refund.md"},
            score=0.6,
            matched_terms=["退款", "到账"],
        )
    ]

    candidates = make_rerank_candidates_from_keyword_results(results)

    assert candidates[0].retrieval_score == 0.6
    assert candidates[0].retrieval_sources == ["keyword"]
    assert candidates[0].matched_terms == ["退款", "到账"]


def test_make_rerank_candidates_from_hybrid_results() -> None:
    results = [
        HybridSearchResult(
            chunk_id="chunk-hybrid",
            content="退款通常 1 到 3 个工作日到账。",
            metadata={"source": "refund.md"},
            hybrid_score=0.77,
            vector_score=0.8,
            keyword_score=0.6,
            retrieval_sources=["vector", "keyword"],
            matched_terms=["退款"],
        )
    ]

    candidates = make_rerank_candidates_from_hybrid_results(results)

    assert candidates[0].retrieval_score == 0.77
    assert candidates[0].retrieval_sources == ["vector", "keyword"]


def test_rule_based_reranker_delegates_to_rerank_candidates() -> None:
    reranker = RuleBasedReranker()

    results = reranker.rerank("退款到账", [make_candidate()], top_k=1)

    assert results[0].chunk_id == "refund_arrival_chunk"
    assert results[0].rerank_rank == 1


def test_http_reranker_posts_candidates_and_builds_ranked_chunks() -> None:
    captured_request: dict | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_request
        captured_request = {
            "path": request.url.path,
            "headers": dict(request.headers),
            "json": request.read().decode("utf-8"),
        }
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.93},
                    {"index": 0, "relevance_score": 0.41},
                ]
            },
            request=request,
        )

    reranker = HttpReranker(
        base_url="https://rerank.example.com/v1",
        model="rerank-demo",
        timeout_seconds=3,
        api_key="test-key",
        transport=httpx.MockTransport(handler),
    )
    candidates = [
        make_candidate(chunk_id="first", content="refund policy"),
        make_candidate(chunk_id="second", content="refund arrival time"),
    ]

    results = reranker.rerank("refund arrival", candidates, top_k=2)

    assert [result.chunk_id for result in results] == ["second", "first"]
    assert results[0].rerank_score == 0.93
    assert results[0].original_rank == 2
    assert results[0].rerank_rank == 1
    assert captured_request is not None
    assert captured_request["path"] == "/v1/rerank"
    assert captured_request["headers"]["authorization"] == "Bearer test-key"
    assert '"model":"rerank-demo"' in captured_request["json"]
    assert '"top_n":2' in captured_request["json"]


def test_http_reranker_supports_dashscope_nested_endpoint_shape() -> None:
    captured_request: dict | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_request
        captured_request = {
            "path": request.url.path,
            "json": request.read().decode("utf-8"),
        }
        return httpx.Response(
            200,
            json={
                "output": {
                    "results": [
                        {"index": 0, "relevance_score": 0.88},
                    ]
                }
            },
            request=request,
        )

    reranker = HttpReranker(
        base_url=(
            "https://workspace.cn-beijing.maas.aliyuncs.com/"
            "api/v1/services/rerank/text-rerank/text-rerank"
        ),
        model="qwen3-rerank",
        timeout_seconds=3,
        api_key="test-key",
        transport=httpx.MockTransport(handler),
    )

    results = reranker.rerank(
        "refund arrival",
        [make_candidate(chunk_id="first", content="refund policy")],
        top_k=1,
    )

    assert results[0].chunk_id == "first"
    assert results[0].rerank_score == 0.88
    assert captured_request is not None
    assert captured_request["path"].endswith("/text-rerank/text-rerank")
    assert '"input":{"query":"refund arrival","documents":["refund policy"]}' in captured_request["json"]
    assert '"parameters":{"return_documents":false,"top_n":1}' in captured_request["json"]


def test_http_reranker_from_settings_uses_rerank_config() -> None:
    reranker = HttpReranker.from_settings(
        Settings(
            rerank_base_url=" https://rerank.example.com/api/ ",
            rerank_model="real-rerank",
            rerank_api_key="rerank-key",
            rerank_timeout_seconds=4.5,
            rerank_max_retries=2,
            _env_file=None,
        ),
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={"results": []},
                request=request,
            )
        ),
    )

    assert reranker.base_url == "https://rerank.example.com/api"
    assert reranker.endpoint_url == "https://rerank.example.com/api/rerank"
    assert reranker.model == "real-rerank"
    assert reranker.api_key == "rerank-key"
    assert reranker.timeout_seconds == 4.5
    assert reranker.max_retries == 2


def test_http_reranker_validates_provider_response() -> None:
    reranker = HttpReranker(
        base_url="https://rerank.example.com",
        model="rerank-demo",
        timeout_seconds=3,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={"results": [{"index": 9, "relevance_score": 0.5}]},
                request=request,
            )
        ),
    )

    with pytest.raises(RerankModelError, match="index"):
        reranker.rerank("refund", [make_candidate()], top_k=1)


def test_rerank_with_fallback_uses_rule_based_reranker_on_provider_failure() -> None:
    primary = HttpReranker(
        base_url="https://rerank.example.com",
        model="rerank-demo",
        timeout_seconds=3,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                500,
                json={"error": "temporary unavailable"},
                request=request,
            )
        ),
    )
    candidates = [
        make_candidate(
            chunk_id="weak_first",
            content="物流异常不能直接退款，需要先确认订单状态。",
            retrieval_score=0.99,
        ),
        make_candidate(
            chunk_id="strong_second",
            content="退款到账时间通常为 1 到 3 个工作日。",
            metadata={"source": "refund.md", "section": "退款到账时间"},
            retrieval_score=0.72,
        ),
    ]

    result = rerank_with_fallback(
        "退款多久到账",
        candidates,
        primary_reranker=primary,
        top_k=2,
    )

    assert result.used_fallback is True
    assert result.fallback_reason == "RerankModelError"
    assert [chunk.chunk_id for chunk in result.results] == [
        "strong_second",
        "weak_first",
    ]


def test_reranked_chunks_to_retrieved_chunks_uses_rerank_score() -> None:
    reranked = rerank_candidates("退款到账", [make_candidate()], top_k=1)

    chunks = reranked_chunks_to_retrieved_chunks(reranked)

    assert chunks[0].chunk_id == "refund_arrival_chunk"
    assert chunks[0].score == reranked[0].rerank_score
    assert chunks[0].point_id == "refund_arrival_chunk"


def test_format_reranked_chunks_for_debug() -> None:
    reranked = rerank_candidates("退款到账", [make_candidate()], top_k=1)

    lines = format_reranked_chunks_for_debug(reranked)

    assert lines[0].startswith("1. rerank_score=")
    assert "original_rank=1" in lines[0]
    assert "chunk_id=refund_arrival_chunk" in lines[0]
