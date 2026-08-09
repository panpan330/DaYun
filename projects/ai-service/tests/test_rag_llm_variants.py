"""Tests for LLM-driven variants of advanced RAG modules (rewrite / multi-query / routing)."""

from __future__ import annotations

import json

import pytest

from app.core.config import Settings
from app.rag.knowledge_routing import (
    LLMRagKnowledgeRouter,
    create_knowledge_router,
    default_rag_knowledge_bases,
)
from app.rag.multi_query import (
    LLMMultiQueryGenerator,
    create_multi_query_generator,
)
from app.rag.query_rewrite import (
    LLMQueryRewriter,
    RuleBasedQueryRewriter,
    create_query_rewriter,
)
from tests.fakes import FakeChatCompletions, FakeOpenAICompatibleClient


def _settings(*, advanced_mode: str = "rule") -> Settings:
    return Settings(
        _env_file=None,
        llm_api_key="test-key",
        llm_model="qwen3.7-plus",
        llm_provider="aliyun-compatible",
        rag_advanced_mode=advanced_mode,
    )


class TestLLMQueryRewriter:
    def test_rewrite_uses_llm_and_returns_changed_result(self) -> None:
        fake = FakeChatCompletions(content="退货运费由谁承担")
        rewriter = LLMQueryRewriter(_settings(advanced_mode="llm"), client=FakeOpenAICompatibleClient(fake))
        result = rewriter.rewrite("运费谁出")
        assert result.rewritten_query == "退货运费由谁承担"
        assert result.changed is True
        assert fake.calls  # LLM 被调用
        assert fake.last_call["model"] == "qwen3.7-plus"

    def test_rewrite_falls_back_on_llm_error(self) -> None:
        fake = FakeChatCompletions(error=RuntimeError("llm down"))
        rewriter = LLMQueryRewriter(_settings(advanced_mode="llm"), client=FakeOpenAICompatibleClient(fake))
        result = rewriter.rewrite("运费谁出")
        # 回退规则实现（不抛异常）
        assert result.original_query
        assert isinstance(result, type(RuleBasedQueryRewriter().rewrite("运费谁出")))


class TestLLMMultiQueryGenerator:
    def test_generate_uses_llm_and_returns_expansion(self) -> None:
        variants = ["退货运费谁承担", "退货邮费规则", "商家承担运费吗"]
        fake = FakeChatCompletions(content=json.dumps(variants, ensure_ascii=False))
        generator = LLMMultiQueryGenerator(
            _settings(advanced_mode="llm"),
            client=FakeOpenAICompatibleClient(fake),
        )
        expansion = generator.generate("运费谁出", max_queries=3)
        assert expansion.expanded is True
        assert [q.query for q in expansion.queries[:3]] == variants
        assert fake.calls

    def test_generate_falls_back_on_llm_error(self) -> None:
        fake = FakeChatCompletions(error=RuntimeError("llm down"))
        generator = LLMMultiQueryGenerator(
            _settings(advanced_mode="llm"),
            client=FakeOpenAICompatibleClient(fake),
        )
        expansion = generator.generate("运费谁出")
        assert expansion.original_query == "运费谁出"  # 回退规则仍产出 original


class TestLLMRagKnowledgeRouter:
    def test_route_uses_llm_and_selects_collection(self) -> None:
        bases = default_rag_knowledge_bases()
        collection_name = "kb_customer_policy"
        fake = FakeChatCompletions(content="customer_policy_refund")
        router = LLMRagKnowledgeRouter(
            _settings(advanced_mode="llm"),
            client=FakeOpenAICompatibleClient(fake),
            knowledge_bases=bases,
        )
        decision = router.route("退款政策是什么")
        assert decision.should_use_rag is True
        assert decision.routes[0].collection_name == collection_name
        assert fake.calls

    def test_route_falls_back_on_llm_error(self) -> None:
        fake = FakeChatCompletions(error=RuntimeError("llm down"))
        router = LLMRagKnowledgeRouter(
            _settings(advanced_mode="llm"),
            client=FakeOpenAICompatibleClient(fake),
        )
        decision = router.route("退款政策是什么")
        assert decision.should_use_rag is True  # 回退规则仍路由


class TestFactories:
    def test_create_query_rewriter_respects_advanced_mode(self) -> None:
        assert isinstance(create_query_rewriter(_settings(advanced_mode="rule")), RuleBasedQueryRewriter)
        assert isinstance(create_query_rewriter(_settings(advanced_mode="llm")), LLMQueryRewriter)

    def test_create_multi_query_generator_respects_advanced_mode(self) -> None:
        from app.rag.multi_query import RuleBasedMultiQueryGenerator

        assert isinstance(
            create_multi_query_generator(_settings(advanced_mode="rule")),
            RuleBasedMultiQueryGenerator,
        )
        assert isinstance(
            create_multi_query_generator(_settings(advanced_mode="llm")),
            LLMMultiQueryGenerator,
        )

    def test_create_knowledge_router_respects_advanced_mode(self) -> None:
        assert isinstance(create_knowledge_router(_settings(advanced_mode="llm")), LLMRagKnowledgeRouter)
        rule_router = create_knowledge_router(_settings(advanced_mode="rule"))
        assert rule_router.__class__.__name__ == "RuleBasedRagKnowledgeRouter"


class TestKeywordRetrieverUpgrade:
    def test_from_retrieved_chunks_builds_index(self) -> None:
        from app.rag.documents import RetrievedChunk
        from app.rag.hybrid import SimpleKeywordRetriever

        chunks = [
            RetrievedChunk(
                point_id="p1",
                chunk_id="c1",
                content="退货运费由商家承担",
                metadata={"source": "refund-return-policy.md"},
                score=0.9,
            ),
            RetrievedChunk(
                point_id="p2",
                chunk_id="c2",
                content="订单发货后不可取消",
                metadata={"source": "order-shipping-policy.md"},
                score=0.8,
            ),
        ]
        retriever = SimpleKeywordRetriever.from_retrieved_chunks(chunks)
        results = retriever.search("退货运费", top_k=3, min_score=0.0)
        assert results and results[0].chunk_id == "c1"

    def test_from_vector_store_uses_scroll_all(self) -> None:
        from app.rag.documents import RetrievedChunk
        from app.rag.hybrid import SimpleKeywordRetriever

        class FakeStore:
            def scroll_all(self):
                yield RetrievedChunk(
                    point_id="p1",
                    chunk_id="c1",
                    content="退款政策：七天无理由",
                    metadata={"source": "refund-return-policy.md"},
                    score=0.9,
                )

        retriever = SimpleKeywordRetriever.from_vector_store(FakeStore())
        results = retriever.search("退款政策", top_k=3, min_score=0.0)
        assert results and results[0].chunk_id == "c1"

    def test_from_vector_store_rejects_unsupported_store(self) -> None:
        from app.rag.hybrid import SimpleKeywordRetriever

        class FakeStore:
            pass

        try:
            SimpleKeywordRetriever.from_vector_store(FakeStore())
        except ValueError as exc:
            assert "scroll_all" in str(exc)
            return
        raise AssertionError("unsupported store should raise ValueError")


class TestRagFeatureSwitches:
    def test_rag_enable_settings_defaults(self) -> None:
        from app.core.config import Settings

        settings = Settings(_env_file=None)
        # routing / hybrid / rewrite 默认开启；multi_query / compression / citation_verify 默认关闭
        assert settings.rag_enable_rewrite is True
        assert settings.rag_enable_multi_query is False
        assert settings.rag_enable_routing is True
        assert settings.rag_enable_hybrid is True
        assert settings.rag_enable_context_compression is False
        assert settings.rag_enable_citation_verify is False
        assert settings.rag_advanced_mode == "rule"


def test_multi_query_blank_variants_falls_back() -> None:
    """LLM 返回全空串/非字符串 → 回退规则实现（不抛 min_length 校验错误）。"""
    from app.rag.multi_query import LLMMultiQueryGenerator, RuleBasedMultiQueryGenerator
    from tests.fakes import FakeChatCompletions, FakeOpenAICompatibleClient

    fake = FakeChatCompletions(content=json.dumps(["", "  ", 123], ensure_ascii=False))
    generator = LLMMultiQueryGenerator(
        _settings(advanced_mode="llm"),
        client=FakeOpenAICompatibleClient(fake),
    )
    expansion = generator.generate("运费谁出")
    # 回退规则仍产出 original query
    assert expansion.original_query == "运费谁出"
    assert isinstance(generator._fallback, RuleBasedMultiQueryGenerator)
