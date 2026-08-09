"""Tests for the enhanced RAG pipeline (feature switches)."""

from __future__ import annotations

from app.core.config import Settings
from app.rag import pipeline as pipeline_module


def _settings(**overrides) -> Settings:
    base = dict(
        llm_api_key="test-key",
        llm_model="qwen3.7-plus",
        llm_provider="aliyun-compatible",
        rag_enable_rewrite=False,
        rag_enable_multi_query=False,
        rag_enable_routing=False,
        rag_enable_hybrid=False,
        rag_enable_context_compression=False,
        rag_enable_citation_verify=False,
    )
    base.update(overrides)
    return Settings(_env_file=None, **base)


def _fake_rag_answer() -> object:
    from app.rag.generator import RagAnswer

    return RagAnswer(
        answer="退款政策：七天无理由退货",
        status="answered",
        citations=[],
        no_context_reason=None,
        suggestions=[],
    )


def test_default_path_retrieves_and_generates(monkeypatch) -> None:
    """默认开关全关：走 retrieve_top_k + rerank + generate（与阶段 1 一致）。"""
    settings = _settings()
    calls: list[str] = []
    monkeypatch.setattr(pipeline_module, "_retrieve", lambda q, **kw: calls.append("retrieve") or [])
    monkeypatch.setattr(
        pipeline_module,
        "create_rag_answer_service",
        lambda s: type("Svc", (), {"generate_answer_with_citations": lambda self, q, chunks: (_fake_rag_answer())})(),
    )
    result = pipeline_module.enhanced_rag_answer("退款政策是什么", settings=settings)
    assert result is not None
    assert "retrieve" in calls


def test_rewrite_enabled_calls_rewriter(monkeypatch) -> None:
    """RAG_ENABLE_REWRITE=true：改写后再检索。"""
    settings = _settings(rag_enable_rewrite=True)
    rewritten = {"value": None}

    class FakeRewriter:
        def rewrite(self, query):
            rewritten["value"] = query
            from app.rag.query_rewrite import QueryRewriteResult

            return QueryRewriteResult(
                original_query=query,
                rewritten_query="退货运费由谁承担",
                changed=True,
            )

    import app.rag.pipeline as pipe

    monkeypatch.setattr(pipe, "create_query_rewriter", lambda s: FakeRewriter())
    monkeypatch.setattr(pipe, "_retrieve", lambda q, **kw: [])
    monkeypatch.setattr(
        pipe,
        "create_rag_answer_service",
        lambda s: type("Svc", (), {"generate_answer_with_citations": lambda self, q, chunks: (_fake_rag_answer())})(),
    )
    pipe.enhanced_rag_answer("运费谁出", settings=settings)
    assert rewritten["value"] == "运费谁出"  # rewriter 收到原问


def test_routing_enabled_selects_collection(monkeypatch) -> None:
    """RAG_ENABLE_ROUTING=true：按 route.collection_name 构造 store。"""
    settings = _settings(rag_enable_routing=True)
    captured = {}

    class FakeRouter:
        def route(self, query, **kw):
            from app.rag.knowledge_routing import RagKnowledgeRoute, RagKnowledgeRouteDecision

            return RagKnowledgeRouteDecision(
                normalized_query=query,
                intent="policy_lookup",
                should_use_rag=True,
                selected_route_count=1,
                routes=[
                    RagKnowledgeRoute(
                        knowledge_base_id="customer_policy_refund",
                        collection_name="kb_customer_policy",
                        display_name="refund",
                        route_score=1.0,
                    )
                ],
            )

    import app.rag.pipeline as pipe

    monkeypatch.setattr(pipe, "create_knowledge_router", lambda s: FakeRouter())
    monkeypatch.setattr(pipe, "_retrieve", lambda q, **kw: captured.update(kw) or [])
    monkeypatch.setattr(
        pipe,
        "create_rag_answer_service",
        lambda s: type("Svc", (), {"generate_answer_with_citations": lambda self, q, chunks: (_fake_rag_answer())})(),
    )
    pipe.enhanced_rag_answer("退款政策是什么", settings=settings)
    assert captured.get("collection_name") == "kb_customer_policy"


def test_hybrid_enabled_calls_hybrid(monkeypatch) -> None:
    """RAG_ENABLE_HYBRID=true：走 hybrid 检索。"""
    settings = _settings(rag_enable_hybrid=True)
    called = {"hybrid": False}
    monkeypatch.setattr(pipeline_module, "_hybrid_retrieve", lambda q, **kw: called.__setitem__("hybrid", True) or [])
    monkeypatch.setattr(
        pipeline_module,
        "create_rag_answer_service",
        lambda s: type("Svc", (), {"generate_answer_with_citations": lambda self, q, chunks: (_fake_rag_answer())})(),
    )
    pipeline_module.enhanced_rag_answer("退款政策是什么", settings=settings)
    assert called["hybrid"] is True


def test_hybrid_uses_real_store_scroll(monkeypatch) -> None:
    """hybrid 开启时 _hybrid_retrieve 用真实 store 的 scroll_all（非 mock 路径验证）。"""
    from app.rag.documents import RetrievedChunk

    class FakeStore:
        def __init__(self):
            self.chunks = [
                RetrievedChunk(
                    point_id="p1",
                    chunk_id="c1",
                    content="退货运费由商家承担",
                    metadata={"source": "refund-return-policy.md"},
                    score=0.9,
                )
            ]

        def scroll_all(self, *, batch_size=100):
            yield from self.chunks

        def query_similar(self, query_vector, **kw):
            return self.chunks

    store = FakeStore()
    import app.rag.pipeline as pipe

    monkeypatch.setattr(pipe, "_vector_store", lambda s, collection_name=None: store)
    monkeypatch.setattr(
        pipe,
        "_embedding_model",
        lambda s: type("E", (), {"embed_texts": lambda self, texts: [[0.1] * 8 for _ in texts], "dimension": 8})(),
    )
    chunks = pipe._hybrid_retrieve("退货运费", settings=_settings(rag_enable_hybrid=True))
    assert chunks and chunks[0].chunk_id == "c1"


def test_enhanced_pipeline_reroutes_config_missing(monkeypatch) -> None:
    """配置缺失（无 embedding key）时提升为 AppException 而非裸 ValueError。"""
    from app.core.exceptions import AppException

    import app.rag.pipeline as pipe

    def boom(*a, **kw):
        raise ValueError("embedding api key is missing")

    monkeypatch.setattr(pipe, "_retrieve", boom)
    try:
        pipe.enhanced_rag_answer("退款政策", settings=_settings())
    except AppException as exc:
        assert exc.code == "RAG_EMBEDDING_CONFIG_MISSING"
        return
    raise AssertionError("expected AppException for missing embedding config")


def test_context_compression_enabled_uses_compressed_chunks(monkeypatch) -> None:
    """RAG_ENABLE_CONTEXT_COMPRESSION=true：压缩结果传给生成器。"""
    from app.rag.documents import RetrievedChunk
    from app.rag.context_compression import ContextCompressionReport

    import app.rag.pipeline as pipe

    settings = _settings(rag_enable_context_compression=True)
    chunk = RetrievedChunk(
        point_id="p1",
        chunk_id="c1",
        content="退款政策七天无理由",
        metadata={"source": "refund-return-policy.md"},
        score=0.9,
    )
    monkeypatch.setattr(pipe, "_retrieve", lambda q, **kw: [chunk])
    monkeypatch.setattr(pipe, "_rerank", lambda q, chunks, settings: chunks)

    captured = {}

    def fake_compress(query, chunks, *, policy=None):
        captured["n_in"] = len(chunks)
        return ContextCompressionReport(
            query=query,
            budget_chars=1800,
            original_total_chars=10,
            final_total_chars=10,
            saved_chars=0,
            input_chunk_count=len(chunks),
            kept_chunk_count=len(chunks),
            compressed_chunk_count=len(chunks),
            dropped_chunk_count=0,
            compressed_chunks=list(chunks),
        )

    monkeypatch.setattr(pipe, "compress_retrieved_context", fake_compress)
    monkeypatch.setattr(
        pipe,
        "create_rag_answer_service",
        lambda s: type(
            "Svc",
            (),
            {"generate_answer_with_citations": lambda self, q, chunks: captured.__setitem__("n_out", len(chunks)) or (_fake_rag_answer())},
        )(),
    )
    pipe.enhanced_rag_answer("退款政策", settings=settings)
    assert captured["n_in"] == 1
    assert captured["n_out"] == 1


class _NoAccessDecision:
    def __init__(self) -> None:
        self.normalized_query = "退款政策是什么"
        self.intent = "knowledge"
        self.should_use_rag = True
        self.routes = []
        self.no_route_reason = "no accessible knowledge base matched this RAG query"
        self.warnings = [
            "RAG_ROUTE_NO_ACCESSIBLE_KNOWLEDGE_BASE",
            "RAG_ROUTE_CANDIDATE_FILTERED_BY_ACCESS_SCOPE",
        ]


class _NoAccessRouter:
    def route(self, query, *, access_scope=None, classification=None, max_routes=2):
        return _NoAccessDecision()


def test_routing_no_access_short_circuits_with_permission_denied(monkeypatch) -> None:
    from app.rag.filters import RagAccessScope
    from app.rag.generator import RagAnswerStatus

    calls: list[str] = []
    monkeypatch.setattr(pipeline_module, "create_knowledge_router", lambda s: _NoAccessRouter())
    monkeypatch.setattr(pipeline_module, "_retrieve", lambda q, **kw: calls.append("retrieve") or [])
    monkeypatch.setattr(
        pipeline_module,
        "create_rag_answer_service",
        lambda s: type(
            "Svc",
            (),
            {"generate_answer_with_citations": lambda self, q, chunks: (_fake_rag_answer())},
        )(),
    )

    answer = pipeline_module.enhanced_rag_answer(
        "退款政策是什么",
        settings=_settings(rag_enable_routing=True),
        access_scope=RagAccessScope(
            user_id="U1",
            tenant_id="t1",
            permission_groups=["customer_service"],
        ),
    )

    assert answer.status is RagAnswerStatus.PERMISSION_DENIED
    assert answer.permission_denied is True
    assert "权限" in answer.answer
    assert calls == []  # 未触达检索


def test_routing_no_access_without_permission_warning_keeps_old_path(monkeypatch) -> None:
    """非权限原因（如纯闲聊）无可达知识库：保持旧行为回退默认 collection 检索。"""

    class NoMatchDecision:
        def __init__(self) -> None:
            self.normalized_query = "在吗"
            self.intent = "chitchat"
            self.should_use_rag = False
            self.routes = []
            self.no_route_reason = "query intent does not use RAG retrieval"
            self.warnings = ["RAG_ROUTE_QUERY_INTENT_NOT_RAG"]

    class NoMatchRouter:
        def route(self, query, *, access_scope=None, classification=None, max_routes=2):
            return NoMatchDecision()

    calls: list[str] = []
    monkeypatch.setattr(pipeline_module, "create_knowledge_router", lambda s: NoMatchRouter())
    monkeypatch.setattr(pipeline_module, "_retrieve", lambda q, **kw: calls.append("retrieve") or [])
    monkeypatch.setattr(
        pipeline_module,
        "create_rag_answer_service",
        lambda s: type(
            "Svc",
            (),
            {"generate_answer_with_citations": lambda self, q, chunks: (_fake_rag_answer())},
        )(),
    )

    answer = pipeline_module.enhanced_rag_answer("在吗", settings=_settings(rag_enable_routing=True))

    assert answer.status.value == "answered"
    assert calls == ["retrieve"]
