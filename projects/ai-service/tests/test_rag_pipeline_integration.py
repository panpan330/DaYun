"""Integration tests for RAG pipeline switch combinations (feature switches)."""

from __future__ import annotations

from types import SimpleNamespace

from app.rag import pipeline as pipeline_module


class _FakeAnswerService:
    def generate_answer_with_citations(self, query, *, chunks, memory_context=None):
        from app.rag.generator import RagAnswer, RagAnswerStatus

        return RagAnswer(answer="生成回答", status=RagAnswerStatus.ANSWERED)


def _fake_settings(**overrides):
    from app.core.config import Settings

    base = dict(
        rag_enable_rewrite=False,
        rag_enable_multi_query=False,
        rag_enable_routing=False,
        rag_enable_hybrid=False,
        rag_enable_context_compression=False,
        rag_enable_citation_verify=False,
    )
    base.update(overrides)
    return Settings(_env_file=None, **base)


def _decision(*, should_use_rag, routes, warnings=None):
    return SimpleNamespace(
        normalized_query="政策",
        intent="knowledge",
        should_use_rag=should_use_rag,
        routes=routes,
        no_route_reason="no accessible knowledge base matched this RAG query",
        warnings=warnings or [],
    )


def _route(collection_name: str):
    return SimpleNamespace(
        collection_name=collection_name,
        knowledge_base_id="kb",
        display_name="知识库",
    )


def _install_fakes(monkeypatch, *, calls):
    monkeypatch.setattr(pipeline_module, "create_rag_answer_service", lambda s: _FakeAnswerService())
    monkeypatch.setattr(pipeline_module, "_retrieve", lambda q, **kw: calls.append(("vector", q)) or [])
    monkeypatch.setattr(pipeline_module, "_hybrid_retrieve", lambda q, **kw: calls.append(("hybrid", q)) or [])
    monkeypatch.setattr(pipeline_module, "_rerank", lambda q, chunks, settings: chunks)
    return calls


def test_all_switches_off_is_old_path(monkeypatch):
    calls: list = []
    _install_fakes(monkeypatch, calls=calls)

    pipeline_module.enhanced_rag_answer("政策", settings=_fake_settings())

    assert calls == [("vector", "政策")]


def test_routing_selects_collection_and_hybrid_runs(monkeypatch):
    calls: list = []
    _install_fakes(monkeypatch, calls=calls)

    class FakeRouter:
        def route(self, query, *, access_scope=None, classification=None, max_routes=2):
            return _decision(should_use_rag=True, routes=[_route("kb_customer_policy")])

    monkeypatch.setattr(pipeline_module, "create_knowledge_router", lambda s: FakeRouter())

    pipeline_module.enhanced_rag_answer(
        "政策", settings=_fake_settings(rag_enable_routing=True, rag_enable_hybrid=True),
    )

    assert calls[0] == ("hybrid", "政策")  # 混合检索被调用


def test_rewrite_changes_query_before_retrieval(monkeypatch):
    calls: list = []
    _install_fakes(monkeypatch, calls=calls)

    class FakeRewriter:
        def rewrite(self, query):
            return SimpleNamespace(rewritten_query="改写后政策")

    monkeypatch.setattr(pipeline_module, "create_query_rewriter", lambda s: FakeRewriter())

    pipeline_module.enhanced_rag_answer(
        "政策", settings=_fake_settings(rag_enable_rewrite=True),
    )

    assert calls[0] == ("vector", "改写后政策")


def test_hybrid_off_routing_on_uses_vector_on_default_collection(monkeypatch):
    calls: list = []
    _install_fakes(monkeypatch, calls=calls)

    class FakeRouter:
        def route(self, query, *, access_scope=None, classification=None, max_routes=2):
            return _decision(should_use_rag=True, routes=[_route("kb_customer_policy")])

    monkeypatch.setattr(pipeline_module, "create_knowledge_router", lambda s: FakeRouter())

    pipeline_module.enhanced_rag_answer(
        "政策", settings=_fake_settings(rag_enable_routing=True),
    )

    assert calls[0] == ("vector", "政策")  # hybrid 关 → 向量检索，collection 已选


def test_no_access_short_circuit_does_not_touch_retrieval(monkeypatch):
    calls: list = []
    _install_fakes(monkeypatch, calls=calls)

    class NoAccessRouter:
        def route(self, query, *, access_scope=None, classification=None, max_routes=2):
            return _decision(
                should_use_rag=True,
                routes=[],
                warnings=["RAG_ROUTE_NO_ACCESSIBLE_KNOWLEDGE_BASE"],
            )

    monkeypatch.setattr(pipeline_module, "create_knowledge_router", lambda s: NoAccessRouter())

    pipeline_module.enhanced_rag_answer(
        "政策", settings=_fake_settings(rag_enable_routing=True),
    )

    assert calls == []  # 不触达任何检索
