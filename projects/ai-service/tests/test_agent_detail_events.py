"""Tests for agent detail (retrieval/tool/model) SSE events."""

from app.services.console_agent_service import build_agent_detail_events


class _Settings:
    def __init__(self, agent_detail_events_enabled: bool = True) -> None:
        self.agent_detail_events_enabled = agent_detail_events_enabled
        self.resolved_llm_model = "qwen3.7-max"


def test_retrieval_detail_from_citations():
    events = build_agent_detail_events(
        "retrieve_policy",
        {"rag_citations": [{"source": "物流政策.md", "chunk_index": 0}]},
        _Settings(),
    )
    assert events == [
        {"event": "detail", "data": {"kind": "retrieval", "kb": "物流政策.md", "hits": 1}}
    ]


def test_tool_detail_from_order_query():
    events = build_agent_detail_events(
        "query_order",
        {"order_query_order_id": "20250101", "intent": "order_query"},
        _Settings(),
    )
    assert events == [
        {"event": "detail", "data": {"kind": "tool", "tool": "query_order", "order_id": "20250101"}}
    ]


def test_model_detail_after_answer():
    events = build_agent_detail_events(
        "build_direct_answer", {"intent": "smalltalk"}, _Settings()
    )
    assert events == [
        {"event": "detail", "data": {"kind": "model", "model": "qwen3.7-max"}}
    ]


def test_no_detail_when_disabled():
    events = build_agent_detail_events(
        "retrieve_policy",
        {"rag_citations": [{"source": "x"}]},
        _Settings(agent_detail_events_enabled=False),
    )
    assert events == []


def test_no_detail_when_missing_data():
    events = build_agent_detail_events("retrieve_policy", {}, _Settings())
    assert events == []
