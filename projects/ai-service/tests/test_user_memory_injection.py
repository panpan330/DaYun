"""Tests for memory injection into agent input and LLM message construction."""

from app.agents.ticket_agent import (
    build_ticket_agent_input,
    build_ticket_intent_classification_messages,
)
from app.rag.generator import build_rag_messages


def test_input_carries_memory_context():
    state = build_ticket_agent_input("你好", memory_context=["用户称呼：小明"])
    assert state["memory_context"] == ["用户称呼：小明"]


def test_classify_messages_include_memory_section():
    messages = build_ticket_intent_classification_messages(
        "你好", memory_context=["用户称呼：小明", "用户地址：朝阳区"]
    )
    content = messages[1]["content"]
    assert "用户已知信息" in content
    assert "用户称呼：小明" in content
    assert "用户地址：朝阳区" in content


def test_classify_messages_without_memory_unchanged():
    messages = build_ticket_intent_classification_messages("你好")
    assert "用户已知信息" not in messages[1]["content"]


def test_rag_messages_include_memory_section():
    messages = build_rag_messages(
        "退货政策是什么",
        chunks=[],
        memory_context=["用户称呼：小明"],
    )
    assert "用户已知信息" in messages[0]["content"]
    assert "用户称呼：小明" in messages[0]["content"]


def test_rag_messages_without_memory_unchanged():
    messages = build_rag_messages("退货政策是什么", chunks=[])
    assert "用户已知信息" not in messages[0]["content"]
