"""工具动态按需注入测试：规则层 / LLM 意图兜底 / 组合。"""

from typing import Any

from app.tools.tool_selection import (
    INTENT_TO_TOOLS,
    extract_order_ids,
    select_by_rules,
    select_tool_names,
)


# ---- 规则层 ----

def test_extract_order_id_a_prefix():
    assert extract_order_ids("我的订单 A1001 退款") == ["A1001"]


def test_extract_order_id_numeric():
    assert extract_order_ids("订单 100123456 查询") == ["100123456"]


def test_extract_order_id_no_match():
    assert extract_order_ids("你好，退款多久到账") == []


def test_rules_order_id_plus_refund_verb():
    tools = select_by_rules("订单 A1001 我要退款", [])
    assert tools is not None
    assert "query_order" in tools
    assert "refund_order" in tools


def test_rules_order_id_only_query():
    tools = select_by_rules("帮我查一下 A1001 的物流", [])
    assert tools is not None
    assert "query_order" in tools
    assert "refund_order" not in tools
    assert "cancel_order" not in tools


def test_rules_refund_verb_without_order_id_is_conservative():
    tools = select_by_rules("我要退款", [])
    assert tools is not None
    assert "refund_order" not in tools  # 无订单号 → 不暴露写工具


def test_rules_smalltalk_returns_query_only():
    tools = select_by_rules("你好，在吗", [])
    assert tools is not None
    assert tools == ["query_order"]


def test_rules_history_order_id_enables_write_tool():
    tools = select_by_rules("帮我退款", ["之前查过订单 A1001"])
    assert tools is not None
    assert "refund_order" in tools  # 历史订单号增强


def test_rules_cancel_verb_with_order_id():
    tools = select_by_rules("取消订单 B1002", [])
    assert tools is not None
    assert "cancel_order" in tools


def test_rules_uncertain_returns_none():
    # 无任何业务信号 → 不确定（交给 LLM 层），返回 None
    tools = select_by_rules("这个怎么弄", [])
    assert tools is None


# ---- LLM 层与组合 ----

class FakeIntentClassifier:
    def __init__(self, intent: str) -> None:
        self._intent = intent

    def classify_intent(self, message: str) -> Any:
        return {"intent": self._intent, "reason": "fake"}


def test_intent_to_tools_mapping():
    assert "query_order" in INTENT_TO_TOOLS["refund_request"]
    assert "refund_order" in INTENT_TO_TOOLS["refund_request"]
    assert "query_order" in INTENT_TO_TOOLS["order_query"]
    assert "cancel_order" in INTENT_TO_TOOLS["cancel_request"]


def test_select_tool_names_rules_win_over_llm():
    # 规则强命中（A1001+退款）优先于 LLM
    tools = select_tool_names(
        "订单 A1001 我要退款",
        [],
        classifier=FakeIntentClassifier("smalltalk"),
    )
    assert "refund_order" in tools


def test_select_tool_names_llm_fallback_when_rules_uncertain():
    tools = select_tool_names(
        "这个怎么弄",
        [],
        classifier=FakeIntentClassifier("order_query"),
    )
    assert tools == ["query_order"]


def test_select_tool_names_empty_fallback_query_only():
    tools = select_tool_names("随便聊聊", [], classifier=None)
    assert tools == ["query_order"]


def test_select_tool_names_default_classifier_rule_based():
    # classifier=None 且无强规则信号 → 默认 rule 分类器兜底
    tools = select_tool_names("退款多久到账？", [])
    assert "query_order" in tools
