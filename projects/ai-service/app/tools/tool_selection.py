"""工具动态按需注入：规则层 + LLM 意图兜底。

核心原则：写工具（refund/cancel/create）需要"业务动词 + 订单号/明确对象"双信号
才暴露给模型，防止模型臆造订单号调用写工具；兜底只暴露只读工具 query_order。
"""
from __future__ import annotations

import re
from typing import Any

# 只读工具永远可用；写工具需要强信号
READONLY_TOOL = "query_order"

ORDER_ID_PATTERNS = (
    re.compile(r"\b[A-Za-z]\d{3,}\b"),   # A1001 / B1002
    re.compile(r"\b\d{6,12}\b"),         # 纯数字订单号
)

# 业务动词 → 写工具
VERB_TO_TOOLS: dict[tuple[str, ...], str] = {
    ("退款", "退钱", "退货", "申请退款"): "refund_order",
    ("取消", "退订"): "cancel_order",
    ("创建工单", "报修", "投诉", "开单"): "create_ticket",
}

_QUERY_VERBS = ("查", "查询", "物流", "到哪", "进度", "状态")

# 开放疑问词：可能隐含业务意图 → 规则不确定，交 LLM 层
_OPEN_QUESTION_WORDS = ("怎么", "什么", "咋", "如何", "为啥")

# 意图 → 工具集（LLM 层映射）
INTENT_TO_TOOLS: dict[str, list[str]] = {
    "order_query": ["query_order"],
    "refund_request": ["query_order", "refund_order"],
    "cancel_request": ["query_order", "cancel_order"],
    "ticket_request": ["create_ticket", "query_order"],
    "policy_question": ["query_order"],
    "smalltalk": ["query_order"],
    "unsupported": ["query_order"],
    "unclear": ["query_order"],
}


def extract_order_ids(text: str) -> list[str]:
    found: list[str] = []
    for pattern in ORDER_ID_PATTERNS:
        for match in pattern.findall(text):
            if match not in found:
                found.append(match)
    return found


def _has_query_verb(message: str) -> bool:
    return any(verb in message for verb in _QUERY_VERBS)


def select_by_rules(
    message: str,
    history: list[str] | None = None,
) -> list[str] | None:
    """规则层：命中业务信号返回工具集；明确无业务返回只读；可能隐含业务返回 None 交 LLM。"""
    history_text = "\n".join(history or [])
    order_ids = extract_order_ids(message) + extract_order_ids(history_text)
    has_order_context = bool(order_ids)
    has_query = _has_query_verb(message)

    matched_write: list[str] = []
    write_verb_hit = False
    for keywords, tool_name in VERB_TO_TOOLS.items():
        if any(keyword in message for keyword in keywords):
            write_verb_hit = True
            if has_order_context:
                matched_write.append(tool_name)
            # 无订单号：保守，不暴露写工具

    if has_order_context or has_query or write_verb_hit:
        tools = [READONLY_TOOL] + matched_write
        return list(dict.fromkeys(tools))

    # 无业务信号：开放疑问词 → 可能隐含业务，交 LLM 层；否则明确无工具需求
    if any(word in message for word in _OPEN_QUESTION_WORDS):
        return None
    return [READONLY_TOOL]


def _intent_result_to_tools(intent_result: Any) -> list[str]:
    intent = intent_result.get("intent") if isinstance(intent_result, dict) else None
    if intent and intent in INTENT_TO_TOOLS:
        return list(INTENT_TO_TOOLS[intent])
    return [READONLY_TOOL]


def select_by_intent(
    message: str,
    classifier: Any,
) -> list[str]:
    """LLM 层：意图 → 工具映射；分类失败/未知意图回退只读。"""
    try:
        return _intent_result_to_tools(classifier.classify_intent(message))
    except Exception:
        return [READONLY_TOOL]


def build_default_tool_selection_classifier(mode: str, settings: Any) -> Any:
    from app.agents.ticket_agent import (
        FakeLLMTicketIntentClassifier,
        LLMTicketIntentClassifier,
        RuleBasedTicketIntentClassifier,
    )

    if mode == "fake_llm":
        return FakeLLMTicketIntentClassifier()
    if mode == "real_llm":
        return LLMTicketIntentClassifier(settings)
    return RuleBasedTicketIntentClassifier()


def select_tool_names(
    message: str,
    history: list[str] | None = None,
    *,
    classifier: Any | None = None,
) -> list[str]:
    """组合：规则强命中优先 → LLM 意图兜底 → 空则只读。"""
    rule_tools = select_by_rules(message, history)
    if rule_tools is not None:
        return rule_tools
    if classifier is not None:
        return select_by_intent(message, classifier)
    return [READONLY_TOOL]
