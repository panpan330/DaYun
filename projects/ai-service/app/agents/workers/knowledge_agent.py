"""Knowledge-base QA worker subgraph."""

from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agents.emotion_classifier import CustomerEmotionClassifier
from app.agents.multi_agent_states import KnowledgeWorkerState
from app.agents.ticket_agent import (
    PolicyRagService,
    decide_ticket_need_node,
    retrieve_policy_node,
)
from app.agents.workers.emotion_node import make_emotion_recognition_node, route_after_emotion


def build_knowledge_agent_graph(
    service: PolicyRagService | None = None,
    *,
    checkpointer: Any | None = None,
    emotion_classifier: CustomerEmotionClassifier | None = None,
) -> Any:
    builder = StateGraph(KnowledgeWorkerState)
    builder.add_node(
        "emotion_recognition",
        make_emotion_recognition_node(emotion_classifier),
    )
    builder.add_node(
        "retrieve_policy",
        lambda state: retrieve_policy_node(state, service=service),
    )
    builder.add_node("decide_ticket_need", decide_ticket_need_node)
    builder.add_edge(START, "emotion_recognition")
    builder.add_conditional_edges(
        "emotion_recognition",
        route_after_emotion,
        {"handoff": END, "continue": "retrieve_policy"},
    )
    builder.add_edge("retrieve_policy", "decide_ticket_need")

    # decide_ticket_need_node 输出 needs_ticket(bool) + ticket_need_source。
    # 本子图不含 create_ticket 节点，needs_ticket=True（含 RAG no_context 转工单）
    # 时子图结束，由监督层 after_knowledge_agent 检查后转 ticket_agent 子图。
    def _route_after_ticket_need(state: KnowledgeWorkerState) -> str:
        return "finish"

    builder.add_conditional_edges(
        "decide_ticket_need",
        _route_after_ticket_need,
        {"finish": END},
    )
    return builder.compile(checkpointer=checkpointer)
