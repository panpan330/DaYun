"""Order query worker subgraph."""

from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agents.emotion_classifier import CustomerEmotionClassifier
from app.agents.multi_agent_states import OrderWorkerState
from app.agents.ticket_agent import OrderQueryExecutor, query_order_node
from app.agents.workers.emotion_node import make_emotion_recognition_node, route_after_emotion


def build_order_agent_graph(
    order_query_executor: OrderQueryExecutor | None = None,
    *,
    checkpointer: Any | None = None,
    emotion_classifier: CustomerEmotionClassifier | None = None,
) -> Any:
    builder = StateGraph(OrderWorkerState)
    builder.add_node(
        "emotion_recognition",
        make_emotion_recognition_node(emotion_classifier),
    )
    builder.add_node(
        "query_order",
        lambda state: query_order_node(
            state,
            order_query_executor=order_query_executor,
        ),
    )
    builder.add_edge(START, "emotion_recognition")
    builder.add_conditional_edges(
        "emotion_recognition",
        route_after_emotion,
        {"handoff": END, "continue": "query_order"},
    )
    builder.add_edge("query_order", END)
    return builder.compile(checkpointer=checkpointer)
