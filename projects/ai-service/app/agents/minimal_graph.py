from operator import add
from typing import Annotated, Any, Literal
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph


MessageStatus = Literal["blank", "ready", "stop"]
MessageRoute = Literal["ready", "blank", "stop"]
MinimalGraphStreamPart = dict[str, Any]

MINIMAL_GRAPH_EDGES: tuple[tuple[str, str], ...] = (
    (START, "normalize_message"),
    ("normalize_message", "classify_message"),
    ("build_ready_reply", END),
    ("build_blank_reply", END),
)

MINIMAL_GRAPH_CONDITIONAL_ROUTES: dict[MessageRoute, str] = {
    "ready": "build_ready_reply",
    "blank": "build_blank_reply",
    "stop": END,
}


class MinimalGraphState(TypedDict, total=False):
    """State shared by the minimal LangGraph learning example."""

    user_message: str
    normalized_message: str
    message_status: MessageStatus
    reply: str
    node_history: Annotated[list[str], add]


def normalize_message_node(state: MinimalGraphState) -> MinimalGraphState:
    user_message = state.get("user_message", "")

    return {
        "normalized_message": user_message.strip(),
        "node_history": ["normalize_message"],
    }


def classify_message_node(state: MinimalGraphState) -> MinimalGraphState:
    normalized_message = state.get("normalized_message", "")
    if normalized_message == "/stop":
        message_status: MessageStatus = "stop"
    elif normalized_message:
        message_status = "ready"
    else:
        message_status = "blank"

    return {
        "message_status": message_status,
        "node_history": ["classify_message"],
    }


def route_by_message_status(state: MinimalGraphState) -> MessageRoute:
    message_status = state.get("message_status")
    if message_status == "ready":
        return "ready"
    if message_status == "stop":
        return "stop"
    return "blank"


def build_ready_reply_node(state: MinimalGraphState) -> MinimalGraphState:
    normalized_message = state.get("normalized_message", "")

    return {
        "reply": f"你说的是：{normalized_message}",
        "node_history": ["build_ready_reply"],
    }


def build_blank_reply_node(state: MinimalGraphState) -> MinimalGraphState:
    return {
        "reply": "你还没有输入内容。",
        "node_history": ["build_blank_reply"],
    }


def build_minimal_graph():
    builder = StateGraph(MinimalGraphState)

    builder.add_node("normalize_message", normalize_message_node)
    builder.add_node("classify_message", classify_message_node)
    builder.add_node("build_ready_reply", build_ready_reply_node)
    builder.add_node("build_blank_reply", build_blank_reply_node)

    for start_node, end_node in MINIMAL_GRAPH_EDGES:
        builder.add_edge(start_node, end_node)

    builder.add_conditional_edges(
        "classify_message",
        route_by_message_status,
        MINIMAL_GRAPH_CONDITIONAL_ROUTES,
    )

    return builder.compile()


minimal_graph = build_minimal_graph()


def build_minimal_graph_input(user_message: str) -> MinimalGraphState:
    return {
        "user_message": user_message,
        "node_history": [],
    }


def run_minimal_graph(user_message: str) -> MinimalGraphState:
    return minimal_graph.invoke(build_minimal_graph_input(user_message))


def stream_minimal_graph_updates(user_message: str) -> list[MinimalGraphStreamPart]:
    return list(
        minimal_graph.stream(
            build_minimal_graph_input(user_message),
            stream_mode="updates",
            version="v2",
        )
    )


def stream_minimal_graph_values(user_message: str) -> list[MinimalGraphStreamPart]:
    return list(
        minimal_graph.stream(
            build_minimal_graph_input(user_message),
            stream_mode="values",
            version="v2",
        )
    )
