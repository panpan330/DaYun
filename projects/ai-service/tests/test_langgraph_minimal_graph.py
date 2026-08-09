from app.agents.minimal_graph import (
    MINIMAL_GRAPH_CONDITIONAL_ROUTES,
    MINIMAL_GRAPH_EDGES,
    build_minimal_graph_input,
    build_blank_reply_node,
    build_minimal_graph,
    build_ready_reply_node,
    classify_message_node,
    normalize_message_node,
    route_by_message_status,
    run_minimal_graph,
    stream_minimal_graph_updates,
    stream_minimal_graph_values,
)
from langgraph.graph import END, START


def test_minimal_graph_edges_define_fixed_parts_of_execution_order() -> None:
    assert MINIMAL_GRAPH_EDGES == (
        (START, "normalize_message"),
        ("normalize_message", "classify_message"),
        ("build_ready_reply", END),
        ("build_blank_reply", END),
    )


def test_minimal_graph_conditional_routes_map_status_to_reply_node() -> None:
    assert MINIMAL_GRAPH_CONDITIONAL_ROUTES == {
        "ready": "build_ready_reply",
        "blank": "build_blank_reply",
        "stop": END,
    }


def test_route_by_message_status_returns_ready_route() -> None:
    route = route_by_message_status({"message_status": "ready"})

    assert route == "ready"


def test_route_by_message_status_defaults_to_blank_route() -> None:
    route = route_by_message_status({})

    assert route == "blank"


def test_route_by_message_status_returns_stop_route() -> None:
    route = route_by_message_status({"message_status": "stop"})

    assert route == "stop"


def test_run_minimal_graph_returns_updated_state() -> None:
    result = run_minimal_graph("  你好，LangGraph  ")

    assert result["user_message"] == "  你好，LangGraph  "
    assert result["normalized_message"] == "你好，LangGraph"
    assert result["message_status"] == "ready"
    assert result["reply"] == "你说的是：你好，LangGraph"
    assert result["node_history"] == [
        "normalize_message",
        "classify_message",
        "build_ready_reply",
    ]


def test_build_minimal_graph_input_returns_initial_state() -> None:
    initial_state = build_minimal_graph_input("hello")

    assert initial_state == {"user_message": "hello", "node_history": []}


def test_stream_minimal_graph_updates_returns_node_updates() -> None:
    chunks = stream_minimal_graph_updates("  hello  ")

    assert chunks == [
        {
            "type": "updates",
            "ns": (),
            "data": {
                "normalize_message": {
                    "normalized_message": "hello",
                    "node_history": ["normalize_message"],
                }
            },
        },
        {
            "type": "updates",
            "ns": (),
            "data": {
                "classify_message": {
                    "message_status": "ready",
                    "node_history": ["classify_message"],
                }
            },
        },
        {
            "type": "updates",
            "ns": (),
            "data": {
                "build_ready_reply": {
                    "reply": "你说的是：hello",
                    "node_history": ["build_ready_reply"],
                }
            },
        },
    ]


def test_stream_minimal_graph_updates_shows_stop_route_without_reply_node() -> None:
    chunks = stream_minimal_graph_updates("  /stop  ")

    assert chunks == [
        {
            "type": "updates",
            "ns": (),
            "data": {
                "normalize_message": {
                    "normalized_message": "/stop",
                    "node_history": ["normalize_message"],
                }
            },
        },
        {
            "type": "updates",
            "ns": (),
            "data": {
                "classify_message": {
                    "message_status": "stop",
                    "node_history": ["classify_message"],
                }
            },
        },
    ]


def test_stream_minimal_graph_values_returns_accumulated_state() -> None:
    chunks = stream_minimal_graph_values("  hello  ")

    assert chunks[-1] == {
        "type": "values",
        "ns": (),
        "data": {
            "user_message": "  hello  ",
            "normalized_message": "hello",
            "message_status": "ready",
            "reply": "你说的是：hello",
            "node_history": [
                "normalize_message",
                "classify_message",
                "build_ready_reply",
            ],
        },
        "interrupts": (),
    }


def test_run_minimal_graph_handles_blank_message() -> None:
    result = run_minimal_graph("   ")

    assert result["normalized_message"] == ""
    assert result["message_status"] == "blank"
    assert result["reply"] == "你还没有输入内容。"
    assert result["node_history"] == [
        "normalize_message",
        "classify_message",
        "build_blank_reply",
    ]


def test_run_minimal_graph_can_end_after_classification() -> None:
    result = run_minimal_graph("  /stop  ")

    assert result["normalized_message"] == "/stop"
    assert result["message_status"] == "stop"
    assert "reply" not in result
    assert result["node_history"] == [
        "normalize_message",
        "classify_message",
    ]


def test_build_minimal_graph_can_compile_independent_graph() -> None:
    graph = build_minimal_graph()

    result = graph.invoke({"user_message": "  test  ", "node_history": []})

    assert result["normalized_message"] == "test"
    assert result["message_status"] == "ready"
    assert result["reply"] == "你说的是：test"


def test_normalize_message_node_returns_only_its_state_update() -> None:
    update = normalize_message_node(
        {"user_message": "  hello  ", "reply": "old", "node_history": []}
    )

    assert update == {
        "normalized_message": "hello",
        "node_history": ["normalize_message"],
    }


def test_classify_message_node_returns_status_update() -> None:
    update = classify_message_node({"normalized_message": "hello"})

    assert update == {
        "message_status": "ready",
        "node_history": ["classify_message"],
    }


def test_classify_message_node_can_mark_stop_message() -> None:
    update = classify_message_node({"normalized_message": "/stop"})

    assert update == {
        "message_status": "stop",
        "node_history": ["classify_message"],
    }


def test_build_ready_reply_node_returns_normal_reply() -> None:
    update = build_ready_reply_node({"normalized_message": "hello"})

    assert update == {
        "reply": "你说的是：hello",
        "node_history": ["build_ready_reply"],
    }


def test_build_blank_reply_node_returns_blank_reply() -> None:
    update = build_blank_reply_node({})

    assert update == {
        "reply": "你还没有输入内容。",
        "node_history": ["build_blank_reply"],
    }
