import json
from pathlib import Path

import pytest

from app.agents.checkpoint_store import (
    FileTicketAgentCheckpointStore,
    TICKET_AGENT_CHECKPOINT_FILE_INVALID_CODE,
    TICKET_AGENT_CHECKPOINT_THREAD_ID_INVALID_CODE,
    TICKET_AGENT_CHECKPOINT_VALUES_INVALID_CODE,
    TicketAgentCheckpointSnapshot,
    build_checkpoint_snapshot_filename,
    normalize_checkpoint_thread_id,
)
from app.agents.ticket_agent import (
    build_checkpointed_ticket_agent_graph,
    build_ticket_agent_checkpoint_snapshot,
    run_ticket_agent_in_thread,
    save_ticket_agent_checkpoint_snapshot,
)
from app.core.exceptions import AppException
from tests.tool_fakes import FakeTicketCreator


def test_normalize_checkpoint_thread_id_strips_and_rejects_blank() -> None:
    assert normalize_checkpoint_thread_id(" ticket-thread-001 ") == "ticket-thread-001"

    with pytest.raises(AppException) as exc_info:
        normalize_checkpoint_thread_id("   ")

    assert exc_info.value.code == TICKET_AGENT_CHECKPOINT_THREAD_ID_INVALID_CODE
    assert exc_info.value.message == "thread_id 不能为空。"


def test_build_checkpoint_snapshot_filename_is_safe_and_collision_resistant() -> None:
    unsafe_name = build_checkpoint_snapshot_filename("../ticket/thread:001")
    similar_name = build_checkpoint_snapshot_filename("ticket_thread_001")

    assert unsafe_name.endswith(".json")
    assert "/" not in unsafe_name
    assert "\\" not in unsafe_name
    assert ":" not in unsafe_name
    assert unsafe_name != similar_name


def test_file_checkpoint_store_saves_and_loads_utf8_json(
    tmp_path: Path,
) -> None:
    store = FileTicketAgentCheckpointStore(tmp_path)
    snapshot = TicketAgentCheckpointSnapshot.create(
        thread_id=" ticket-thread-001 ",
        values={
            "final_answer": "已经为你生成待确认工单。",
            "node_history": ["normalize_user_input", "request_ticket_confirmation"],
        },
        metadata={"checkpoint_kind": "pending_confirmation"},
        saved_at="2026-07-25T00:00:00+00:00",
    )

    path = store.save(snapshot)
    raw = json.loads(path.read_text(encoding="utf-8"))
    loaded = store.load("ticket-thread-001")

    assert raw["values"]["final_answer"] == "已经为你生成待确认工单。"
    assert loaded == snapshot
    assert store.load("missing-thread") is None


def test_file_checkpoint_store_rejects_non_json_values(tmp_path: Path) -> None:
    store = FileTicketAgentCheckpointStore(tmp_path)
    snapshot = TicketAgentCheckpointSnapshot.create(
        thread_id="ticket-thread-bad-json",
        values={"file_handle": object()},
        saved_at="2026-07-25T00:00:00+00:00",
    )

    with pytest.raises(AppException) as exc_info:
        store.save(snapshot)

    assert exc_info.value.code == TICKET_AGENT_CHECKPOINT_VALUES_INVALID_CODE


def test_file_checkpoint_store_rejects_mismatched_thread_id(tmp_path: Path) -> None:
    store = FileTicketAgentCheckpointStore(tmp_path)
    path = store.build_path("ticket-thread-001")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            TicketAgentCheckpointSnapshot.create(
                thread_id="ticket-thread-002",
                values={"final_answer": "wrong thread"},
                saved_at="2026-07-25T00:00:00+00:00",
            ).to_json_dict(),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(AppException) as exc_info:
        store.load("ticket-thread-001")

    assert exc_info.value.code == TICKET_AGENT_CHECKPOINT_FILE_INVALID_CODE


def test_build_ticket_agent_checkpoint_snapshot_reads_current_thread_state() -> None:
    graph = build_checkpointed_ticket_agent_graph(ticket_creator=FakeTicketCreator())
    run_ticket_agent_in_thread(
        graph,
        "我要投诉订单 1001，物流一直不动",
        thread_id="ticket-thread-snapshot-001",
        actor_id="demo_user_001",
    )

    snapshot = build_ticket_agent_checkpoint_snapshot(
        graph,
        thread_id="ticket-thread-snapshot-001",
        metadata={"source": "unit_test"},
    )

    assert snapshot.thread_id == "ticket-thread-snapshot-001"
    assert snapshot.metadata == {"source": "unit_test"}
    assert snapshot.values["pending_ticket_confirmation"]["status"] == "pending"
    assert snapshot.values["ticket_actor_id"] == "demo_user_001"


def test_save_ticket_agent_checkpoint_snapshot_persists_current_thread_state(
    tmp_path: Path,
) -> None:
    graph = build_checkpointed_ticket_agent_graph(ticket_creator=FakeTicketCreator())
    store = FileTicketAgentCheckpointStore(tmp_path)
    thread_id = "ticket-thread-persist-001"
    result = run_ticket_agent_in_thread(
        graph,
        "我要投诉订单 1001，物流一直不动",
        thread_id=thread_id,
        actor_id="demo_user_001",
    )

    path = save_ticket_agent_checkpoint_snapshot(
        graph,
        thread_id=thread_id,
        store=store,
        metadata={"checkpoint_kind": "pending_confirmation"},
    )
    loaded = store.load(thread_id)

    assert path.exists()
    assert loaded is not None
    assert loaded.values["pending_ticket_confirmation"] == result[
        "pending_ticket_confirmation"
    ]
    assert loaded.values["node_history"] == [
        "normalize_user_input",
        "emotion_recognition",
        "classify_intent",
        "decide_ticket_need",
        "extract_ticket_fields",
        "request_ticket_confirmation",
    ]
    assert loaded.metadata == {"checkpoint_kind": "pending_confirmation"}
