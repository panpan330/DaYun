import uuid

import httpx
import pytest

pytestmark = pytest.mark.e2e

AI = "http://127.0.0.1:8000"
JAVA = "http://127.0.0.1:18004"


def _auth(user_id: str) -> dict[str, str]:
    return {"Authorization": f"Bearer local-dev-token:default:{user_id}"}


def _new_id() -> str:
    return uuid.uuid4().hex[:8]


def test_handoff_full_flow() -> None:
    conversation_id = f"e2e-handoff-{_new_id()}"
    with httpx.Client(base_url=AI, timeout=120) as ai, httpx.Client(base_url=JAVA, timeout=15) as java:
        # 1. 触发转人工（愤怒消息）
        response = ai.post(
            "/api/ai/agent/conversations",
            headers=_auth("U1001"),
            json={
                "message": "气死我了！我的订单 202501010001 还不发货！",
                "conversation_id": conversation_id,
            },
        )
        assert response.status_code == 200
        assert response.json().get("human_handoff") is not None

        # 2. 队列 pending 含该会话
        queue = java.get("/api/human-handoffs?status=pending", headers=_auth("S1001")).json()
        handoff = next((h for h in queue["data"] if h["conversation_id"] == conversation_id), None)
        assert handoff is not None, f"handoff {conversation_id} not in pending queue"
        handoff_id = handoff["id"]

        # 3. claim（A1001）
        claimed = java.post(f"/api/human-handoffs/{handoff_id}/claim", headers=_auth("A1001")).json()
        assert claimed["data"]["status"] == "in_progress"
        assert claimed["data"]["assigned_agent"] == "A1001"

        # 4. transfer（A1001 → A1002）
        transferred = java.post(
            f"/api/human-handoffs/{handoff_id}/transfer",
            headers=_auth("A1001"),
            json={"target_agent": "A1002"},
        ).json()
        assert transferred["data"]["assigned_agent"] == "A1002"
        assert transferred["data"]["status"] == "in_progress"
        assert "转交" in (transferred["data"]["note"] or "")

        # 5. human-reply（supervisor S1001 代答）
        replied = ai.post(
            f"/api/ai/agent/conversations/{conversation_id}/human-reply",
            headers=_auth("S1001"),
            json={"content": "您好，我是人工客服，您的订单已加急。"},
        )
        assert replied.status_code == 200

        # 6. history 含 human_agent
        history = ai.get(
            f"/api/ai/agent/conversations/{conversation_id}/history",
            headers=_auth("U1001"),
        ).json()
        assert any(m["role"] == "human_agent" for m in history["messages"])

        # 7. summary
        summary = ai.get(
            f"/api/ai/agent/conversations/{conversation_id}/summary",
            headers=_auth("S1001"),
        )
        assert summary.status_code == 200
        assert summary.json()["source"] in ("llm", "rule")

        # 8. close
        closed = java.post(
            f"/api/human-handoffs/{handoff_id}/close",
            headers=_auth("S1001"),
            json={"note": "已电话联系客户"},
        ).json()
        assert closed["data"]["status"] == "closed"
