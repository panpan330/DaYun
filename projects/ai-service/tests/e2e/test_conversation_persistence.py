import time
import uuid

import httpx
import pytest
import redis

pytestmark = pytest.mark.e2e

AI = "http://127.0.0.1:8000"
JAVA = "http://127.0.0.1:18004"

CUSTOMER_AUTH = {"Authorization": "Bearer local-dev-token:default:U1001"}


def _internal_headers() -> dict[str, str]:
    from uuid import uuid4

    from app.core.business_context import build_java_internal_headers
    from app.core.config import get_settings
    from app.core.trace import build_trace_headers

    settings = get_settings()
    headers = build_trace_headers()
    headers.update(build_java_internal_headers(settings))
    if "X-Trace-Id" not in headers:
        headers["X-Trace-Id"] = f"sync-{uuid4().hex[:24]}"
    return headers


def _new_id() -> str:
    return uuid.uuid4().hex[:8]


def test_conversation_persists_to_mysql_and_backfills_from_history() -> None:
    conversation_id = f"e2e-persist-{_new_id()}"
    with httpx.Client(base_url=AI, timeout=120) as ai, httpx.Client(base_url=JAVA, timeout=15) as java:
        response = ai.post(
            "/api/ai/agent/conversations",
            headers=CUSTOMER_AUTH,
            json={"message": f"查一下订单 {conversation_id} 的状态", "conversation_id": conversation_id},
        )
        assert response.status_code == 200

        # 1. 等批刷落 MySQL（重试 ≤ 6 次 × 10s）
        persisted = False
        for _ in range(6):
            messages = java.get(
                f"/internal/ai-conversations/{conversation_id}/messages",
                headers=_internal_headers(),
            )
            if messages.status_code == 200 and messages.json().get("data"):
                persisted = True
                break
            time.sleep(10)
        assert persisted, f"conversation {conversation_id} not persisted to MySQL after batch sync"

        # 2. 删 Redis 会话 key → history 回源
        settings = None
        from app.core.config import get_settings

        settings = get_settings()
        redis_client = redis.Redis.from_url(settings.resolved_agent_redis_url)
        keys = [
            key.decode()
            for key in redis_client.keys(f"*{conversation_id}*")
            if b"conversation" in key
        ]
        for key in keys:
            redis_client.delete(key)

        history = ai.get(
            f"/api/ai/agent/conversations/{conversation_id}/history",
            headers=CUSTOMER_AUTH,
        ).json()
        assert any(m["role"] == "user" for m in history.get("messages", []))
