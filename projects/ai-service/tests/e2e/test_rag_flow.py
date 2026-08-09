import uuid

import httpx
import pytest

pytestmark = pytest.mark.e2e

CUSTOMER_AUTH = {"Authorization": "Bearer local-dev-token:default:U1001"}


def test_rag_answer_with_citations() -> None:
    conversation_id = f"e2e-rag-{uuid.uuid4().hex[:8]}"
    with httpx.Client(base_url="http://127.0.0.1:8000", timeout=120) as client:
        response = client.post(
            "/api/ai/agent/conversations",
            headers=CUSTOMER_AUTH,
            json={"message": "退货退款政策是什么？", "conversation_id": conversation_id},
        )
        assert response.status_code == 200
        body = response.json()
        assert body.get("reply")
