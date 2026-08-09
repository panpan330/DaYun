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


def test_ticket_creation_via_internal_visible_in_java_queue() -> None:
    """Python JavaTicketClient（internal）创建工单 → MySQL 落库 → Java 公共列表可见。"""
    from app.core.config import get_settings
    from app.schemas.ticket import CreateTicketArgs, TicketCategory
    from app.services.java_ticket_client import JavaTicketClient

    title_hint = f"e2e-{_new_id()}"

    settings = get_settings()
    client = JavaTicketClient.from_settings(settings)
    created = client.create_ticket(
        CreateTicketArgs(
            requester_id="U1001",
            title=f"端到端工单 {title_hint}",
            description="物流未到货，请跟进处理。",
            category=TicketCategory.LOGISTICS,
            # 不传 related_order_id：Java 会校验订单存在，随机单号会 404/拒绝
        ),
        idempotency_key=uuid.uuid4().hex,
    )
    assert created.ticket_id

    with httpx.Client(base_url=JAVA, timeout=15) as java:
        tickets = java.get("/api/tickets", headers=_auth("U1001")).json()
        assert tickets.get("success") is True
        data = tickets.get("data") or []
        matched = [t for t in data if title_hint in (t.get("title") or "")]
        assert matched, f"ticket with {title_hint} not found in java queue"
