"""Tests for user memory get/clear endpoints."""

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import create_app

AUTH = {"Authorization": "Bearer local-dev-token:default:U1001"}


@pytest.fixture()
def client(monkeypatch):
    app = create_app(get_settings())

    from app.routers.chat import get_console_agent_actor
    from app.services.console_agent_service import ConsoleAgentActor

    app.dependency_overrides[get_console_agent_actor] = lambda: ConsoleAgentActor(
        user_id="U1001", tenant_id="default", roles=("customer",)
    )

    class _FakeMemoryStore:
        def __init__(self) -> None:
            self.facts: list[dict] = []

        def get_facts(self, actor):
            return self.facts

        def clear(self, actor) -> None:
            self.facts = []

    fake = _FakeMemoryStore()

    import app.services.user_memory as user_memory_module

    monkeypatch.setattr(user_memory_module.UserMemoryStore, "get_facts", fake.get_facts)
    monkeypatch.setattr(user_memory_module.UserMemoryStore, "clear", fake.clear)
    return TestClient(app), fake


def test_get_memory_returns_facts(client):
    test_client, fake = client
    fake.facts = [{"fact": "用户称呼：小明", "source_conversation_id": "c1"}]
    resp = test_client.get("/api/ai/agent/memory", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["facts"] == [
        {"fact": "用户称呼：小明", "source_conversation_id": "c1"}
    ]


def test_delete_memory_clears(client):
    test_client, fake = client
    fake.facts = [{"fact": "用户称呼：小明"}]
    resp = test_client.delete("/api/ai/agent/memory", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["cleared"] is True
    assert fake.facts == []


def test_memory_requires_auth():
    app = create_app(get_settings())
    resp = TestClient(app).get("/api/ai/agent/memory")
    assert resp.status_code == 401
