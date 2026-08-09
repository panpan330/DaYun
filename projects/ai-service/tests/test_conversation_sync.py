from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from app.schemas.console_agent import ConsoleAgentConversation, ConsoleAgentConversationMessage
from app.services.conversation_sync import PendingConversationSync


class FakeRedis:
    def __init__(self) -> None:
        self.members: set[str] = set()
        self.sadd_calls: list[str] = []

    def sadd(self, key: str, value: str) -> None:
        self.sadd_calls.append(value)
        self.members.add(value)

    def smembers(self, key: str) -> set[str]:
        return set(self.members)

    def srem(self, key: str, value: str) -> None:
        self.members.discard(value)


class FakeStore:
    def __init__(self, conversation: ConsoleAgentConversation | None) -> None:
        self.conversation = conversation
        self.get_calls: list[str] = []

    def get(self, *, actor, conversation_id: str) -> ConsoleAgentConversation | None:
        self.get_calls.append(conversation_id)
        return self.conversation


class FakeClient:
    def __init__(self) -> None:
        self.upserts: list[dict] = []
        self.message_batches: list[dict] = []
        self.cleanups: list[dict] = []
        self.fail_next_upsert = False
        self.fail_next_batch = False

    def upsert_conversation(self, **kwargs) -> None:
        if self.fail_next_upsert:
            self.fail_next_upsert = False
            raise RuntimeError("java down")
        self.upserts.append(kwargs)

    def batch_write_messages(self, *, tenant_id, conversation_id, messages) -> int:
        if self.fail_next_batch:
            self.fail_next_batch = False
            raise RuntimeError("java down")
        self.message_batches.append({"tenant_id": tenant_id, "conversation_id": conversation_id, "messages": messages})
        return len(messages)

    def request_cleanup(self, *, older_than_days: int = 30) -> int:
        self.cleanups.append({"older_than_days": older_than_days})
        return 0


def make_conversation(conversation_id: str = "conv-1") -> ConsoleAgentConversation:
    now = datetime.now(timezone.utc)
    return ConsoleAgentConversation(
        conversation_id=conversation_id,
        title="我的订单",
        updated_at=now,
        messages=[
            ConsoleAgentConversationMessage(id="m-1", role="user", content="你好", created_at=now),
            ConsoleAgentConversationMessage(id="m-2", role="assistant", content="您好", created_at=now),
        ],
    )


def actor_for(token: str) -> SimpleNamespace:
    tenant_id, user_id, conversation_id = token.split(":", 2)
    return SimpleNamespace(tenant_id=tenant_id, user_id=user_id)


def make_sync(*, store: FakeStore, client: FakeClient, redis: FakeRedis) -> PendingConversationSync:
    return PendingConversationSync(
        store=store,
        client=client,
        redis=redis,
        interval_seconds=30.0,
    )


def test_mark_pending_writes_to_redis_set():
    redis = FakeRedis()
    sync = make_sync(store=FakeStore(None), client=FakeClient(), redis=redis)

    sync.mark_pending(tenant_id="default", user_id="U1001", conversation_id="conv-1")

    assert redis.sadd_calls == ["default:U1001:conv-1"]
    assert "default:U1001:conv-1" in redis.members


def test_sync_once_upserts_conversation_and_batches_messages_then_clears_marker():
    redis = FakeRedis()
    redis.members.add("default:U1001:conv-1")
    store = FakeStore(make_conversation())
    client = FakeClient()
    sync = make_sync(store=store, client=client, redis=redis)

    synced = sync.sync_once()

    assert synced == 1
    assert client.upserts == [
        {
            "tenant_id": "default",
            "conversation_id": "conv-1",
            "user_id": "U1001",
            "title": "我的订单",
            "conversation_status": "active",
        }
    ]
    assert len(client.message_batches) == 1
    batch = client.message_batches[0]
    assert batch["tenant_id"] == "default"
    assert batch["conversation_id"] == "conv-1"
    assert [m["message_id"] for m in batch["messages"]] == ["m-1", "m-2"]
    assert batch["messages"][0]["sender_type"] == "user"
    assert batch["messages"][1]["sender_type"] == "assistant"
    assert redis.members == set()


def test_sync_once_keeps_marker_when_client_fails():
    redis = FakeRedis()
    redis.members.add("default:U1001:conv-1")
    store = FakeStore(make_conversation())
    client = FakeClient()
    client.fail_next_upsert = True
    sync = make_sync(store=store, client=client, redis=redis)

    sync.sync_once()

    assert "default:U1001:conv-1" in redis.members
    assert client.upserts == []


def test_sync_once_keeps_marker_when_message_batch_fails():
    redis = FakeRedis()
    redis.members.add("default:U1001:conv-1")
    store = FakeStore(make_conversation())
    client = FakeClient()
    client.fail_next_batch = True
    sync = make_sync(store=store, client=client, redis=redis)

    sync.sync_once()

    assert "default:U1001:conv-1" in redis.members
    assert len(client.upserts) == 1
    assert client.message_batches == []


def test_sync_once_is_noop_when_no_pending_markers():
    redis = FakeRedis()
    store = FakeStore(make_conversation())
    client = FakeClient()
    sync = make_sync(store=store, client=client, redis=redis)

    synced = sync.sync_once()

    assert synced == 0
    assert client.upserts == []
    assert client.message_batches == []
    assert store.get_calls == []


def test_sync_once_skips_marker_when_transcript_expired_in_redis():
    redis = FakeRedis()
    redis.members.add("default:U1001:conv-expired")
    store = FakeStore(None)
    client = FakeClient()
    sync = make_sync(store=store, client=client, redis=redis)

    synced = sync.sync_once()

    assert synced == 0
    assert client.upserts == []
    assert redis.members == set()


def test_sync_once_handles_multiple_pending_conversations():
    redis = FakeRedis()
    redis.members.add("default:U1001:conv-1")
    redis.members.add("default:U1001:conv-2")
    store = FakeStore(make_conversation())
    client = FakeClient()
    sync = make_sync(store=store, client=client, redis=redis)

    synced = sync.sync_once()

    assert synced == 2
    assert sorted(u["conversation_id"] for u in client.upserts) == ["conv-1", "conv-2"]
    assert redis.members == set()


def test_sync_once_does_not_duplicate_work_for_already_synced():
    redis = FakeRedis()
    redis.members.add("default:U1001:conv-1")
    store = FakeStore(make_conversation())
    client = FakeClient()
    sync = make_sync(store=store, client=client, redis=redis)

    sync.sync_once()
    sync.sync_once()

    assert len(client.upserts) == 1
    assert len(client.message_batches) == 1


def test_sync_once_handles_bytes_markers_from_real_redis():
    """redis-py returns bytes from smembers on real connections; markers must decode."""
    from app.services.conversation_sync import PENDING_SYNC_KEY

    class BytesFakeRedis(FakeRedis):
        def smembers(self, key: str) -> set[bytes]:
            return {b"default:U1001:e2e-bytes"}

        def srem(self, key: str, value) -> None:
            self.members.discard(value)

    redis = BytesFakeRedis()
    redis.members = {b"default:U1001:e2e-bytes"}
    store = FakeStore(make_conversation())
    sync = make_sync(store=store, client=FakeClient(), redis=redis)

    result = sync.sync_once()

    assert result == 1
    assert store.get_calls == ["e2e-bytes"]
