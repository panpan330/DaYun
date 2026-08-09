from app.core.config import Settings
from app.schemas.console_agent import ConsoleAgentResponse
from app.services.console_agent_conversation_store import ConsoleAgentConversationStore
from app.services.console_agent_service import ConsoleAgentActor
from app.services.java_conversation_client import JavaConversation, JavaConversationMessage


class FakeRedisPipeline:
    def __init__(self, redis: "FakeRedis") -> None:
        self.redis = redis

    def __enter__(self) -> "FakeRedisPipeline":
        return self

    def __exit__(self, *_: object) -> None:
        pass

    def set(self, key: str, value: str, *, ex: int) -> "FakeRedisPipeline":
        self.redis.values[key] = value
        self.redis.expirations[key] = ex
        return self

    def zadd(self, key: str, values: dict[str, float]) -> "FakeRedisPipeline":
        self.redis.sorted_sets.setdefault(key, {}).update(values)
        return self

    def expire(self, key: str, seconds: int) -> "FakeRedisPipeline":
        self.redis.expirations[key] = seconds
        return self

    def execute(self) -> None:
        pass


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.sorted_sets: dict[str, dict[str, float]] = {}
        self.expirations: dict[str, int] = {}

    def close(self) -> None:
        pass

    def get(self, key: str) -> str | None:
        return self.values.get(key)

    def zrevrange(self, key: str, start: int, end: int) -> list[str]:
        values = self.sorted_sets.get(key, {})
        ordered = [item[0] for item in sorted(values.items(), key=lambda item: item[1], reverse=True)]
        return ordered[start : end + 1]

    def zrem(self, key: str, *members: str) -> None:
        values = self.sorted_sets.get(key, {})
        for member in members:
            values.pop(member, None)

    def pipeline(self, *, transaction: bool) -> FakeRedisPipeline:
        assert transaction is True
        return FakeRedisPipeline(self)


def _response(conversation_id: str, reply: str) -> ConsoleAgentResponse:
    return ConsoleAgentResponse(
        reply=reply,
        conversation_id=conversation_id,
        trace_id=f"trace-{conversation_id}",
        route="order_query",
    )


def test_append_human_reply_stores_human_agent_message() -> None:
    redis = FakeRedis()
    settings = Settings(
        _env_file=None,
        agent_redis_url="redis://redis.example:6379/0",
        agent_checkpoint_ttl_minutes=45,
        agent_checkpoint_key_prefix="test-agent",
    )
    store = ConsoleAgentConversationStore(settings, client=redis)  # type: ignore[arg-type]
    owner = ConsoleAgentActor(user_id="U1001", tenant_id="default", roles=("customer",))

    store.append_human_reply(
        actor=owner,
        conversation_id="conv-h1",
        content="我是人工客服，您的问题已解决",
    )
    conversation = store.get(actor=owner, conversation_id="conv-h1")
    assert conversation is not None
    assert conversation.messages[-1].role == "human_agent"
    assert conversation.messages[-1].content == "我是人工客服，您的问题已解决"


def test_store_persists_only_owner_visible_messages_and_recent_summaries() -> None:
    redis = FakeRedis()
    settings = Settings(
        _env_file=None,
        agent_redis_url="redis://redis.example:6379/0",
        agent_checkpoint_ttl_minutes=45,
        agent_checkpoint_key_prefix="test-agent",
    )
    store = ConsoleAgentConversationStore(settings, client=redis)  # type: ignore[arg-type]
    owner = ConsoleAgentActor(user_id="U1001", tenant_id="default", roles=("customer",))
    another_user = ConsoleAgentActor(user_id="U2001", tenant_id="default", roles=("customer",))

    store.append_exchange(
        actor=owner,
        conversation_id="conversation-001",
        user_message="Please check order A1001.",
        response=_response("conversation-001", "Order A1001 is in transit."),
    )
    store.append_exchange(
        actor=owner,
        conversation_id="conversation-002",
        user_message="I need refund help.",
        response=_response("conversation-002", "I can help with the refund policy."),
    )

    conversation = store.get(actor=owner, conversation_id="conversation-001")

    assert conversation is not None
    assert [message.role for message in conversation.messages] == ["user", "assistant"]
    assert conversation.messages[1].trace_id == "trace-conversation-001"
    assert store.get(actor=another_user, conversation_id="conversation-001") is None
    assert [item.conversation_id for item in store.list_recent(actor=owner, limit=20)] == [
        "conversation-002",
        "conversation-001",
    ]
    assert store.list_recent(actor=another_user, limit=20) == []
    assert set(redis.expirations.values()) == {45 * 60}


def test_store_keeps_existing_transcript_when_appending_new_exchange() -> None:
    redis = FakeRedis()
    store = ConsoleAgentConversationStore(
        Settings(_env_file=None, agent_redis_url="redis://redis.example:6379/0"),
        client=redis,  # type: ignore[arg-type]
    )
    actor = ConsoleAgentActor(user_id="U1001", tenant_id="default", roles=("customer",))

    store.append_exchange(
        actor=actor,
        conversation_id="conversation-001",
        user_message="First question",
        response=_response("conversation-001", "First answer"),
    )
    store.append_exchange(
        actor=actor,
        conversation_id="conversation-001",
        user_message="Second question",
        response=_response("conversation-001", "Second answer"),
    )

    conversation = store.get(actor=actor, conversation_id="conversation-001")

    assert conversation is not None
    assert [message.content for message in conversation.messages] == [
        "First question",
        "First answer",
        "Second question",
        "Second answer",
    ]


class FakeJavaClient:
    def __init__(self, conversations: dict[str, dict], messages: dict[str, list[dict]]) -> None:
        self.conversations = {
            cid: JavaConversation(
                conversation_id=str(item["conversation_id"]),
                user_id=str(item["user_id"]),
                title=str(item["title"]),
                conversation_status=str(item["conversation_status"]),
                updated_at=str(item["updated_at"]),
            )
            for cid, item in conversations.items()
        }
        self.messages = {
            cid: [
                JavaConversationMessage(
                    message_id=str(item["message_id"]),
                    conversation_id=str(item["conversation_id"]),
                    sender_type=str(item["sender_type"]),
                    content=str(item["content"]),
                    trace_id=str(item["trace_id"]),
                    created_at=str(item["created_at"]),
                )
                for item in items
            ]
            for cid, items in messages.items()
        }
        self.list_calls = 0
        self.get_messages_calls = 0

    def list_conversations(self, *, tenant_id: str, user_id: str, limit: int = 20) -> list:
        self.list_calls += 1
        items = list(self.conversations.values())
        items.sort(key=lambda item: item.updated_at, reverse=True)
        return items[:limit]

    def get_messages(self, *, tenant_id: str, conversation_id: str) -> list:
        self.get_messages_calls += 1
        return self.messages.get(conversation_id, [])


def _make_store(redis: FakeRedis, client: FakeJavaClient | None = None) -> ConsoleAgentConversationStore:
    settings = Settings(
        _env_file=None,
        agent_redis_url="redis://redis.example:6379/0",
        agent_checkpoint_ttl_minutes=45,
        agent_checkpoint_key_prefix="test-agent",
    )
    return ConsoleAgentConversationStore(settings, client=redis, java_client=client)  # type: ignore[arg-type]


def _java_message(message_id: str, sender_type: str, content: str, created_at: str) -> dict:
    return {
        "message_id": message_id,
        "conversation_id": "conv-backfill",
        "sender_type": sender_type,
        "content": content,
        "trace_id": f"trace-{message_id}",
        "created_at": created_at,
    }


def test_get_backfills_from_java_when_redis_miss() -> None:
    redis = FakeRedis()
    java = FakeJavaClient(
        conversations={},
        messages={
            "conv-backfill": [
                _java_message("m-1", "user", "历史问题", "2026-08-01T09:00:00Z"),
                _java_message("m-2", "assistant", "历史回答", "2026-08-01T09:01:00Z"),
            ]
        },
    )
    store = _make_store(redis, java)
    actor = ConsoleAgentActor(user_id="U1001", tenant_id="default", roles=("customer",))

    conversation = store.get(actor=actor, conversation_id="conv-backfill")
    second = store.get(actor=actor, conversation_id="conv-backfill")

    assert conversation is not None
    assert [message.role for message in conversation.messages] == ["user", "assistant"]
    assert conversation.messages[0].content == "历史问题"
    assert conversation.messages[1].content == "历史回答"
    assert conversation.messages[1].trace_id == "trace-m-2"
    assert second is not None
    assert java.get_messages_calls == 1  # 回填后命中 Redis，不再调 Java


def test_get_returns_none_when_java_has_no_messages() -> None:
    redis = FakeRedis()
    java = FakeJavaClient(conversations={}, messages={})
    store = _make_store(redis, java)
    actor = ConsoleAgentActor(user_id="U1001", tenant_id="default", roles=("customer",))

    assert store.get(actor=actor, conversation_id="conv-empty") is None


def test_list_recent_backfills_index_from_java_when_zset_empty() -> None:
    redis = FakeRedis()
    java = FakeJavaClient(
        conversations={
            "conv-old": {
                "conversation_id": "conv-old",
                "user_id": "U1001",
                "title": "旧会话",
                "conversation_status": "active",
                "updated_at": "2026-08-01T09:00:00Z",
            },
            "conv-new": {
                "conversation_id": "conv-new",
                "user_id": "U1001",
                "title": "新会话",
                "conversation_status": "completed",
                "updated_at": "2026-08-08T09:00:00Z",
            },
        },
        messages={
            "conv-old": [_java_message("mo-1", "user", "旧会话问题", "2026-08-01T09:00:00Z")],
            "conv-new": [_java_message("mn-1", "user", "新会话问题", "2026-08-08T09:00:00Z")],
        },
    )
    store = _make_store(redis, java)
    actor = ConsoleAgentActor(user_id="U1001", tenant_id="default", roles=("customer",))

    summaries = store.list_recent(actor=actor, limit=20)
    second_call = store.list_recent(actor=actor, limit=20)

    assert [item.conversation_id for item in summaries] == ["conv-new", "conv-old"]
    assert [item.title for item in summaries] == ["新会话", "旧会话"]
    assert second_call == summaries
    assert java.list_calls == 1  # 重建索引后命中 Redis zset


def test_list_recent_does_not_call_java_when_zset_present() -> None:
    redis = FakeRedis()
    java = FakeJavaClient(conversations={}, messages={})
    store = _make_store(redis, java)
    actor = ConsoleAgentActor(user_id="U1001", tenant_id="default", roles=("customer",))
    store.append_exchange(
        actor=actor,
        conversation_id="conv-1",
        user_message="hi",
        response=_response("conv-1", "hello"),
    )

    store.list_recent(actor=actor, limit=20)

    assert java.list_calls == 0


def test_backfill_degrades_gracefully_when_java_client_raises() -> None:
    class ExplodingClient(FakeJavaClient):
        def list_conversations(self, *, tenant_id: str, user_id: str, limit: int = 20) -> list:
            raise RuntimeError("java down")

        def get_messages(self, *, tenant_id: str, conversation_id: str) -> list:
            raise RuntimeError("java down")

    redis = FakeRedis()
    store = _make_store(redis, ExplodingClient(conversations={}, messages={}))
    actor = ConsoleAgentActor(user_id="U1001", tenant_id="default", roles=("customer",))

    assert store.get(actor=actor, conversation_id="conv-x") is None
    assert store.list_recent(actor=actor, limit=20) == []
