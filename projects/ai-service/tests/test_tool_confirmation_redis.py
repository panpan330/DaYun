from datetime import datetime, timedelta, timezone

import pytest

from app.core.config import Settings
from app.core.exceptions import AppException
from app.schemas.tool_confirmation import ToolConfirmationStatus
from app.tools.tool_confirmation import (
    RedisToolConfirmationStore,
    ToolConfirmationRecord,
    ToolConfirmationStore,
    create_tool_confirmation_store,
)


class FakeRedisClient:
    """Minimal dict-backed redis client with the methods the store uses."""

    def __init__(self) -> None:
        self.data: dict[str, str] = {}
        self.set_ex: dict[str, int | None] = {}

    def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.data[key] = value
        self.set_ex[key] = ex

    def get(self, key: str) -> str | None:
        return self.data.get(key)

    def delete(self, key: str) -> int:
        return 1 if self.data.pop(key, None) is not None else 0

    def scan_iter(self, match: str = "*") -> list[str]:
        # 真实 redis-py 的 scan_iter 返回 generator；SCAN 是快照语义，
        # 因此迭代期间 delete 不会破坏遍历。
        prefix = match.rstrip("*")
        return (k for k in list(self.data) if k.startswith(prefix))

    def expire(self, key: str, seconds: int) -> None:
        return None


def _clock() -> datetime:
    return datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_redis_store_register_confirmed_roundtrip() -> None:
    store = RedisToolConfirmationStore(
        FakeRedisClient(),
        clock=_clock,
    )
    record = store.register_confirmed(
        confirmation_id="a" * 16,
        actor_id="user_001",
        tool_name="create_ticket",
        arguments={"title": "t"},
        ttl_seconds=300,
    )
    assert record.confirmation_id == "a" * 16
    assert record.status == ToolConfirmationStatus.CONFIRMED

    verified = store.require_confirmed("a" * 16, actor_id="user_001")
    assert verified.confirmation_id == "a" * 16
    assert store.count() == 1


def test_redis_store_require_confirmed_rejects_unknown() -> None:
    store = RedisToolConfirmationStore(FakeRedisClient(), clock=_clock)
    with pytest.raises(AppException) as exc:
        store.require_confirmed("b" * 16, actor_id="user_001")
    assert exc.value.code == "TOOL_CONFIRMATION_NOT_FOUND"


def test_redis_store_require_confirmed_rejects_wrong_actor() -> None:
    store = RedisToolConfirmationStore(FakeRedisClient(), clock=_clock)
    store.register_confirmed(
        confirmation_id="c" * 16,
        actor_id="user_001",
        tool_name="create_ticket",
        arguments={},
        ttl_seconds=300,
    )
    with pytest.raises(AppException) as exc:
        store.require_confirmed("c" * 16, actor_id="user_002")
    assert exc.value.code == "TOOL_CONFIRMATION_FORBIDDEN"


def test_redis_store_clear_removes_all() -> None:
    store = RedisToolConfirmationStore(FakeRedisClient(), clock=_clock)
    store.register_confirmed(
        confirmation_id="d" * 16,
        actor_id="user_001",
        tool_name="create_ticket",
        arguments={},
        ttl_seconds=300,
    )
    store.clear()
    assert store.count() == 0


def test_redis_store_count_handles_generator_scan_iter() -> None:
    # 真实 redis-py 的 scan_iter 返回 generator；count() 不能对其使用 len()。
    store = RedisToolConfirmationStore(FakeRedisClient(), clock=_clock)
    store.register_confirmed(
        confirmation_id="i" * 16,
        actor_id="user_001",
        tool_name="create_ticket",
        arguments={},
        ttl_seconds=300,
    )
    store.register_confirmed(
        confirmation_id="j" * 16,
        actor_id="user_001",
        tool_name="create_ticket",
        arguments={},
        ttl_seconds=300,
    )
    assert store.count() == 2


def test_redis_store_confirm_resets_ttl_to_remaining() -> None:
    class MutableClock:
        def __init__(self) -> None:
            self.value = datetime(2026, 1, 1, tzinfo=timezone.utc)

        def __call__(self) -> datetime:
            return self.value

        def advance(self, *, seconds: int) -> None:
            self.value += timedelta(seconds=seconds)

    clock = MutableClock()
    fake = FakeRedisClient()
    store = RedisToolConfirmationStore(fake, clock=clock)
    pending = store.create(
        actor_id="user_001",
        tool_name="create_ticket",
        arguments={},
        ttl_seconds=300,
    )
    clock.advance(seconds=120)
    store.confirm(pending.confirmation_id, actor_id="user_001")
    key = store._key(pending.confirmation_id)
    assert fake.set_ex[key] == 180  # 剩余 TTL，而非完整 300


def test_redis_store_register_confirmed_rejects_other_actor_overwrite() -> None:
    store = RedisToolConfirmationStore(FakeRedisClient(), clock=_clock)
    store.register_confirmed(
        confirmation_id="e" * 16,
        actor_id="user_001",
        tool_name="create_ticket",
        arguments={},
        ttl_seconds=300,
    )
    with pytest.raises(AppException) as exc:
        store.register_confirmed(
            confirmation_id="e" * 16,
            actor_id="user_002",
            tool_name="create_ticket",
            arguments={},
            ttl_seconds=300,
        )
    assert exc.value.code == "TOOL_CONFIRMATION_FORBIDDEN"
    assert exc.value.status_code == 403


def test_redis_store_register_confirmed_idempotent_for_same_actor() -> None:
    store = RedisToolConfirmationStore(FakeRedisClient(), clock=_clock)
    store.register_confirmed(
        confirmation_id="f" * 16,
        actor_id="user_001",
        tool_name="create_ticket",
        arguments={},
        ttl_seconds=300,
    )
    record = store.register_confirmed(
        confirmation_id="f" * 16,
        actor_id="user_001",
        tool_name="create_ticket",
        arguments={},
        ttl_seconds=300,
    )
    assert record.status == ToolConfirmationStatus.CONFIRMED
    assert store.count() == 1


def test_memory_store_register_confirmed_rejects_other_actor_overwrite() -> None:
    store = ToolConfirmationStore()
    store.register_confirmed(
        confirmation_id="m" * 16,
        actor_id="user_001",
        tool_name="create_ticket",
        arguments={},
        ttl_seconds=300,
    )
    with pytest.raises(AppException) as exc:
        store.register_confirmed(
            confirmation_id="m" * 16,
            actor_id="user_002",
            tool_name="create_ticket",
            arguments={},
            ttl_seconds=300,
        )
    assert exc.value.code == "TOOL_CONFIRMATION_FORBIDDEN"
    assert exc.value.status_code == 403


def test_redis_store_store_rejects_non_serializable_arguments() -> None:
    store = RedisToolConfirmationStore(FakeRedisClient(), clock=_clock)
    record = ToolConfirmationRecord(
        confirmation_id="h" * 16,
        status=ToolConfirmationStatus.CONFIRMED,
        actor_id="user_001",
        tool_name="create_ticket",
        arguments={"when": datetime(2026, 1, 1, tzinfo=timezone.utc)},
        arguments_fingerprint="fp",
        created_at=_clock(),
        expires_at=_clock() + timedelta(seconds=300),
    )
    with pytest.raises(AppException) as exc:
        store._store(record, ttl_seconds=300)
    assert exc.value.code == "TOOL_CONFIRMATION_SERIALIZATION_FAILED"
    assert exc.value.status_code == 422


def test_create_tool_confirmation_store_returns_memory_by_default() -> None:
    store = create_tool_confirmation_store(Settings(_env_file=None))
    assert store.count() == 0  # memory singleton usable without redis


def test_create_tool_confirmation_store_returns_redis_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = create_tool_confirmation_store(
        Settings(
            _env_file=None,
            tool_confirmation_backend="redis",
            agent_redis_url="redis://redis.example:6379/3",
        )
    )
    assert isinstance(store, RedisToolConfirmationStore)
