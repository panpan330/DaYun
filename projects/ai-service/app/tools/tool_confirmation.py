import json
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Any
from uuid import uuid4

from app.core.config import Settings
from app.core.exceptions import AppException
from app.schemas.tool_confirmation import ToolConfirmationStatus
from app.tools.idempotency import build_arguments_fingerprint


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class ToolConfirmationRecord:
    confirmation_id: str
    status: ToolConfirmationStatus
    actor_id: str
    tool_name: str
    arguments: dict[str, Any]
    arguments_fingerprint: str
    created_at: datetime
    expires_at: datetime


class ToolConfirmationStore:
    def __init__(self, clock: Callable[[], datetime] = utc_now) -> None:
        self._clock = clock
        self._lock = Lock()
        self._records: dict[str, ToolConfirmationRecord] = {}

    def create(
        self,
        *,
        actor_id: str,
        tool_name: str,
        arguments: Mapping[str, Any],
        ttl_seconds: int,
    ) -> ToolConfirmationRecord:
        created_at = self._clock()
        stored_arguments = deepcopy(dict(arguments))
        record = ToolConfirmationRecord(
            confirmation_id=uuid4().hex,
            status=ToolConfirmationStatus.PENDING,
            actor_id=actor_id,
            tool_name=tool_name,
            arguments=stored_arguments,
            arguments_fingerprint=build_arguments_fingerprint(
                tool_name,
                stored_arguments,
            ),
            created_at=created_at,
            expires_at=created_at + timedelta(seconds=ttl_seconds),
        )
        with self._lock:
            self._records[record.confirmation_id] = record
        return deepcopy(record)

    def confirm(
        self,
        confirmation_id: str,
        *,
        actor_id: str,
    ) -> ToolConfirmationRecord:
        with self._lock:
            record = self._records.get(confirmation_id)
            if record is None:
                raise AppException(
                    code="TOOL_CONFIRMATION_NOT_FOUND",
                    message="确认请求不存在或已失效。",
                    status_code=404,
                )

            if record.actor_id != actor_id:
                raise AppException(
                    code="TOOL_CONFIRMATION_FORBIDDEN",
                    message="当前操作者不能确认其他人的工具请求。",
                    status_code=403,
                )

            if self._clock() >= record.expires_at:
                raise AppException(
                    code="TOOL_CONFIRMATION_EXPIRED",
                    message="确认请求已过期，请重新发起操作。",
                    status_code=409,
                )

            if record.status == ToolConfirmationStatus.PENDING:
                record.status = ToolConfirmationStatus.CONFIRMED

            return deepcopy(record)

    def register_confirmed(
        self,
        *,
        confirmation_id: str,
        actor_id: str,
        tool_name: str,
        arguments: Mapping[str, Any],
        ttl_seconds: int,
    ) -> ToolConfirmationRecord:
        """Register a confirmation as already confirmed (id supplied by caller)."""
        created_at = self._clock()
        stored_arguments = deepcopy(dict(arguments))
        record = ToolConfirmationRecord(
            confirmation_id=confirmation_id,
            status=ToolConfirmationStatus.CONFIRMED,
            actor_id=actor_id,
            tool_name=tool_name,
            arguments=stored_arguments,
            arguments_fingerprint=build_arguments_fingerprint(
                tool_name,
                stored_arguments,
            ),
            created_at=created_at,
            expires_at=created_at + timedelta(seconds=ttl_seconds),
        )
        with self._lock:
            existing = self._records.get(confirmation_id)
            if existing is not None and existing.actor_id != actor_id:
                raise AppException(
                    code="TOOL_CONFIRMATION_FORBIDDEN",
                    message="当前操作者不能覆盖其他人的确认记录。",
                    status_code=403,
                )
            self._records[record.confirmation_id] = record
        return deepcopy(record)

    def require_confirmed(
        self,
        confirmation_id: str,
        *,
        actor_id: str,
    ) -> ToolConfirmationRecord:
        with self._lock:
            record = self._records.get(confirmation_id)
            if record is None:
                raise AppException(
                    code="TOOL_CONFIRMATION_NOT_FOUND",
                    message="确认请求不存在或已失效。",
                    status_code=404,
                )

            if record.actor_id != actor_id:
                raise AppException(
                    code="TOOL_CONFIRMATION_FORBIDDEN",
                    message="当前操作者不能执行其他人的工具请求。",
                    status_code=403,
                )

            if self._clock() >= record.expires_at:
                raise AppException(
                    code="TOOL_CONFIRMATION_EXPIRED",
                    message="确认请求已过期，请重新发起操作。",
                    status_code=409,
                )

            if record.status != ToolConfirmationStatus.CONFIRMED:
                raise AppException(
                    code="TOOL_CONFIRMATION_REQUIRED",
                    message="该工具请求尚未获得用户确认。",
                    status_code=409,
                )

            return deepcopy(record)

    def clear(self) -> None:
        with self._lock:
            self._records.clear()

    def count(self) -> int:
        with self._lock:
            return len(self._records)


_TOOL_CONFIRMATION_STORE = ToolConfirmationStore()


def get_tool_confirmation_store() -> ToolConfirmationStore:
    return _TOOL_CONFIRMATION_STORE


def clear_tool_confirmation_store() -> None:
    _TOOL_CONFIRMATION_STORE.clear()


class RedisToolConfirmationStore:
    """ToolConfirmationStore backed by redis, shared across processes."""

    def __init__(
        self,
        redis_client: Any,
        *,
        key_prefix: str = "ai-service:tool-confirmation",
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self._redis = redis_client
        self._key_prefix = key_prefix
        self._clock = clock

    def _key(self, confirmation_id: str) -> str:
        return f"{self._key_prefix}:{confirmation_id}"

    def _load(self, confirmation_id: str) -> ToolConfirmationRecord | None:
        raw = self._redis.get(self._key(confirmation_id))
        if raw is None:
            return None
        payload = json.loads(raw)
        return ToolConfirmationRecord(
            confirmation_id=payload["confirmation_id"],
            status=ToolConfirmationStatus(payload["status"]),
            actor_id=payload["actor_id"],
            tool_name=payload["tool_name"],
            arguments=payload["arguments"],
            arguments_fingerprint=payload["arguments_fingerprint"],
            created_at=datetime.fromisoformat(payload["created_at"]),
            expires_at=datetime.fromisoformat(payload["expires_at"]),
        )

    def _store(self, record: ToolConfirmationRecord, ttl_seconds: int) -> None:
        payload = {
            "confirmation_id": record.confirmation_id,
            "status": record.status.value,
            "actor_id": record.actor_id,
            "tool_name": record.tool_name,
            "arguments": record.arguments,
            "arguments_fingerprint": record.arguments_fingerprint,
            "created_at": record.created_at.isoformat(),
            "expires_at": record.expires_at.isoformat(),
        }
        try:
            raw_value = json.dumps(payload, ensure_ascii=True)
        except TypeError as exc:
            raise AppException(
                code="TOOL_CONFIRMATION_SERIALIZATION_FAILED",
                message="确认记录参数包含无法序列化的类型，无法存储。",
                status_code=422,
            ) from exc
        self._redis.set(
            self._key(record.confirmation_id),
            raw_value,
            ex=ttl_seconds,
        )

    def _require_record(self, confirmation_id: str) -> ToolConfirmationRecord:
        record = self._load(confirmation_id)
        if record is None:
            raise AppException(
                code="TOOL_CONFIRMATION_NOT_FOUND",
                message="确认请求不存在或已失效。",
                status_code=404,
            )
        return record

    def _check_expired(self, record: ToolConfirmationRecord) -> None:
        if self._clock() >= record.expires_at:
            raise AppException(
                code="TOOL_CONFIRMATION_EXPIRED",
                message="确认请求已过期，请重新发起操作。",
                status_code=409,
            )

    def create(
        self,
        *,
        actor_id: str,
        tool_name: str,
        arguments: Mapping[str, Any],
        ttl_seconds: int,
    ) -> ToolConfirmationRecord:
        created_at = self._clock()
        record = ToolConfirmationRecord(
            confirmation_id=uuid4().hex,
            status=ToolConfirmationStatus.PENDING,
            actor_id=actor_id,
            tool_name=tool_name,
            arguments=deepcopy(dict(arguments)),
            arguments_fingerprint=build_arguments_fingerprint(tool_name, arguments),
            created_at=created_at,
            expires_at=created_at + timedelta(seconds=ttl_seconds),
        )
        self._store(record, ttl_seconds)
        return deepcopy(record)

    def confirm(
        self,
        confirmation_id: str,
        *,
        actor_id: str,
    ) -> ToolConfirmationRecord:
        record = self._require_record(confirmation_id)
        if record.actor_id != actor_id:
            raise AppException(
                code="TOOL_CONFIRMATION_FORBIDDEN",
                message="当前操作者不能确认其他人的工具请求。",
                status_code=403,
            )
        self._check_expired(record)
        if record.status == ToolConfirmationStatus.PENDING:
            record = ToolConfirmationRecord(
                confirmation_id=record.confirmation_id,
                status=ToolConfirmationStatus.CONFIRMED,
                actor_id=record.actor_id,
                tool_name=record.tool_name,
                arguments=record.arguments,
                arguments_fingerprint=record.arguments_fingerprint,
                created_at=record.created_at,
                expires_at=record.expires_at,
            )
            remaining = int((record.expires_at - self._clock()).total_seconds())
            self._store(record, ttl_seconds=max(1, remaining))
        return deepcopy(record)

    def require_confirmed(
        self,
        confirmation_id: str,
        *,
        actor_id: str,
    ) -> ToolConfirmationRecord:
        record = self._require_record(confirmation_id)
        if record.actor_id != actor_id:
            raise AppException(
                code="TOOL_CONFIRMATION_FORBIDDEN",
                message="当前操作者不能执行其他人的工具请求。",
                status_code=403,
            )
        self._check_expired(record)
        if record.status != ToolConfirmationStatus.CONFIRMED:
            raise AppException(
                code="TOOL_CONFIRMATION_REQUIRED",
                message="该工具请求尚未获得用户确认。",
                status_code=409,
            )
        return deepcopy(record)

    def register_confirmed(
        self,
        *,
        confirmation_id: str,
        actor_id: str,
        tool_name: str,
        arguments: Mapping[str, Any],
        ttl_seconds: int,
    ) -> ToolConfirmationRecord:
        created_at = self._clock()
        existing = self._load(confirmation_id)
        if existing is not None and existing.actor_id != actor_id:
            raise AppException(
                code="TOOL_CONFIRMATION_FORBIDDEN",
                message="当前操作者不能覆盖其他人的确认记录。",
                status_code=403,
            )
        record = ToolConfirmationRecord(
            confirmation_id=confirmation_id,
            status=ToolConfirmationStatus.CONFIRMED,
            actor_id=actor_id,
            tool_name=tool_name,
            arguments=deepcopy(dict(arguments)),
            arguments_fingerprint=build_arguments_fingerprint(tool_name, arguments),
            created_at=created_at,
            expires_at=created_at + timedelta(seconds=ttl_seconds),
        )
        self._store(record, ttl_seconds)
        return deepcopy(record)

    def clear(self) -> None:
        for key in self._redis.scan_iter(match=f"{self._key_prefix}:*"):
            self._redis.delete(key)

    def count(self) -> int:
        return sum(1 for _ in self._redis.scan_iter(match=f"{self._key_prefix}:*"))


def create_tool_confirmation_store(
    settings: Settings | None = None,
) -> ToolConfirmationStore | RedisToolConfirmationStore:
    from app.core.config import get_settings

    resolved_settings = settings or get_settings()
    if resolved_settings.resolved_tool_confirmation_backend == "redis":
        import redis as redis_lib

        redis_client = redis_lib.Redis.from_url(
            resolved_settings.resolved_agent_redis_url,
            decode_responses=True,
        )
        return RedisToolConfirmationStore(redis_client)
    return get_tool_confirmation_store()
