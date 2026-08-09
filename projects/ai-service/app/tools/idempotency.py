from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import re
from threading import Lock
from typing import Any, TypeVar

from pydantic import BaseModel

from app.core.exceptions import AppException


IDEMPOTENCY_KEY_HEADER = "Idempotency-Key"

_IDEMPOTENCY_KEY_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{8,128}$")
_STORE_LOCK = Lock()
_IDEMPOTENCY_STORE: dict[str, "IdempotencyRecord"] = {}

ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class IdempotencyRecord:
    tool_name: str
    arguments_fingerprint: str
    result: Any


def validate_idempotency_key(idempotency_key: str | None) -> str | None:
    if idempotency_key is None:
        return None

    normalized_key = idempotency_key.strip()
    if not normalized_key:
        return None

    if not _IDEMPOTENCY_KEY_PATTERN.fullmatch(normalized_key):
        raise AppException(
            code="IDEMPOTENCY_KEY_INVALID",
            message="幂等键格式不正确，请使用 8 到 128 位的字母、数字、点、下划线、冒号或短横线。",
            status_code=422,
        )

    return normalized_key


def build_arguments_fingerprint(
    tool_name: str,
    arguments: BaseModel | Mapping[str, Any],
) -> str:
    if isinstance(arguments, BaseModel):
        payload = arguments.model_dump(mode="json")
    else:
        payload = dict(arguments)

    canonical_payload = json.dumps(
        {
            "tool_name": tool_name,
            "arguments": payload,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()


def run_idempotent_tool(
    tool_name: str,
    arguments: BaseModel | Mapping[str, Any],
    idempotency_key: str | None,
    executor: Callable[[], ResultT],
) -> ResultT:
    normalized_key = validate_idempotency_key(idempotency_key)
    if normalized_key is None:
        return executor()

    arguments_fingerprint = build_arguments_fingerprint(tool_name, arguments)

    with _STORE_LOCK:
        existing_record = _IDEMPOTENCY_STORE.get(normalized_key)
        if existing_record is not None:
            if existing_record.arguments_fingerprint != arguments_fingerprint:
                raise AppException(
                    code="IDEMPOTENCY_KEY_CONFLICT",
                    message="同一个幂等键不能用于不同的工具调用参数。",
                    status_code=409,
                )

            return deepcopy(existing_record.result)

        result = executor()
        _IDEMPOTENCY_STORE[normalized_key] = IdempotencyRecord(
            tool_name=tool_name,
            arguments_fingerprint=arguments_fingerprint,
            result=deepcopy(result),
        )
        return result


def clear_idempotency_store() -> None:
    with _STORE_LOCK:
        _IDEMPOTENCY_STORE.clear()


def get_idempotency_record_count() -> int:
    with _STORE_LOCK:
        return len(_IDEMPOTENCY_STORE)
