from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.exceptions import AppException


TICKET_AGENT_CHECKPOINT_SNAPSHOT_SCHEMA_VERSION = "ticket-agent-checkpoint-snapshot:v1"
TICKET_AGENT_CHECKPOINT_THREAD_ID_INVALID_CODE = (
    "TICKET_AGENT_CHECKPOINT_THREAD_ID_INVALID"
)
TICKET_AGENT_CHECKPOINT_VALUES_INVALID_CODE = (
    "TICKET_AGENT_CHECKPOINT_VALUES_INVALID"
)
TICKET_AGENT_CHECKPOINT_FILE_INVALID_CODE = "TICKET_AGENT_CHECKPOINT_FILE_INVALID"
TICKET_AGENT_CHECKPOINT_WRITE_FAILED_CODE = "TICKET_AGENT_CHECKPOINT_WRITE_FAILED"

_CHECKPOINT_FILENAME_UNSAFE_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


def normalize_checkpoint_thread_id(thread_id: str) -> str:
    normalized_thread_id = thread_id.strip()
    if not normalized_thread_id:
        raise AppException(
            code=TICKET_AGENT_CHECKPOINT_THREAD_ID_INVALID_CODE,
            message="thread_id 不能为空。",
            status_code=400,
        )
    return normalized_thread_id


def build_checkpoint_snapshot_filename(thread_id: str) -> str:
    normalized_thread_id = normalize_checkpoint_thread_id(thread_id)
    readable_part = _CHECKPOINT_FILENAME_UNSAFE_PATTERN.sub(
        "_",
        normalized_thread_id,
    ).strip("._")
    if not readable_part:
        readable_part = "thread"

    digest = hashlib.sha256(normalized_thread_id.encode("utf-8")).hexdigest()[:12]
    return f"{readable_part[:80]}-{digest}.json"


@dataclass(frozen=True)
class TicketAgentCheckpointSnapshot:
    thread_id: str
    values: dict[str, Any]
    saved_at: str
    schema_version: str = TICKET_AGENT_CHECKPOINT_SNAPSHOT_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        thread_id: str,
        values: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        saved_at: str | None = None,
    ) -> "TicketAgentCheckpointSnapshot":
        if not isinstance(values, dict):
            raise AppException(
                code=TICKET_AGENT_CHECKPOINT_VALUES_INVALID_CODE,
                message="checkpoint values 必须是字典。",
                status_code=500,
            )

        return cls(
            thread_id=normalize_checkpoint_thread_id(thread_id),
            values=values,
            saved_at=saved_at or datetime.now(timezone.utc).isoformat(),
            metadata=metadata or {},
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "thread_id": self.thread_id,
            "saved_at": self.saved_at,
            "metadata": self.metadata,
            "values": self.values,
        }

    @classmethod
    def from_json_dict(
        cls,
        data: dict[str, Any],
    ) -> "TicketAgentCheckpointSnapshot":
        if data.get("schema_version") != TICKET_AGENT_CHECKPOINT_SNAPSHOT_SCHEMA_VERSION:
            raise AppException(
                code=TICKET_AGENT_CHECKPOINT_FILE_INVALID_CODE,
                message="checkpoint 文件版本不受支持。",
                status_code=500,
            )

        values = data.get("values")
        if not isinstance(values, dict):
            raise AppException(
                code=TICKET_AGENT_CHECKPOINT_VALUES_INVALID_CODE,
                message="checkpoint values 必须是字典。",
                status_code=500,
            )

        metadata = data.get("metadata")
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise AppException(
                code=TICKET_AGENT_CHECKPOINT_FILE_INVALID_CODE,
                message="checkpoint metadata 必须是字典。",
                status_code=500,
            )

        saved_at = data.get("saved_at")
        if not isinstance(saved_at, str) or not saved_at.strip():
            raise AppException(
                code=TICKET_AGENT_CHECKPOINT_FILE_INVALID_CODE,
                message="checkpoint saved_at 必须是非空字符串。",
                status_code=500,
            )

        thread_id = data.get("thread_id")
        if not isinstance(thread_id, str):
            raise AppException(
                code=TICKET_AGENT_CHECKPOINT_THREAD_ID_INVALID_CODE,
                message="thread_id 不能为空。",
                status_code=500,
            )

        return cls.create(
            thread_id=thread_id,
            values=values,
            metadata=metadata,
            saved_at=saved_at,
        )


class FileTicketAgentCheckpointStore:
    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)

    def build_path(self, thread_id: str) -> Path:
        return self.root_dir / build_checkpoint_snapshot_filename(thread_id)

    def save(self, snapshot: TicketAgentCheckpointSnapshot) -> Path:
        path = self.build_path(snapshot.thread_id)
        payload = snapshot.to_json_dict()
        try:
            text = json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            self.root_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        except TypeError as exc:
            raise AppException(
                code=TICKET_AGENT_CHECKPOINT_VALUES_INVALID_CODE,
                message="checkpoint values 包含不能写入 JSON 的对象。",
                status_code=500,
            ) from exc
        except OSError as exc:
            raise AppException(
                code=TICKET_AGENT_CHECKPOINT_WRITE_FAILED_CODE,
                message="checkpoint 文件写入失败。",
                status_code=500,
            ) from exc
        return path

    def load(self, thread_id: str) -> TicketAgentCheckpointSnapshot | None:
        expected_thread_id = normalize_checkpoint_thread_id(thread_id)
        path = self.build_path(thread_id)
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise AppException(
                code=TICKET_AGENT_CHECKPOINT_FILE_INVALID_CODE,
                message="checkpoint 文件无法读取或不是合法 JSON。",
                status_code=500,
            ) from exc

        if not isinstance(data, dict):
            raise AppException(
                code=TICKET_AGENT_CHECKPOINT_FILE_INVALID_CODE,
                message="checkpoint 文件顶层必须是 JSON 对象。",
                status_code=500,
            )
        snapshot = TicketAgentCheckpointSnapshot.from_json_dict(data)
        if snapshot.thread_id != expected_thread_id:
            raise AppException(
                code=TICKET_AGENT_CHECKPOINT_FILE_INVALID_CODE,
                message="checkpoint 文件 thread_id 与请求不一致。",
                status_code=500,
            )
        return snapshot
