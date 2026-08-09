from app.core.config import Settings
from app.core.exceptions import AppException
from app.schemas.tool_confirmation import (
    ToolConfirmationRequest,
    ToolConfirmationResponse,
    ToolConfirmationStatus,
)
from app.tools.tool_confirmation import ToolConfirmationRecord, ToolConfirmationStore
from app.tools.tool_registry import require_enabled_tool_definition


class ToolConfirmationService:
    def __init__(self, settings: Settings, store: ToolConfirmationStore) -> None:
        self.settings = settings
        self.store = store

    def _to_response(
        self,
        record: ToolConfirmationRecord,
    ) -> ToolConfirmationResponse:
        if record.status == ToolConfirmationStatus.PENDING:
            message = "请核对工具和参数后再确认；当前不会执行工具。"
        else:
            message = "确认已记录；可通过专用执行接口执行这份固定计划。"

        return ToolConfirmationResponse(
            confirmation_id=record.confirmation_id,
            status=record.status,
            actor_id=record.actor_id,
            tool_name=record.tool_name,
            arguments=record.arguments,
            arguments_fingerprint=record.arguments_fingerprint,
            created_at=record.created_at,
            expires_at=record.expires_at,
            message=message,
        )

    def request_confirmation(
        self,
        request: ToolConfirmationRequest,
    ) -> ToolConfirmationResponse:
        definition = require_enabled_tool_definition(request.tool_name)
        if not definition.requires_confirmation:
            raise AppException(
                code="TOOL_CONFIRMATION_NOT_REQUIRED",
                message="该工具不需要用户确认。",
                status_code=409,
            )

        record = self.store.create(
            actor_id=request.actor_id,
            tool_name=definition.name,
            arguments=request.arguments,
            ttl_seconds=self.settings.tool_confirmation_ttl_seconds,
        )
        return self._to_response(record)

    def confirm(
        self,
        confirmation_id: str,
        *,
        actor_id: str,
    ) -> ToolConfirmationResponse:
        return self._to_response(
            self.store.confirm(confirmation_id, actor_id=actor_id)
        )

    def require_confirmed(
        self,
        confirmation_id: str,
        *,
        actor_id: str,
    ) -> ToolConfirmationRecord:
        return self.store.require_confirmed(
            confirmation_id,
            actor_id=actor_id,
        )
