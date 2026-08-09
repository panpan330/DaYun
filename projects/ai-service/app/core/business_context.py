from contextvars import ContextVar, Token

from app.core.config import Settings


_current_user_id: ContextVar[str | None] = ContextVar("current_user_id", default=None)
_current_tenant_id: ContextVar[str | None] = ContextVar("current_tenant_id", default=None)


def set_business_context(
    *,
    user_id: str | None,
    tenant_id: str | None,
) -> tuple[Token[str | None], Token[str | None]]:
    return (
        _current_user_id.set(_normalize_optional_header(user_id)),
        _current_tenant_id.set(_normalize_optional_header(tenant_id)),
    )


def reset_business_context(
    tokens: tuple[Token[str | None], Token[str | None]],
) -> None:
    user_token, tenant_token = tokens
    _current_user_id.reset(user_token)
    _current_tenant_id.reset(tenant_token)


def get_business_context() -> tuple[str | None, str | None]:
    """Return the currently set (user_id, tenant_id), or (None, None).

    Used by MCP tool adapters running in the AI-service process to forward
    the authenticated actor identity to the standalone product MCP server,
    which has no contextvar of its own.
    """
    return _current_user_id.get(), _current_tenant_id.get()


def build_java_internal_headers(settings: Settings) -> dict[str, str]:
    return {
        "X-Caller": settings.java_business_internal_caller.strip() or "ai-service",
        "X-User-Id": _current_user_id.get() or settings.java_business_default_user_id,
        "X-Tenant-Id": _current_tenant_id.get() or settings.java_business_default_tenant_id,
        "X-Internal-Token": settings.java_business_internal_token,
    }


def _normalize_optional_header(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None
