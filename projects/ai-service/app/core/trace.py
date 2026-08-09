from contextvars import ContextVar, Token
from uuid import uuid4


TRACE_ID_HEADER = "X-Trace-Id"
DEFAULT_TRACE_ID = "-"

_trace_id: ContextVar[str] = ContextVar("trace_id", default=DEFAULT_TRACE_ID)


def generate_trace_id() -> str:
    return uuid4().hex


def get_or_create_trace_id(incoming_trace_id: str | None) -> str:
    if incoming_trace_id and incoming_trace_id.strip():
        return incoming_trace_id.strip()
    return generate_trace_id()


def get_trace_id() -> str:
    return _trace_id.get()


def build_trace_headers() -> dict[str, str]:
    trace_id = get_trace_id()
    if trace_id == DEFAULT_TRACE_ID:
        return {}
    return {TRACE_ID_HEADER: trace_id}


def set_trace_id(trace_id: str) -> Token[str]:
    return _trace_id.set(trace_id)


def reset_trace_id(token: Token[str]) -> None:
    _trace_id.reset(token)
