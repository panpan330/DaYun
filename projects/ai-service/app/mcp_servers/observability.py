"""Observability helpers for MCP tool and resource calls."""

from collections.abc import Callable
from functools import wraps
import logging
from time import perf_counter
from typing import ParamSpec, TypeVar

from app.core.trace import get_trace_id


P = ParamSpec("P")
R = TypeVar("R")

logger = logging.getLogger("app.mcp")


def _elapsed_ms(start_time: float) -> float:
    return round((perf_counter() - start_time) * 1000, 2)


def _classify_tool_result(result: object) -> tuple[str, str | None]:
    if isinstance(result, dict) and result.get("ok") is False:
        error_code = result.get("error_code")
        return "business_error", str(error_code) if error_code else None
    return "succeeded", None


def observe_mcp_tool(
    *,
    tool_name: str,
    action_type: str,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Log a safe lifecycle record around an MCP tool function."""

    def decorator(function: Callable[P, R]) -> Callable[P, R]:
        @wraps(function)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start_time = perf_counter()
            trace_id = get_trace_id()
            logger.info(
                (
                    "mcp_tool_call_started trace_id=%s tool_name=%s "
                    "action_type=%s"
                ),
                trace_id,
                tool_name,
                action_type,
            )
            try:
                result = function(*args, **kwargs)
            except Exception as exc:
                logger.warning(
                    (
                        "mcp_tool_call_failed trace_id=%s tool_name=%s "
                        "action_type=%s status=system_error error_type=%s "
                        "elapsed_ms=%.2f"
                    ),
                    trace_id,
                    tool_name,
                    action_type,
                    exc.__class__.__name__,
                    _elapsed_ms(start_time),
                )
                raise

            status, error_code = _classify_tool_result(result)
            logger.info(
                (
                    "mcp_tool_call_finished trace_id=%s tool_name=%s "
                    "action_type=%s status=%s error_code=%s elapsed_ms=%.2f"
                ),
                trace_id,
                tool_name,
                action_type,
                status,
                error_code or "-",
                _elapsed_ms(start_time),
            )
            return result

        return wrapper

    return decorator


def observe_mcp_resource(
    *,
    resource_uri: str,
    mime_type: str,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Log a safe lifecycle record around an MCP resource function."""

    def decorator(function: Callable[P, R]) -> Callable[P, R]:
        @wraps(function)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start_time = perf_counter()
            trace_id = get_trace_id()
            logger.info(
                (
                    "mcp_resource_read_started trace_id=%s resource_uri=%s "
                    "mime_type=%s"
                ),
                trace_id,
                resource_uri,
                mime_type,
            )
            try:
                result = function(*args, **kwargs)
            except Exception as exc:
                logger.warning(
                    (
                        "mcp_resource_read_failed trace_id=%s resource_uri=%s "
                        "mime_type=%s status=system_error error_type=%s "
                        "elapsed_ms=%.2f"
                    ),
                    trace_id,
                    resource_uri,
                    mime_type,
                    exc.__class__.__name__,
                    _elapsed_ms(start_time),
                )
                raise

            logger.info(
                (
                    "mcp_resource_read_finished trace_id=%s resource_uri=%s "
                    "mime_type=%s status=succeeded elapsed_ms=%.2f"
                ),
                trace_id,
                resource_uri,
                mime_type,
                _elapsed_ms(start_time),
            )
            return result

        return wrapper

    return decorator
