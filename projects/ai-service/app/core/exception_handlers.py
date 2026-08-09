import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.exceptions import AppException
from app.core.trace import TRACE_ID_HEADER, get_trace_id, reset_trace_id, set_trace_id
from app.schemas.error import ErrorResponse


logger = logging.getLogger(__name__)


def build_error_response(
    status_code: int,
    code: str,
    message: str,
    details: list[dict[str, Any]] | None = None,
    headers: dict[str, str] | None = None,
    trace_id: str | None = None,
) -> JSONResponse:
    trace_id = trace_id or get_trace_id()
    body = ErrorResponse(
        code=code,
        message=message,
        trace_id=trace_id,
        details=details,
    ).model_dump(exclude_none=True)
    response_headers = dict(headers or {})
    response_headers[TRACE_ID_HEADER] = trace_id
    return JSONResponse(
        status_code=status_code,
        content=jsonable_encoder(body),
        headers=response_headers,
    )


def get_request_trace_id(request: Request) -> str:
    return getattr(request.state, "trace_id", None) or get_trace_id()


def get_http_error_code(status_code: int) -> str:
    if status_code == 404:
        return "NOT_FOUND"
    if status_code == 405:
        return "METHOD_NOT_ALLOWED"
    return "HTTP_ERROR"


def get_http_error_message(exc: StarletteHTTPException) -> str:
    if exc.status_code == 404:
        return "资源不存在"
    if exc.status_code == 405:
        return "请求方法不允许"
    if isinstance(exc.detail, str):
        return exc.detail
    return "请求处理失败"


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppException)
    async def app_exception_handler(
        request: Request,
        exc: AppException,
    ) -> JSONResponse:
        trace_id = get_request_trace_id(request)
        token = set_trace_id(trace_id)
        try:
            logger.warning(
                "app_exception code=%s method=%s path=%s",
                exc.code,
                request.method,
                request.url.path,
            )
            return build_error_response(
                status_code=exc.status_code,
                code=exc.code,
                message=exc.message,
                details=exc.details,
                trace_id=trace_id,
            )
        finally:
            reset_trace_id(token)

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request,
        exc: StarletteHTTPException,
    ) -> JSONResponse:
        trace_id = get_request_trace_id(request)
        token = set_trace_id(trace_id)
        try:
            logger.warning(
                "http_exception status_code=%s method=%s path=%s",
                exc.status_code,
                request.method,
                request.url.path,
            )
            return build_error_response(
                status_code=exc.status_code,
                code=get_http_error_code(exc.status_code),
                message=get_http_error_message(exc),
                headers=exc.headers,
                trace_id=trace_id,
            )
        finally:
            reset_trace_id(token)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        trace_id = get_request_trace_id(request)
        token = set_trace_id(trace_id)
        try:
            logger.warning(
                "validation_exception method=%s path=%s errors=%s",
                request.method,
                request.url.path,
                len(exc.errors()),
            )
            return build_error_response(
                status_code=422,
                code="VALIDATION_ERROR",
                message="请求参数校验失败",
                details=jsonable_encoder(exc.errors()),
                trace_id=trace_id,
            )
        finally:
            reset_trace_id(token)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        trace_id = get_request_trace_id(request)
        token = set_trace_id(trace_id)
        try:
            logger.exception(
                "unhandled_exception method=%s path=%s",
                request.method,
                request.url.path,
            )
            return build_error_response(
                status_code=500,
                code="INTERNAL_SERVER_ERROR",
                message="服务器内部错误",
                trace_id=trace_id,
            )
        finally:
            reset_trace_id(token)
