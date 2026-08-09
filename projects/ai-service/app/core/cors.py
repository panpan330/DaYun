from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


def register_cors_middleware(
    app: FastAPI,
    allowed_origins: list[str],
    allowed_origin_regex: str | None = None,
) -> None:
    if not allowed_origins and not allowed_origin_regex:
        return

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_origin_regex=allowed_origin_regex,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
