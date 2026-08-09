"""Cost overview API: LLM cost aggregation (model x intent) from Java (MySQL)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings
from app.core.exceptions import AppException
from app.services.java_cost_client import JavaCostClient, JavaCostClientError

router = APIRouter(prefix="/api/ai/cost", tags=["cost"])


@router.get("/overview")
def cost_overview(
    settings: Settings = Depends(get_settings),
) -> dict:
    client = JavaCostClient.from_settings(settings)
    try:
        return client.fetch_overview()
    except JavaCostClientError as exc:
        raise AppException(
            code=exc.code,
            message=exc.message,
            status_code=exc.status_code or 502,
        ) from exc
