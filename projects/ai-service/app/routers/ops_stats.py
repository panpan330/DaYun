"""Ops stats endpoints (proxy to Java internal aggregation)."""

from typing import Any

from fastapi import APIRouter, Depends, Query

from app.core.config import Settings, get_settings
from app.core.exceptions import AppException
from app.services.java_ops_stats_client import JavaOpsStatsClient, JavaOpsStatsClientError

router = APIRouter(tags=["ops-stats"])


def get_ops_stats_client(settings: Settings = Depends(get_settings)) -> JavaOpsStatsClient:
    return JavaOpsStatsClient.from_settings(settings)


@router.get("/api/ai/ops-stats/summary")
def ops_stats_summary(
    days: int = Query(default=7),
    client: JavaOpsStatsClient = Depends(get_ops_stats_client),
) -> dict[str, Any]:
    if days not in (7, 30):
        raise AppException(code="INVALID_DAYS", message="days 仅支持 7 或 30。", status_code=400)
    try:
        return client.get_summary(days)
    except JavaOpsStatsClientError as exc:
        raise AppException(
            code=exc.code,
            message="运营统计数据暂不可用。",
            status_code=502,
        ) from exc
