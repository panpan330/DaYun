from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Response, status

from app.core.config import Settings, get_settings
from app.core.config_safety import (
    SecretConfigurationCheck,
    build_secret_configuration_checks,
)
from app.schemas.health import (
    DependencyProbe,
    DependencyProbeResponse,
    HealthResponse,
    ReadinessCheck,
    ReadinessResponse,
)
from app.services.dependency_probe import build_dependency_probes


router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="ai-service",
        time=_utc_now(),
    )


@router.get("/health/dependencies", response_model=DependencyProbeResponse)
def dependency_check(
    settings: Settings = Depends(get_settings),
) -> DependencyProbeResponse:
    probes = [DependencyProbe(**probe) for probe in build_dependency_probes(settings)]
    degraded = any(probe.status != "ok" for probe in probes)
    return DependencyProbeResponse(
        status="degraded" if degraded else "ok",
        service="ai-service",
        dependencies=probes,
        time=_utc_now(),
    )


@router.get("/ready", response_model=ReadinessResponse)
def readiness_check(
    response: Response,
    settings: Settings = Depends(get_settings),
) -> ReadinessResponse:
    checks = build_ai_service_readiness_checks(settings)
    ready = all(check.status != "not_configured" for check in checks if check.required)
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(
        status="ready" if ready else "not_ready",
        service="ai-service",
        ready=ready,
        checks=checks,
        time=_utc_now(),
    )


def build_ai_service_readiness_checks(settings: Settings) -> list[ReadinessCheck]:
    checks = [
        ReadinessCheck(
            name="java_business_service_base_url",
            status="configured"
            if settings.resolved_java_business_service_base_url
            else "not_configured",
            required=True,
            message="Java business service base URL is configured.",
        ),
        ReadinessCheck(
            name="ticket_agent_model_mode",
            status="ok",
            required=True,
            message=f"Ticket Agent model mode is {settings.ticket_agent_model_mode}.",
        ),
        ReadinessCheck(
            name="qdrant_base_url",
            status="configured",
            required=False,
            message="Qdrant base URL is configured for RAG workflows.",
        ),
        ReadinessCheck(
            name="milvus_uri",
            status="configured",
            required=False,
            message="Milvus URI is configured for optional Milvus workflows.",
        ),
    ]
    llm_secret_check = _get_secret_check("llm_api_key", settings)
    checks.append(
        ReadinessCheck(
            name=llm_secret_check.name,
            status=llm_secret_check.readiness_status,
            required=llm_secret_check.required,
            message=llm_secret_check.message,
        )
    )
    return checks


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_secret_check(name: str, settings: Settings) -> SecretConfigurationCheck:
    for check in build_secret_configuration_checks(settings):
        if check.name == name:
            return check
    raise RuntimeError(f"missing secret configuration check: {name}")
