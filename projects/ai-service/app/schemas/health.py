from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


DependencyProbeStatus = Literal["ok", "configured", "skipped", "not_configured"]

RuntimeDependencyStatus = Literal["ok", "unreachable", "not_configured"]


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"] = Field(description="Liveness status.")
    service: str = Field(description="Service name.")
    time: str = Field(description="Current UTC time in ISO 8601 format.")


class ReadinessCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Readiness check name.")
    status: DependencyProbeStatus = Field(description="Check result.")
    required: bool = Field(description="Whether this check blocks readiness.")
    message: str = Field(description="Human-readable check result.")


class ReadinessResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ready", "not_ready"] = Field(description="Readiness status.")
    service: str = Field(description="Service name.")
    ready: bool = Field(description="Whether the service can accept traffic.")
    checks: list[ReadinessCheck] = Field(description="Readiness checks.")
    time: str = Field(description="Current UTC time in ISO 8601 format.")


class DependencyProbe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Dependency name.")
    status: RuntimeDependencyStatus = Field(description="Probe result.")
    message: str = Field(description="Human-readable probe result.")
    latency_ms: float | None = Field(description="Probe latency in milliseconds.")


class DependencyProbeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "degraded"] = Field(description="Overall dependency status.")
    service: str = Field(description="Service name.")
    dependencies: list[DependencyProbe] = Field(description="Runtime dependency probes.")
    time: str = Field(description="Current UTC time in ISO 8601 format.")
