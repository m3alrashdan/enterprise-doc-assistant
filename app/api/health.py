"""Liveness and readiness endpoints (unauthenticated, outside /api/v1)."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel

router = APIRouter(tags=["health"])


class ComponentHealth(BaseModel):
    healthy: bool
    detail: str = ""


class ReadinessResponse(BaseModel):
    status: str
    components: dict[str, ComponentHealth]


@router.get("/health", summary="Liveness probe")
async def health() -> dict[str, str]:
    """Process is up and able to serve requests."""
    return {"status": "ok"}


@router.get(
    "/health/ready",
    summary="Readiness probe",
    response_model=ReadinessResponse,
    responses={503: {"description": "One or more dependencies are unavailable"}},
)
async def health_ready(request: Request, response: Response) -> ReadinessResponse:
    """Check dependencies (vector store, LLM provider, metadata DB)."""
    container = getattr(request.app.state, "container", None)
    components: dict[str, ComponentHealth] = {}
    if container is not None:
        for name, (healthy, detail) in (await container.check_readiness()).items():
            components[name] = ComponentHealth(healthy=healthy, detail=detail)
    all_healthy = all(c.healthy for c in components.values())
    if not all_healthy:
        response.status_code = 503
    return ReadinessResponse(status="ready" if all_healthy else "degraded", components=components)
