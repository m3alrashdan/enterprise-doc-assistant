"""Health endpoint and cross-cutting API behaviour."""

from __future__ import annotations

import httpx


async def test_liveness(anon_client: httpx.AsyncClient) -> None:
    resp = await anon_client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


async def test_readiness_reports_components(anon_client: httpx.AsyncClient) -> None:
    resp = await anon_client.get("/health/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready"
    assert isinstance(body["components"], dict)


async def test_request_id_echoed(anon_client: httpx.AsyncClient) -> None:
    resp = await anon_client.get("/health", headers={"X-Request-ID": "trace-me-123"})
    assert resp.headers["x-request-id"] == "trace-me-123"


async def test_request_id_generated(anon_client: httpx.AsyncClient) -> None:
    resp = await anon_client.get("/health")
    assert len(resp.headers["x-request-id"]) >= 8


async def test_unknown_route_uses_error_envelope(anon_client: httpx.AsyncClient) -> None:
    resp = await anon_client.get("/does-not-exist")
    assert resp.status_code == 404
    body = resp.json()
    assert set(body["error"].keys()) == {"code", "message", "details"}
