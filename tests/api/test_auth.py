"""Authentication behaviour: API key enforcement on /api/v1, open health."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest

from app.main import create_app
from tests.conftest import make_test_settings


async def test_missing_api_key_is_rejected(anon_client: httpx.AsyncClient) -> None:
    resp = await anon_client.get("/api/v1/documents")
    assert resp.status_code == 401
    body = resp.json()
    assert body["error"]["code"] == "unauthorized"
    assert "X-API-Key" in body["error"]["message"]


async def test_wrong_api_key_is_rejected(anon_client: httpx.AsyncClient) -> None:
    resp = await anon_client.get("/api/v1/documents", headers={"X-API-Key": "wrong-key"})
    assert resp.status_code == 401


async def test_valid_api_key_is_accepted(client: httpx.AsyncClient) -> None:
    resp = await client.get("/api/v1/documents")
    assert resp.status_code == 200


async def test_chat_requires_auth_too(anon_client: httpx.AsyncClient) -> None:
    resp = await anon_client.post("/api/v1/chat/query", json={"question": "hello?"})
    assert resp.status_code == 401


async def test_health_endpoints_do_not_require_auth(anon_client: httpx.AsyncClient) -> None:
    assert (await anon_client.get("/health")).status_code == 200
    assert (await anon_client.get("/health/ready")).status_code == 200


@pytest.fixture
async def no_auth_client(tmp_path) -> AsyncIterator[httpx.AsyncClient]:
    """App configured with an empty API_KEYS list (auth disabled, dev mode)."""
    app = create_app(make_test_settings(tmp_path, api_keys=[]))
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield client


async def test_empty_api_keys_disables_auth(no_auth_client: httpx.AsyncClient) -> None:
    resp = await no_auth_client.get("/api/v1/documents")
    assert resp.status_code == 200
