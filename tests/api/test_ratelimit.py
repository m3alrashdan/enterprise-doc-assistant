"""Rate limiting behaviour."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest

from app.api.ratelimit import parse_rate
from app.main import create_app
from tests.conftest import TEST_API_KEY, make_test_settings


class TestParseRate:
    def test_minute(self) -> None:
        assert parse_rate("60/minute") == (60, 60)

    def test_second(self) -> None:
        assert parse_rate("5/second") == (5, 1)

    def test_hour(self) -> None:
        assert parse_rate("100/hour") == (100, 3600)

    def test_invalid(self) -> None:
        with pytest.raises(ValueError, match="Invalid rate limit"):
            parse_rate("lots/day")


@pytest.fixture
async def limited_client(tmp_path) -> AsyncIterator[httpx.AsyncClient]:
    settings = make_test_settings(tmp_path, rate_limit_enabled=True, rate_limit="3/hour")
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            headers={"X-API-Key": TEST_API_KEY},
        ) as client:
            yield client


async def test_requests_over_limit_get_429(limited_client: httpx.AsyncClient) -> None:
    for _ in range(3):
        assert (await limited_client.get("/api/v1/documents")).status_code == 200

    resp = await limited_client.get("/api/v1/documents")
    assert resp.status_code == 429
    body = resp.json()
    assert body["error"]["code"] == "rate_limited"
    assert int(resp.headers["Retry-After"]) > 0


async def test_buckets_are_per_api_key(limited_client: httpx.AsyncClient) -> None:
    for _ in range(4):
        await limited_client.get("/api/v1/documents")
    # a different caller identity is not affected (auth still fails, but not 429)
    resp = await limited_client.get("/api/v1/documents", headers={"X-API-Key": "other-key"})
    assert resp.status_code == 401


async def test_health_is_exempt_from_rate_limiting(limited_client: httpx.AsyncClient) -> None:
    for _ in range(10):
        assert (await limited_client.get("/health")).status_code == 200
