"""Shared fixtures.

All tests run fully offline: fake embedding and LLM providers, a temp SQLite
database and a temp ChromaDB directory. No network, no API keys.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest
from fastapi import FastAPI

from app.core.config import Settings
from app.main import create_app

TEST_API_KEY = "test-key"


def make_test_settings(tmp_path, **overrides) -> Settings:
    """Fully offline settings: fake providers, temp storage."""
    defaults: dict = {
        "environment": "test",
        "log_json": False,
        "log_level": "WARNING",
        "data_dir": tmp_path / "data",
        "database_url": f"sqlite+aiosqlite:///{tmp_path}/test.db",
        "chroma_mode": "embedded",
        "chroma_persist_dir": tmp_path / "chroma",
        "embedding_provider": "fake",
        "llm_provider": "fake",
        "api_keys": [TEST_API_KEY],
        "rate_limit_enabled": False,
        "chunk_size": 200,
        "chunk_overlap": 40,
        "similarity_threshold": -1.0,  # fake embeddings produce arbitrary similarities
    }
    defaults.update(overrides)
    # _env_file=None: never read the developer's .env in tests
    return Settings(_env_file=None, **defaults)


@pytest.fixture
def test_settings(tmp_path) -> Settings:
    return make_test_settings(tmp_path)


@pytest.fixture
async def test_app(test_settings: Settings) -> AsyncIterator[FastAPI]:
    app = create_app(test_settings)
    async with app.router.lifespan_context(app):
        yield app


@pytest.fixture
async def client(test_app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    """Authenticated API client."""
    transport = httpx.ASGITransport(app=test_app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
        headers={"X-API-Key": TEST_API_KEY},
    ) as client:
        yield client


@pytest.fixture
async def anon_client(test_app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    """Client without credentials, for auth-failure tests."""
    transport = httpx.ASGITransport(app=test_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client
