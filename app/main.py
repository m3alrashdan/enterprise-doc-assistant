"""Application entrypoint and factory."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI

from app.api.errors import register_exception_handlers
from app.api.health import router as health_router
from app.api.middleware import RequestContextMiddleware
from app.api.ratelimit import RateLimitMiddleware
from app.api.v1.router import api_v1_router
from app.api.v1.schemas import ErrorEnvelope
from app.core.config import Settings, load_settings
from app.core.container import build_container
from app.core.logging import configure_logging
from app.core.security import require_api_key

logger = logging.getLogger("app.main")

_DESCRIPTION = """
Enterprise question answering over internal documents using
Retrieval-Augmented Generation (RAG).

Upload PDF / DOCX / TXT / MD / HTML documents, then ask natural-language
questions. Answers are generated strictly from document content and carry
inline citations back to the source document and page.

Authenticate by sending your API key in the `X-API-Key` header.
"""


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build a fully wired application instance.

    Tests call this with their own ``Settings`` (fake providers, temp dirs);
    production uses the environment-derived settings.
    """
    settings = settings or load_settings()
    configure_logging(settings.log_level, settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        settings.upload_dir.mkdir(parents=True, exist_ok=True)
        if not settings.auth_enabled:
            logger.warning(
                "authentication is DISABLED (API_KEYS is empty) - do not run in production"
            )
        app.state.container = await build_container(settings)
        logger.info(
            "application started",
            extra={
                "environment": settings.environment,
                "llm_provider": settings.llm_provider,
                "embedding_provider": settings.embedding_provider,
            },
        )
        yield
        await app.state.container.shutdown()
        logger.info("application stopped")

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=_DESCRIPTION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.state.settings = settings

    # Middleware order: last added runs first, so requests pass through
    # RequestContext (IDs + timing) before hitting the rate limiter.
    if settings.rate_limit_enabled:
        app.add_middleware(RateLimitMiddleware, rate=settings.rate_limit)
    app.add_middleware(RequestContextMiddleware)
    register_exception_handlers(app)

    app.include_router(health_router)
    app.include_router(
        api_v1_router,
        prefix=settings.api_v1_prefix,
        dependencies=[Depends(require_api_key)],
        responses={
            401: {"model": ErrorEnvelope, "description": "Missing or invalid API key"},
            429: {"model": ErrorEnvelope, "description": "Rate limit exceeded"},
        },
    )
    return app


app = create_app()
