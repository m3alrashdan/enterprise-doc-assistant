"""Application entrypoint and factory."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.errors import register_exception_handlers
from app.api.health import router as health_router
from app.api.middleware import RequestContextMiddleware
from app.api.v1.router import api_v1_router
from app.core.config import Settings, load_settings
from app.core.container import build_container
from app.core.logging import configure_logging

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

    app.add_middleware(RequestContextMiddleware)
    register_exception_handlers(app)

    app.include_router(health_router)
    app.include_router(api_v1_router, prefix=settings.api_v1_prefix)
    return app


app = create_app()
