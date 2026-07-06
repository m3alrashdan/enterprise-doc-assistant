"""Dependency container: builds and owns all providers for the app lifetime.

Constructed once at startup (see app/main.py lifespan) from ``Settings``.
Providers are behind Protocol interfaces, so tests build a container with
fakes and production swaps implementations purely through configuration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import anyio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.config import Settings
from app.db.repositories import ConversationRepository, DocumentRepository
from app.db.session import create_engine_and_factory, init_db
from app.rag.chunking.base import ChunkingStrategy
from app.rag.chunking.factory import build_chunker
from app.rag.embeddings.base import EmbeddingProvider
from app.rag.embeddings.factory import build_embedding_provider
from app.rag.vectorstore.base import VectorStore
from app.rag.vectorstore.chroma import ChromaVectorStore
from app.services.documents import DocumentService
from app.services.ingestion import IngestionService

logger = logging.getLogger("app.container")

# (healthy, detail) per component name.
ReadinessReport = dict[str, tuple[bool, str]]


@dataclass
class Container:
    """Holds singletons shared across requests."""

    settings: Settings
    engine: AsyncEngine
    document_repo: DocumentRepository
    conversation_repo: ConversationRepository
    vector_store: VectorStore
    embedder: EmbeddingProvider
    chunker: ChunkingStrategy
    document_service: DocumentService
    ingestion_service: IngestionService

    async def check_readiness(self) -> ReadinessReport:
        """Probe critical dependencies for the readiness endpoint."""
        report: ReadinessReport = {}

        try:
            async with self.engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            report["database"] = (True, "connected")
        except Exception as exc:
            report["database"] = (False, str(exc))

        try:
            alive = await anyio.to_thread.run_sync(self.vector_store.heartbeat)
            report["vector_store"] = (alive, "connected" if alive else "unreachable")
        except Exception as exc:
            report["vector_store"] = (False, str(exc))

        return report

    async def shutdown(self) -> None:
        """Release resources on application shutdown."""
        await self.engine.dispose()


async def build_container(settings: Settings) -> Container:
    """Wire up all providers from configuration."""
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)

    engine, session_factory = create_engine_and_factory(settings.database_url)
    await init_db(engine)

    document_repo = DocumentRepository(session_factory)
    conversation_repo = ConversationRepository(session_factory)

    vector_store = ChromaVectorStore(settings)
    embedder = build_embedding_provider(settings)
    chunker = build_chunker(settings)

    document_service = DocumentService(settings, document_repo, vector_store)
    ingestion_service = IngestionService(settings, document_repo, vector_store, embedder, chunker)

    return Container(
        settings=settings,
        engine=engine,
        document_repo=document_repo,
        conversation_repo=conversation_repo,
        vector_store=vector_store,
        embedder=embedder,
        chunker=chunker,
        document_service=document_service,
        ingestion_service=ingestion_service,
    )
