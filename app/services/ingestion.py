"""Background ingestion: load -> chunk -> embed -> index.

Runs as a FastAPI background task after upload. Status transitions are
persisted at every step (pending -> processing -> ready | failed) so clients
can poll GET /documents/{id}.
"""

from __future__ import annotations

import logging
import time
from functools import partial
from pathlib import Path

import anyio

from app.core.config import Settings
from app.core.exceptions import IngestionError
from app.db.repositories import DocumentRepository
from app.models.document import DocumentStatus
from app.rag.chunking.base import ChunkingStrategy
from app.rag.embeddings.base import EmbeddingProvider
from app.rag.loaders.registry import get_loader
from app.rag.vectorstore.base import VectorStore

logger = logging.getLogger("app.ingestion")


class IngestionService:
    def __init__(
        self,
        settings: Settings,
        repository: DocumentRepository,
        vector_store: VectorStore,
        embedder: EmbeddingProvider,
        chunker: ChunkingStrategy,
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._vector_store = vector_store
        self._embedder = embedder
        self._chunker = chunker

    async def ingest_document(self, document_id: str) -> None:
        """Process one uploaded document end to end. Never raises; failures
        are recorded on the document record."""
        record = await self._repository.get(document_id)
        if record is None:
            logger.error(
                "ingestion requested for unknown document", extra={"document_id": document_id}
            )
            return

        await self._repository.update_status(document_id, DocumentStatus.PROCESSING)
        started = time.perf_counter()
        try:
            chunk_count, page_count = await self._process(document_id, record.stored_path, record)
            await self._repository.update_status(
                document_id,
                DocumentStatus.READY,
                chunk_count=chunk_count,
                page_count=page_count,
            )
            logger.info(
                "document ingested",
                extra={
                    "document_id": document_id,
                    "chunks": chunk_count,
                    "pages": page_count,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                },
            )
        except Exception as exc:
            logger.exception("ingestion failed", extra={"document_id": document_id})
            await self._repository.update_status(
                document_id, DocumentStatus.FAILED, error=str(exc)[:2000]
            )

    async def _process(self, document_id: str, stored_path: str, record) -> tuple[int, int | None]:
        loader = get_loader(record.extension)
        elements = await anyio.to_thread.run_sync(loader.load, Path(stored_path))

        # Chroma metadata must be scalar and non-None: include optional keys only when set.
        base_metadata: dict[str, object] = {
            "document_id": document_id,
            "document_name": record.filename,
            "uploaded_at": record.created_at.isoformat() if record.created_at else "",
        }
        if record.uploader:
            base_metadata["uploader"] = record.uploader
        if record.tag:
            base_metadata["tag"] = record.tag

        chunks = self._chunker.split(elements, base_metadata)
        if not chunks:
            raise IngestionError("Document produced no chunks after splitting.")

        texts = [chunk.content for chunk in chunks]
        embeddings: list[list[float]] = []
        batch_size = self._settings.embedding_batch_size
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            embeddings.extend(await anyio.to_thread.run_sync(self._embedder.embed_documents, batch))

        ids = [f"{document_id}:{index}" for index in range(len(chunks))]
        metadatas = [chunk.metadata for chunk in chunks]
        await anyio.to_thread.run_sync(
            partial(self._vector_store.add, ids, texts, metadatas, embeddings)
        )

        pages = [element.page for element in elements if element.page is not None]
        return len(chunks), (max(pages) if pages else None)
