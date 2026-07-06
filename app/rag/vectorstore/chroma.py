"""ChromaDB vector store.

Supports two modes selected by configuration:
- ``embedded``: persistent local storage (single-process, simplest ops)
- ``http``: a separate ChromaDB server (docker-compose, horizontal scaling)

The collection uses cosine space; Chroma returns distances, converted here to
similarities (``score = 1 - distance``) so callers never see distances.
"""

from __future__ import annotations

import logging
from typing import Any

import chromadb
from chromadb.api import ClientAPI

from app.core.config import Settings
from app.core.exceptions import ProviderError
from app.rag.vectorstore.base import StoredChunk, VectorSearchResult

logger = logging.getLogger("app.vectorstore")


class ChromaVectorStore:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: ClientAPI | None = None

    def _get_client(self) -> ClientAPI:
        if self._client is None:
            settings = self._settings
            if settings.chroma_mode == "http":
                self._client = chromadb.HttpClient(
                    host=settings.chroma_host, port=settings.chroma_port
                )
            else:
                settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
        return self._client

    def _collection(self):
        return self._get_client().get_or_create_collection(
            name=self._settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        ids: list[str],
        texts: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> None:
        if not ids:
            return
        try:
            self._collection().add(
                ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings
            )
        except Exception as exc:
            raise ProviderError(f"Failed to write to ChromaDB: {exc}") from exc

    def query(
        self,
        embedding: list[float],
        n_results: int,
        where: dict[str, Any] | None = None,
        include_embeddings: bool = False,
    ) -> list[VectorSearchResult]:
        include = ["documents", "metadatas", "distances"]
        if include_embeddings:
            include.append("embeddings")
        try:
            response = self._collection().query(
                query_embeddings=[embedding],
                n_results=max(1, n_results),
                where=where,
                include=include,
            )
        except Exception as exc:
            raise ProviderError(f"ChromaDB query failed: {exc}") from exc

        results: list[VectorSearchResult] = []
        ids = response["ids"][0]
        documents = (response.get("documents") or [[]])[0]
        metadatas = (response.get("metadatas") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]
        raw_embeddings = response.get("embeddings")
        embeddings_row = raw_embeddings[0] if raw_embeddings is not None else None
        for position, chunk_id in enumerate(ids):
            vector = None
            if embeddings_row is not None:
                candidate = embeddings_row[position]
                vector = list(candidate) if candidate is not None else None
            results.append(
                VectorSearchResult(
                    id=chunk_id,
                    content=documents[position] or "",
                    metadata=dict(metadatas[position] or {}),
                    score=1.0 - float(distances[position]),
                    embedding=vector,
                )
            )
        return results

    def get_all(self, where: dict[str, Any] | None = None) -> list[StoredChunk]:
        try:
            response = self._collection().get(where=where, include=["documents", "metadatas"])
        except Exception as exc:
            raise ProviderError(f"ChromaDB get failed: {exc}") from exc
        return [
            StoredChunk(
                id=chunk_id,
                content=document or "",
                metadata=dict(metadata or {}),
            )
            for chunk_id, document, metadata in zip(
                response["ids"],
                response.get("documents") or [],
                response.get("metadatas") or [],
                strict=False,
            )
        ]

    def delete_by_document(self, document_id: str) -> None:
        try:
            self._collection().delete(where={"document_id": document_id})
        except Exception as exc:
            raise ProviderError(f"Failed to delete vectors from ChromaDB: {exc}") from exc

    def count(self) -> int:
        return self._collection().count()

    def heartbeat(self) -> bool:
        try:
            self._get_client().heartbeat()
            self._collection()
            return True
        except Exception:
            logger.warning("chromadb heartbeat failed", exc_info=True)
            return False
