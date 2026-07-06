"""Vector store interface.

Scores are cosine similarities in [~-1, 1], higher is better. Implementations
are synchronous; services dispatch them to a worker thread.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(slots=True)
class VectorSearchResult:
    id: str
    content: str
    metadata: dict[str, Any]
    score: float
    embedding: list[float] | None = None


@dataclass(slots=True)
class StoredChunk:
    id: str
    content: str
    metadata: dict[str, Any]


@runtime_checkable
class VectorStore(Protocol):
    def add(
        self,
        ids: list[str],
        texts: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> None: ...

    def query(
        self,
        embedding: list[float],
        n_results: int,
        where: dict[str, Any] | None = None,
        include_embeddings: bool = False,
    ) -> list[VectorSearchResult]: ...

    def get_all(self, where: dict[str, Any] | None = None) -> list[StoredChunk]:
        """Return all stored chunks (used by BM25 hybrid search)."""
        ...

    def delete_by_document(self, document_id: str) -> None: ...

    def count(self) -> int: ...

    def heartbeat(self) -> bool:
        """True if the store is reachable and usable."""
        ...
