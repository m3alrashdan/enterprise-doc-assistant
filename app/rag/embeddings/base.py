"""Embedding provider interface.

Implementations are synchronous (CPU- or network-bound); services run them in
a worker thread (``anyio.to_thread``) to keep the event loop responsive.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passages for indexing."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query (may apply model-specific query prefixes)."""
        ...
