"""Chunking strategy factory.

To add a semantic strategy: implement ``ChunkingStrategy`` (e.g. embedding
similarity-based boundary detection), register it here, and select it with
``CHUNKING_STRATEGY=semantic``. No other code changes are needed.
"""

from __future__ import annotations

from app.core.config import Settings
from app.core.exceptions import InvalidRequestError
from app.rag.chunking.base import ChunkingStrategy
from app.rag.chunking.recursive import RecursiveChunker


def build_chunker(settings: Settings) -> ChunkingStrategy:
    if settings.chunking_strategy == "recursive":
        return RecursiveChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
    raise InvalidRequestError(
        f"Unknown chunking strategy '{settings.chunking_strategy}'.",
        details={"available": ["recursive"]},
    )
