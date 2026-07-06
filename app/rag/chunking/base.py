"""Chunking strategy interface (Strategy pattern).

A strategy turns loader elements into embedding-ready chunks. Element-level
metadata (page, section) is merged with document-level metadata onto every
chunk, so citations survive all the way to the answer.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from app.models.document import ChunkPayload, LoadedElement


@runtime_checkable
class ChunkingStrategy(Protocol):
    name: str

    def split(
        self, elements: list[LoadedElement], base_metadata: dict[str, Any]
    ) -> list[ChunkPayload]:
        """Split elements into chunks, attaching merged metadata to each."""
        ...
