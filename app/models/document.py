"""Domain models shared across services and the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class DocumentStatus(StrEnum):
    """Lifecycle of an uploaded document."""

    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


@dataclass(slots=True)
class LoadedElement:
    """A unit of text produced by a document loader.

    For PDFs an element is a page; for structured formats (DOCX/MD/HTML) it is
    a run of content under one section heading. Loaders keep elements coarse;
    the chunking strategy decides final chunk boundaries.
    """

    content: str
    page: int | None = None
    section: str | None = None


@dataclass(slots=True)
class ChunkPayload:
    """A chunk ready for embedding, carrying its full metadata."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievedChunk:
    """A chunk returned by retrieval, scored and ready for citation assembly."""

    id: str
    content: str
    metadata: dict[str, Any]
    score: float
