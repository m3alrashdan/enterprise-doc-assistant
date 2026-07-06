"""Chat domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Citation:
    """A resolved inline citation: the [n] marker mapped to its source chunk."""

    index: int
    document_id: str
    document_name: str
    snippet: str
    score: float
    page: int | None = None
    section: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "page": self.page,
            "section": self.section,
            "snippet": self.snippet,
            "score": self.score,
        }


@dataclass(slots=True)
class ChatResult:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    conversation_id: str = ""
    latency_ms: float = 0.0
    model_used: str = ""
