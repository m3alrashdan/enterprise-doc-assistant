"""Request/response schemas for API v1."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.db.models import DocumentRecord

# --- Documents ----------------------------------------------------------------


class DocumentResponse(BaseModel):
    id: str = Field(examples=["1f2a3b4c5d6e7f80"])
    filename: str = Field(examples=["employee_handbook.pdf"])
    status: str = Field(examples=["ready"], description="pending | processing | ready | failed")
    error: str | None = Field(default=None, description="Failure reason when status=failed")
    chunk_count: int = Field(examples=[42])
    page_count: int | None = Field(default=None, examples=[12])
    size_bytes: int = Field(examples=[102400])
    extension: str = Field(examples=[".pdf"])
    uploader: str | None = None
    tag: str | None = Field(default=None, examples=["hr-policies"])
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_record(cls, record: DocumentRecord) -> DocumentResponse:
        return cls(
            id=record.id,
            filename=record.filename,
            status=record.status,
            error=record.error,
            chunk_count=record.chunk_count,
            page_count=record.page_count,
            size_bytes=record.size_bytes,
            extension=record.extension,
            uploader=record.uploader,
            tag=record.tag,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )


class DocumentListResponse(BaseModel):
    items: list[DocumentResponse]
    total: int = Field(examples=[57])
    page: int = Field(examples=[1])
    page_size: int = Field(examples=[20])


class UploadResult(BaseModel):
    filename: str = Field(examples=["employee_handbook.pdf"])
    accepted: bool
    document_id: str | None = Field(default=None, examples=["1f2a3b4c5d6e7f80"])
    status: str = Field(examples=["pending"], description="pending | rejected | duplicate")
    detail: str | None = Field(default=None, description="Rejection reason, if any")


class UploadResponse(BaseModel):
    results: list[UploadResult]


# --- Chat -----------------------------------------------------------------------


class ChatQueryRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=4000,
        examples=["How many vacation days do new employees get?"],
    )
    conversation_id: str | None = Field(
        default=None,
        max_length=64,
        description="Continue an existing conversation; omit to start fresh.",
        examples=["b71c9d2e"],
    )
    document_ids: list[str] | None = Field(
        default=None, description="Restrict retrieval to these document IDs."
    )
    tag: str | None = Field(
        default=None, description="Restrict retrieval to documents uploaded with this tag."
    )
    top_k: int | None = Field(
        default=None, ge=1, le=20, description="Override the configured number of chunks."
    )


class CitationResponse(BaseModel):
    index: int = Field(examples=[1], description="Marker used in the answer text, e.g. [1]")
    document_id: str = Field(examples=["1f2a3b4c5d6e7f80"])
    document_name: str = Field(examples=["employee_handbook.pdf"])
    page: int | None = Field(default=None, examples=[4])
    section: str | None = Field(default=None, examples=["Paid Time Off"])
    snippet: str = Field(examples=["New employees accrue 20 vacation days per year..."])
    score: float = Field(examples=[0.83], description="Retrieval similarity, higher is better")


class ChatQueryResponse(BaseModel):
    answer: str = Field(examples=["New employees receive 20 paid vacation days per year [1]."])
    citations: list[CitationResponse]
    conversation_id: str = Field(examples=["b71c9d2e"])
    latency_ms: float = Field(examples=[912.4])
    model_used: str = Field(examples=["ollama:qwen2.5:7b"])


class ErrorDetail(BaseModel):
    code: str = Field(examples=["not_found"])
    message: str = Field(examples=["Document 'abc' not found."])
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorEnvelope(BaseModel):
    """Standard error shape returned by every endpoint."""

    error: ErrorDetail
