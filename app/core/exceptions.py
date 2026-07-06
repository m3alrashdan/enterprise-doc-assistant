"""Application exception hierarchy.

Services raise these; the global handlers in app/api/errors.py translate them
into the consistent ``{"error": {"code", "message", "details"}}`` envelope.
Routers never construct HTTP error responses by hand.
"""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Base class for all expected application failures."""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"


class InvalidRequestError(AppError):
    status_code = 400
    code = "invalid_request"


class AuthenticationError(AppError):
    status_code = 401
    code = "unauthorized"


class DuplicateDocumentError(AppError):
    status_code = 409
    code = "duplicate_document"


class DocumentNotReadyError(AppError):
    status_code = 409
    code = "document_not_ready"


class FileTooLargeError(AppError):
    status_code = 413
    code = "file_too_large"


class UnsupportedFileTypeError(AppError):
    status_code = 415
    code = "unsupported_file_type"


class IngestionError(AppError):
    status_code = 500
    code = "ingestion_failed"


class ProviderError(AppError):
    """An upstream provider (LLM, embeddings, vector store) failed."""

    status_code = 502
    code = "provider_error"
