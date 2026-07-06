"""Document lifecycle: upload validation, dedup, listing, deletion."""

from __future__ import annotations

import hashlib
import logging
import uuid
from pathlib import Path

import anyio

from app.core.config import Settings
from app.core.exceptions import (
    FileTooLargeError,
    InvalidRequestError,
    NotFoundError,
    UnsupportedFileTypeError,
)
from app.db.models import DocumentRecord
from app.db.repositories import DocumentRepository
from app.models.document import DocumentStatus
from app.rag.vectorstore.base import VectorStore

logger = logging.getLogger("app.documents")


class DuplicateDocumentUpload(Exception):
    """Internal signal carrying the pre-existing record for a duplicate upload."""

    def __init__(self, existing: DocumentRecord) -> None:
        super().__init__(existing.id)
        self.existing = existing


class DocumentService:
    def __init__(
        self,
        settings: Settings,
        repository: DocumentRepository,
        vector_store: VectorStore,
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._vector_store = vector_store

    async def register_upload(
        self,
        filename: str,
        content: bytes,
        uploader: str | None = None,
        tag: str | None = None,
    ) -> DocumentRecord:
        """Validate and persist an upload; returns the pending document record.

        Raises UnsupportedFileTypeError / FileTooLargeError / InvalidRequestError
        on validation failure and DuplicateDocumentUpload when the exact same
        content (by SHA-256) was already ingested.
        """
        safe_name = Path(filename or "").name
        if not safe_name:
            raise InvalidRequestError("Uploaded file is missing a filename.")
        extension = Path(safe_name).suffix.lower()
        if extension not in self._settings.allowed_extensions:
            raise UnsupportedFileTypeError(
                f"File type '{extension or 'unknown'}' is not supported.",
                details={"filename": safe_name, "allowed": self._settings.allowed_extensions},
            )
        if not content:
            raise InvalidRequestError(f"File '{safe_name}' is empty.")
        if len(content) > self._settings.max_upload_size_bytes:
            raise FileTooLargeError(
                f"File '{safe_name}' exceeds the {self._settings.max_upload_size_mb} MB limit.",
                details={"size_bytes": len(content)},
            )

        content_hash = hashlib.sha256(content).hexdigest()
        existing = await self._repository.get_by_hash(content_hash)
        if existing is not None:
            raise DuplicateDocumentUpload(existing)

        document_id = uuid.uuid4().hex
        stored_path = self._settings.upload_dir / f"{document_id}{extension}"
        stored_path.parent.mkdir(parents=True, exist_ok=True)
        await anyio.to_thread.run_sync(stored_path.write_bytes, content)

        record = DocumentRecord(
            id=document_id,
            filename=safe_name,
            content_hash=content_hash,
            size_bytes=len(content),
            extension=extension,
            stored_path=str(stored_path),
            status=DocumentStatus.PENDING.value,
            uploader=uploader,
            tag=tag,
        )
        await self._repository.create(record)
        logger.info(
            "document registered",
            extra={"document_id": document_id, "filename": safe_name, "bytes": len(content)},
        )
        return record

    async def get_document(self, document_id: str) -> DocumentRecord:
        record = await self._repository.get(document_id)
        if record is None:
            raise NotFoundError(f"Document '{document_id}' not found.")
        return record

    async def list_documents(
        self, page: int = 1, page_size: int = 20
    ) -> tuple[list[DocumentRecord], int]:
        offset = (page - 1) * page_size
        return await self._repository.list(offset=offset, limit=page_size)

    async def delete_document(self, document_id: str) -> None:
        """Remove vectors, the metadata record and the stored file."""
        record = await self.get_document(document_id)
        await anyio.to_thread.run_sync(self._vector_store.delete_by_document, document_id)
        await self._repository.delete(document_id)
        try:
            Path(record.stored_path).unlink(missing_ok=True)
        except OSError:
            logger.warning("could not delete stored file", extra={"document_id": document_id})
        logger.info("document deleted", extra={"document_id": document_id})
