"""Document management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, Form, Query, UploadFile, status

from app.api.deps import get_container
from app.api.v1.schemas import (
    DocumentListResponse,
    DocumentResponse,
    ErrorEnvelope,
    UploadResponse,
    UploadResult,
)
from app.core.container import Container
from app.core.exceptions import AppError, InvalidRequestError
from app.services.documents import DuplicateDocumentUpload

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "/upload",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=UploadResponse,
    summary="Upload one or more documents",
    responses={400: {"model": ErrorEnvelope}, 413: {"model": ErrorEnvelope}},
)
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: list[UploadFile],
    uploader: str | None = Form(default=None, description="Who is uploading (audit metadata)"),
    tag: str | None = Form(default=None, description="Optional tag for filtered retrieval"),
    container: Container = Depends(get_container),
) -> UploadResponse:
    """Accepts PDF, DOCX, TXT, MD and HTML files (multipart/form-data).

    Files are validated (type, size, duplicate content) and indexed
    asynchronously; poll `GET /documents/{id}` until status is `ready`.
    Each file is reported individually so one bad file never blocks a batch.
    """
    if not files:
        raise InvalidRequestError("No files provided.")

    results: list[UploadResult] = []
    for file in files:
        content = await file.read()
        try:
            record = await container.document_service.register_upload(
                filename=file.filename or "",
                content=content,
                uploader=uploader,
                tag=tag,
            )
        except DuplicateDocumentUpload as dup:
            results.append(
                UploadResult(
                    filename=file.filename or "",
                    accepted=False,
                    document_id=dup.existing.id,
                    status="duplicate",
                    detail="Identical content already uploaded.",
                )
            )
        except AppError as exc:
            results.append(
                UploadResult(
                    filename=file.filename or "",
                    accepted=False,
                    status="rejected",
                    detail=exc.message,
                )
            )
        else:
            background_tasks.add_task(container.ingestion_service.ingest_document, record.id)
            results.append(
                UploadResult(
                    filename=record.filename,
                    accepted=True,
                    document_id=record.id,
                    status=record.status,
                )
            )
    return UploadResponse(results=results)


@router.get("", response_model=DocumentListResponse, summary="List documents")
async def list_documents(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    container: Container = Depends(get_container),
) -> DocumentListResponse:
    records, total = await container.document_service.list_documents(page, page_size)
    return DocumentListResponse(
        items=[DocumentResponse.from_record(record) for record in records],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Get a document and its processing status",
    responses={404: {"model": ErrorEnvelope}},
)
async def get_document(
    document_id: str,
    container: Container = Depends(get_container),
) -> DocumentResponse:
    record = await container.document_service.get_document(document_id)
    return DocumentResponse.from_record(record)


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document (vectors + metadata + file)",
    responses={404: {"model": ErrorEnvelope}},
)
async def delete_document(
    document_id: str,
    container: Container = Depends(get_container),
) -> None:
    await container.document_service.delete_document(document_id)
