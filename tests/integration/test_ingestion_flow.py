"""End-to-end ingestion: upload -> chunk -> embed -> index -> status tracking.

Runs against a real (temporary) ChromaDB with fake embeddings; no network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.container import Container, build_container
from app.core.exceptions import FileTooLargeError, UnsupportedFileTypeError
from app.models.document import DocumentStatus
from app.services.documents import DuplicateDocumentUpload
from tests.conftest import make_test_settings

POLICY_MD = b"""# Vacation Policy
Full-time employees accrue twenty (20) vacation days per calendar year.

# Expense Policy
Meal expenses are reimbursed up to fifty dollars per day with receipts.
"""


@pytest.fixture
async def container(tmp_path) -> Container:
    container = await build_container(make_test_settings(tmp_path))
    yield container
    await container.shutdown()


async def test_full_ingestion_flow(container: Container) -> None:
    record = await container.document_service.register_upload(
        "policies.md", POLICY_MD, uploader="alice", tag="hr"
    )
    assert record.status == DocumentStatus.PENDING.value

    await container.ingestion_service.ingest_document(record.id)

    refreshed = await container.document_service.get_document(record.id)
    assert refreshed.status == DocumentStatus.READY.value
    assert refreshed.chunk_count > 0

    # vectors are queryable and carry full citation metadata
    embedding = container.embedder.embed_query("how many vacation days")
    results = container.vector_store.query(embedding, n_results=2)
    assert results
    top = results[0]
    assert top.metadata["document_id"] == record.id
    assert top.metadata["document_name"] == "policies.md"
    assert top.metadata["section"] in {"Vacation Policy", "Expense Policy"}
    assert top.metadata["uploader"] == "alice"
    assert top.metadata["tag"] == "hr"


async def test_ingestion_failure_is_recorded(container: Container) -> None:
    record = await container.document_service.register_upload("broken.pdf", b"not really a pdf")
    await container.ingestion_service.ingest_document(record.id)
    refreshed = await container.document_service.get_document(record.id)
    assert refreshed.status == DocumentStatus.FAILED.value
    assert refreshed.error


async def test_duplicate_upload_detected(container: Container) -> None:
    await container.document_service.register_upload("a.md", POLICY_MD)
    with pytest.raises(DuplicateDocumentUpload):
        await container.document_service.register_upload("renamed.md", POLICY_MD)


async def test_unsupported_type_rejected(container: Container) -> None:
    with pytest.raises(UnsupportedFileTypeError):
        await container.document_service.register_upload("virus.exe", b"data")


async def test_size_limit_enforced(tmp_path) -> None:
    settings = make_test_settings(tmp_path / "sized", max_upload_size_mb=1)
    container = await build_container(settings)
    try:
        with pytest.raises(FileTooLargeError):
            await container.document_service.register_upload("big.txt", b"x" * (1024 * 1024 + 1))
    finally:
        await container.shutdown()


async def test_delete_removes_vectors_and_file(container: Container) -> None:
    record = await container.document_service.register_upload("todelete.md", POLICY_MD)
    await container.ingestion_service.ingest_document(record.id)
    assert container.vector_store.count() > 0

    await container.document_service.delete_document(record.id)
    assert container.vector_store.count() == 0
    assert not Path(record.stored_path).exists()
