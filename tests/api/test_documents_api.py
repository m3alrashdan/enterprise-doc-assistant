"""API tests for document endpoints (httpx.AsyncClient against the ASGI app)."""

from __future__ import annotations

import httpx

HANDBOOK_MD = b"# PTO\nEmployees receive twenty vacation days per year.\n"


def upload_files(*specs: tuple[str, bytes]) -> list[tuple[str, tuple[str, bytes, str]]]:
    return [("files", (name, content, "application/octet-stream")) for name, content in specs]


async def test_upload_and_status_lifecycle(client: httpx.AsyncClient) -> None:
    resp = await client.post("/api/v1/documents/upload", files=upload_files(("h.md", HANDBOOK_MD)))
    assert resp.status_code == 202
    result = resp.json()["results"][0]
    assert result["accepted"] is True
    document_id = result["document_id"]

    # background ingestion has completed by the time the transport returns
    resp = await client.get(f"/api/v1/documents/{document_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready"
    assert body["chunk_count"] > 0
    assert body["filename"] == "h.md"


async def test_multi_file_upload_reports_each_file(client: httpx.AsyncClient) -> None:
    resp = await client.post(
        "/api/v1/documents/upload",
        files=upload_files(("good.md", HANDBOOK_MD), ("bad.exe", b"nope"), ("empty.txt", b"")),
    )
    assert resp.status_code == 202
    results = {r["filename"]: r for r in resp.json()["results"]}
    assert results["good.md"]["accepted"] is True
    assert results["bad.exe"]["status"] == "rejected"
    assert "not supported" in results["bad.exe"]["detail"]
    assert results["empty.txt"]["status"] == "rejected"


async def test_duplicate_upload_is_flagged(client: httpx.AsyncClient) -> None:
    first = await client.post("/api/v1/documents/upload", files=upload_files(("a.md", HANDBOOK_MD)))
    dup = await client.post("/api/v1/documents/upload", files=upload_files(("b.md", HANDBOOK_MD)))
    result = dup.json()["results"][0]
    assert result["status"] == "duplicate"
    assert result["document_id"] == first.json()["results"][0]["document_id"]


async def test_list_documents_paginates(client: httpx.AsyncClient) -> None:
    for i in range(3):
        content = f"# Doc {i}\nUnique content number {i}.\n".encode()
        await client.post("/api/v1/documents/upload", files=upload_files((f"doc{i}.md", content)))

    resp = await client.get("/api/v1/documents", params={"page": 1, "page_size": 2})
    body = resp.json()
    assert body["total"] == 3
    assert len(body["items"]) == 2
    resp = await client.get("/api/v1/documents", params={"page": 2, "page_size": 2})
    assert len(resp.json()["items"]) == 1


async def test_get_missing_document_returns_envelope(client: httpx.AsyncClient) -> None:
    resp = await client.get("/api/v1/documents/nope")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"


async def test_delete_document(client: httpx.AsyncClient) -> None:
    resp = await client.post("/api/v1/documents/upload", files=upload_files(("d.md", HANDBOOK_MD)))
    document_id = resp.json()["results"][0]["document_id"]

    resp = await client.delete(f"/api/v1/documents/{document_id}")
    assert resp.status_code == 204
    resp = await client.get(f"/api/v1/documents/{document_id}")
    assert resp.status_code == 404


async def test_invalid_pagination_rejected(client: httpx.AsyncClient) -> None:
    resp = await client.get("/api/v1/documents", params={"page": 0})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"
