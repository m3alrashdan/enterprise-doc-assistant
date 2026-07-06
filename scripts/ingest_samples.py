"""Upload the sample_data documents through the running API.

Usage:
    make ingest-sample                 # server on localhost:8000
    BASE_URL=... API_KEY=... python scripts/ingest_samples.py

Generates the sample PDF on first run, uploads every document, then polls
until ingestion finishes.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.generate_sample_pdf import generate  # noqa: E402

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("API_KEY", "dev-secret-key")
SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"
UPLOADABLE = {".pdf", ".md", ".html", ".htm", ".txt", ".docx"}


def main() -> int:
    pdf_path = SAMPLE_DIR / "employee_handbook.pdf"
    if not pdf_path.exists():
        generate(pdf_path)

    files = sorted(p for p in SAMPLE_DIR.iterdir() if p.suffix.lower() in UPLOADABLE)
    if not files:
        print("no sample documents found")
        return 1

    client = httpx.Client(base_url=BASE_URL, headers={"X-API-Key": API_KEY}, timeout=60)

    payload = [("files", (path.name, path.read_bytes())) for path in files]
    response = client.post("/api/v1/documents/upload", files=payload, data={"tag": "sample"})
    response.raise_for_status()
    results = response.json()["results"]

    pending: dict[str, str] = {}
    for result in results:
        status = result["status"]
        print(
            f"  {result['filename']:<28} -> {status}"
            + (f" ({result['detail']})" if result.get("detail") else "")
        )
        if result["accepted"]:
            pending[result["document_id"]] = result["filename"]

    deadline = time.time() + 300
    while pending and time.time() < deadline:
        time.sleep(1.5)
        for document_id in list(pending):
            doc = client.get(f"/api/v1/documents/{document_id}").json()
            if doc["status"] in ("ready", "failed"):
                name = pending.pop(document_id)
                detail = f" ({doc['error']})" if doc.get("error") else ""
                print(f"  {name:<28} -> {doc['status']}: " f"{doc['chunk_count']} chunks{detail}")

    if pending:
        print(f"timed out waiting for: {list(pending.values())}")
        return 1
    print("done - ask questions via POST /api/v1/chat/query or the UI at /")
    return 0


if __name__ == "__main__":
    sys.exit(main())
