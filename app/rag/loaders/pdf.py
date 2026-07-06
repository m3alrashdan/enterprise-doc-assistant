"""PDF loader backed by pypdf. One element per page."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from app.core.exceptions import IngestionError
from app.models.document import LoadedElement


class PdfLoader:
    extensions: tuple[str, ...] = (".pdf",)

    def load(self, path: Path) -> list[LoadedElement]:
        try:
            reader = PdfReader(str(path))
        except Exception as exc:  # pypdf raises a zoo of exception types
            raise IngestionError(f"Failed to parse PDF: {exc}") from exc

        elements: list[LoadedElement] = []
        for index, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:  # a single broken page should not sink the document
                text = ""
            text = text.strip()
            if text:
                elements.append(LoadedElement(content=text, page=index))
        if not elements:
            raise IngestionError(
                "No extractable text found in PDF (it may be scanned images without OCR)."
            )
        return elements
