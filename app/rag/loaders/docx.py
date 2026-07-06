"""DOCX loader backed by python-docx.

Paragraphs are grouped into one element per section, where a section starts at
each Heading-styled paragraph. Section titles are preserved as chunk metadata
so citations can point at "Security Policy > Data Retention".
"""

from __future__ import annotations

from pathlib import Path

import docx

from app.core.exceptions import IngestionError
from app.models.document import LoadedElement


class DocxLoader:
    extensions: tuple[str, ...] = (".docx",)

    def load(self, path: Path) -> list[LoadedElement]:
        try:
            document = docx.Document(str(path))
        except Exception as exc:
            raise IngestionError(f"Failed to parse DOCX: {exc}") from exc

        elements: list[LoadedElement] = []
        section: str | None = None
        buffer: list[str] = []

        def flush() -> None:
            text = "\n".join(buffer).strip()
            if text:
                elements.append(LoadedElement(content=text, section=section))
            buffer.clear()

        for paragraph in document.paragraphs:
            text = paragraph.text.strip()
            if not text:
                continue
            style = (paragraph.style.name or "") if paragraph.style else ""
            if style.startswith("Heading") or style == "Title":
                flush()
                section = text
                buffer.append(text)  # keep the heading inside its section's text
            else:
                buffer.append(text)
        flush()

        if not elements:
            raise IngestionError("No extractable text found in DOCX.")
        return elements
