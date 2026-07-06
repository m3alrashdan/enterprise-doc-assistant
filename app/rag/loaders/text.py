"""Plain-text loader."""

from __future__ import annotations

from pathlib import Path

from app.core.exceptions import IngestionError
from app.models.document import LoadedElement


class TextLoader:
    extensions: tuple[str, ...] = (".txt",)

    def load(self, path: Path) -> list[LoadedElement]:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            raise IngestionError("Text file is empty.")
        return [LoadedElement(content=text)]
