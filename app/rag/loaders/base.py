"""Loader interface. One implementation per file format."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from app.models.document import LoadedElement


@runtime_checkable
class DocumentLoader(Protocol):
    """Extracts text elements (with page/section metadata) from a file."""

    extensions: tuple[str, ...]

    def load(self, path: Path) -> list[LoadedElement]:
        """Parse the file into elements. Raises IngestionError on parse failure."""
        ...
