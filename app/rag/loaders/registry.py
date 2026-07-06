"""Maps file extensions to loader implementations."""

from __future__ import annotations

from app.core.exceptions import UnsupportedFileTypeError
from app.rag.loaders.base import DocumentLoader
from app.rag.loaders.docx import DocxLoader
from app.rag.loaders.html import HtmlLoader
from app.rag.loaders.markdown import MarkdownLoader
from app.rag.loaders.pdf import PdfLoader
from app.rag.loaders.text import TextLoader

_LOADERS: tuple[DocumentLoader, ...] = (
    PdfLoader(),
    DocxLoader(),
    TextLoader(),
    MarkdownLoader(),
    HtmlLoader(),
)

LOADERS_BY_EXTENSION: dict[str, DocumentLoader] = {
    ext: loader for loader in _LOADERS for ext in loader.extensions
}


def get_loader(extension: str) -> DocumentLoader:
    """Return the loader for a file extension (e.g. ``".pdf"``)."""
    loader = LOADERS_BY_EXTENSION.get(extension.lower())
    if loader is None:
        supported = ", ".join(sorted(LOADERS_BY_EXTENSION))
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{extension}'. Supported: {supported}",
            details={"extension": extension, "supported": sorted(LOADERS_BY_EXTENSION)},
        )
    return loader
