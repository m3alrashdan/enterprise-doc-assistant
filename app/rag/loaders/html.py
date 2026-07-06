"""HTML loader backed by BeautifulSoup.

Strips script/style/nav noise and groups body text under the nearest h1-h4
heading, preserved as section metadata.
"""

from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup

from app.core.exceptions import IngestionError
from app.models.document import LoadedElement

_HEADING_TAGS = {"h1", "h2", "h3", "h4"}
_CONTENT_TAGS = _HEADING_TAGS | {"p", "li", "td", "th", "pre", "blockquote"}
_NOISE_TAGS = ("script", "style", "noscript", "nav", "footer", "header")


class HtmlLoader:
    extensions: tuple[str, ...] = (".html", ".htm")

    def load(self, path: Path) -> list[LoadedElement]:
        markup = path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(markup, "html.parser")
        for tag in soup(_NOISE_TAGS):
            tag.decompose()

        elements: list[LoadedElement] = []
        section: str | None = None
        buffer: list[str] = []

        def flush() -> None:
            content = "\n".join(buffer).strip()
            if content:
                elements.append(LoadedElement(content=content, section=section))
            buffer.clear()

        for node in soup.find_all(_CONTENT_TAGS):
            text = node.get_text(separator=" ", strip=True)
            if not text:
                continue
            if node.name in _HEADING_TAGS:
                flush()
                section = text
            buffer.append(text)
        flush()

        if not elements:  # no structural tags; fall back to raw text
            text = soup.get_text(separator="\n", strip=True)
            if not text:
                raise IngestionError("No extractable text found in HTML.")
            elements.append(LoadedElement(content=text))
        return elements
