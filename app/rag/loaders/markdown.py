"""Markdown loader. Splits on ATX headings, keeping the heading as section."""

from __future__ import annotations

import re
from pathlib import Path

from app.core.exceptions import IngestionError
from app.models.document import LoadedElement

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


class MarkdownLoader:
    extensions: tuple[str, ...] = (".md",)

    def load(self, path: Path) -> list[LoadedElement]:
        text = path.read_text(encoding="utf-8", errors="replace")
        elements: list[LoadedElement] = []
        section: str | None = None
        buffer: list[str] = []
        in_code_block = False

        def flush() -> None:
            content = "\n".join(buffer).strip()
            if content:
                elements.append(LoadedElement(content=content, section=section))
            buffer.clear()

        for line in text.splitlines():
            if line.lstrip().startswith("```"):
                in_code_block = not in_code_block
                buffer.append(line)
                continue
            match = None if in_code_block else _HEADING_RE.match(line)
            if match:
                flush()
                section = match.group(2).strip()
                buffer.append(line)
            else:
                buffer.append(line)
        flush()

        if not elements:
            raise IngestionError("Markdown file is empty.")
        return elements
