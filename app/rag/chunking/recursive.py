"""Recursive character chunking via LangChain's RecursiveCharacterTextSplitter.

Splits each loader element independently so page/section metadata is never
mixed across boundaries; chunk_index is global within the document to keep a
stable ordering.
"""

from __future__ import annotations

from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.models.document import ChunkPayload, LoadedElement


class RecursiveChunker:
    name = "recursive"

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def split(
        self, elements: list[LoadedElement], base_metadata: dict[str, Any]
    ) -> list[ChunkPayload]:
        chunks: list[ChunkPayload] = []
        index = 0
        for element in elements:
            for piece in self._splitter.split_text(element.content):
                piece = piece.strip()
                if not piece:
                    continue
                metadata: dict[str, Any] = {**base_metadata, "chunk_index": index}
                if element.page is not None:
                    metadata["page"] = element.page
                if element.section is not None:
                    metadata["section"] = element.section
                chunks.append(ChunkPayload(content=piece, metadata=metadata))
                index += 1
        return chunks
