"""Unit tests for chunking strategies."""

from __future__ import annotations

import pytest

from app.core.exceptions import InvalidRequestError
from app.models.document import LoadedElement
from app.rag.chunking.factory import build_chunker
from app.rag.chunking.recursive import RecursiveChunker
from tests.conftest import make_test_settings


def test_short_element_stays_single_chunk() -> None:
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.split([LoadedElement(content="Short text.")], {"document_id": "d1"})
    assert len(chunks) == 1
    assert chunks[0].content == "Short text."
    assert chunks[0].metadata["document_id"] == "d1"
    assert chunks[0].metadata["chunk_index"] == 0


def test_long_element_is_split_with_overlap() -> None:
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=30)
    sentences = " ".join(f"Sentence number {i} about the vacation policy." for i in range(30))
    chunks = chunker.split([LoadedElement(content=sentences)], {})
    assert len(chunks) > 3
    for chunk in chunks:
        assert len(chunk.content) <= 100
    # consecutive chunks share overlapping text
    assert chunks[0].content[-10:] in sentences
    indexes = [chunk.metadata["chunk_index"] for chunk in chunks]
    assert indexes == list(range(len(chunks)))


def test_page_and_section_metadata_preserved() -> None:
    chunker = RecursiveChunker(chunk_size=80, chunk_overlap=10)
    elements = [
        LoadedElement(content="Alpha " * 40, page=3),
        LoadedElement(content="Beta " * 40, section="Security"),
    ]
    chunks = chunker.split(elements, {"document_id": "d1", "document_name": "x.pdf"})
    page_chunks = [c for c in chunks if "page" in c.metadata]
    section_chunks = [c for c in chunks if "section" in c.metadata]
    assert page_chunks and all(c.metadata["page"] == 3 for c in page_chunks)
    assert section_chunks and all(c.metadata["section"] == "Security" for c in section_chunks)
    # chunk_index is global across elements
    indexes = [c.metadata["chunk_index"] for c in chunks]
    assert indexes == list(range(len(chunks)))


def test_whitespace_only_elements_produce_no_chunks() -> None:
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=0)
    assert chunker.split([LoadedElement(content="   \n  ")], {}) == []


def test_factory_builds_configured_strategy(tmp_path) -> None:
    settings = make_test_settings(tmp_path, chunk_size=123, chunk_overlap=7)
    chunker = build_chunker(settings)
    assert chunker.name == "recursive"


def test_factory_rejects_unknown_strategy(tmp_path) -> None:
    settings = make_test_settings(tmp_path, chunking_strategy="quantum")
    with pytest.raises(InvalidRequestError):
        build_chunker(settings)
