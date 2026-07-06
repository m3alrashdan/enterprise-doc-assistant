"""ChromaDB wrapper tests against a temporary embedded store (offline)."""

from __future__ import annotations

import pytest

from app.rag.embeddings.fake import FakeEmbeddings
from app.rag.vectorstore.chroma import ChromaVectorStore
from tests.conftest import make_test_settings


@pytest.fixture
def store(tmp_path) -> ChromaVectorStore:
    return ChromaVectorStore(make_test_settings(tmp_path))


@pytest.fixture
def embedder() -> FakeEmbeddings:
    return FakeEmbeddings()


def seed(store: ChromaVectorStore, embedder: FakeEmbeddings) -> None:
    texts = [
        "employees receive twenty vacation days per year",
        "the vpn is mandatory for all remote connections",
        "quarterly financial reports are due in april",
    ]
    metadatas = [
        {"document_id": "doc-hr", "document_name": "hr.md", "chunk_index": 0},
        {"document_id": "doc-sec", "document_name": "sec.md", "chunk_index": 0},
        {"document_id": "doc-fin", "document_name": "fin.md", "chunk_index": 0},
    ]
    store.add(
        ids=["doc-hr:0", "doc-sec:0", "doc-fin:0"],
        texts=texts,
        metadatas=metadatas,
        embeddings=embedder.embed_documents(texts),
    )


def test_add_query_roundtrip(store, embedder) -> None:
    seed(store, embedder)
    results = store.query(embedder.embed_query("how many vacation days"), n_results=3)
    assert len(results) == 3
    assert results[0].metadata["document_id"] == "doc-hr"
    assert results[0].score >= results[-1].score
    assert "vacation" in results[0].content


def test_metadata_filter(store, embedder) -> None:
    seed(store, embedder)
    results = store.query(
        embedder.embed_query("vacation days"),
        n_results=3,
        where={"document_id": "doc-sec"},
    )
    assert len(results) == 1
    assert results[0].metadata["document_id"] == "doc-sec"


def test_query_can_return_embeddings(store, embedder) -> None:
    seed(store, embedder)
    results = store.query(embedder.embed_query("vpn"), n_results=2, include_embeddings=True)
    assert results[0].embedding is not None
    assert len(results[0].embedding) == 256


def test_delete_by_document(store, embedder) -> None:
    seed(store, embedder)
    assert store.count() == 3
    store.delete_by_document("doc-hr")
    assert store.count() == 2
    results = store.query(embedder.embed_query("vacation days"), n_results=3)
    assert all(r.metadata["document_id"] != "doc-hr" for r in results)


def test_get_all_with_filter(store, embedder) -> None:
    seed(store, embedder)
    chunks = store.get_all(where={"document_id": "doc-fin"})
    assert len(chunks) == 1
    assert "financial" in chunks[0].content


def test_heartbeat(store) -> None:
    assert store.heartbeat() is True
