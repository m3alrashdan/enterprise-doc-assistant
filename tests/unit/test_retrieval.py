"""Unit tests for retrieval: filters, threshold, MMR."""

from __future__ import annotations

from typing import Any

from app.rag.vectorstore.base import StoredChunk, VectorSearchResult
from app.services.retrieval import (
    RetrievalService,
    build_where_filter,
    maximal_marginal_relevance,
)
from tests.conftest import make_test_settings


class TestWhereFilter:
    def test_no_restrictions(self) -> None:
        assert build_where_filter() is None

    def test_single_document(self) -> None:
        assert build_where_filter(document_ids=["a"]) == {"document_id": "a"}

    def test_multiple_documents(self) -> None:
        assert build_where_filter(document_ids=["a", "b"]) == {"document_id": {"$in": ["a", "b"]}}

    def test_tag_only(self) -> None:
        assert build_where_filter(tag="hr") == {"tag": "hr"}

    def test_combined_uses_and(self) -> None:
        where = build_where_filter(document_ids=["a"], tag="hr")
        assert where == {"$and": [{"document_id": "a"}, {"tag": "hr"}]}


def make_result(id_: str, score: float, embedding: list[float]) -> VectorSearchResult:
    return VectorSearchResult(
        id=id_,
        content=f"text {id_}",
        metadata={"document_id": id_},
        score=score,
        embedding=embedding,
    )


class TestMMR:
    def test_prefers_diverse_results(self) -> None:
        # NB: when the top pick equals the query vector, relevance == redundancy
        # for every other candidate, so lambda=0.5 ties everything. Use a
        # diversity-weighted lambda where the expected pick is unambiguous.
        query = [1.0, 0.0]
        near_duplicate_a = make_result("a", 0.99, [1.0, 0.0])
        near_duplicate_b = make_result("b", 0.95, [0.95, 0.31])
        different = make_result("c", 0.7, [0.6, 0.8])
        picked = maximal_marginal_relevance(
            query, [near_duplicate_a, near_duplicate_b, different], k=2, lambda_mult=0.3
        )
        assert [r.id for r in picked] == ["a", "c"]

    def test_pure_relevance_when_lambda_one(self) -> None:
        query = [1.0, 0.0]
        results = [
            make_result("a", 0.99, [1.0, 0.0]),
            make_result("b", 0.98, [0.99, 0.14]),
            make_result("c", 0.7, [0.6, 0.8]),
        ]
        picked = maximal_marginal_relevance(query, results, k=2, lambda_mult=1.0)
        assert [r.id for r in picked] == ["a", "b"]

    def test_returns_all_when_fewer_than_k(self) -> None:
        query = [1.0, 0.0]
        results = [make_result("a", 0.9, [1.0, 0.0])]
        assert len(maximal_marginal_relevance(query, results, k=5)) == 1


class StubEmbedder:
    name = "stub"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


class StubVectorStore:
    """Returns canned results; records the filters it was called with."""

    def __init__(self, results: list[VectorSearchResult]) -> None:
        self._results = results
        self.last_where: dict[str, Any] | None = None

    def add(self, ids, texts, metadatas, embeddings) -> None:  # pragma: no cover
        raise NotImplementedError

    def query(self, embedding, n_results, where=None, include_embeddings=False):
        self.last_where = where
        return self._results[:n_results]

    def get_all(self, where=None) -> list[StoredChunk]:
        return []

    def delete_by_document(self, document_id) -> None:  # pragma: no cover
        raise NotImplementedError

    def count(self) -> int:
        return len(self._results)

    def heartbeat(self) -> bool:
        return True


class TestRetrievalService:
    async def test_similarity_threshold_filters_weak_matches(self, tmp_path) -> None:
        settings = make_test_settings(tmp_path, similarity_threshold=0.5, use_mmr=False, top_k=10)
        store = StubVectorStore(
            [
                make_result("strong", 0.9, [1.0, 0.0]),
                make_result("weak", 0.2, [0.0, 1.0]),
            ]
        )
        service = RetrievalService(settings, store, StubEmbedder())
        chunks = await service.retrieve("query")
        assert [c.id for c in chunks] == ["strong"]

    async def test_top_k_override_and_filter_passthrough(self, tmp_path) -> None:
        settings = make_test_settings(tmp_path, use_mmr=False, top_k=5)
        store = StubVectorStore(
            [make_result(str(i), 0.9 - i * 0.01, [1.0, 0.0]) for i in range(10)]
        )
        service = RetrievalService(settings, store, StubEmbedder())
        chunks = await service.retrieve("query", document_ids=["d1"], tag="hr", top_k=2)
        assert len(chunks) == 2
        assert store.last_where == {"$and": [{"document_id": "d1"}, {"tag": "hr"}]}

    async def test_mmr_path_returns_top_k(self, tmp_path) -> None:
        settings = make_test_settings(
            tmp_path,
            use_mmr=True,
            top_k=2,
            fetch_k=10,
            similarity_threshold=-1.0,
            mmr_lambda=0.3,
        )
        store = StubVectorStore(
            [
                make_result("a", 0.99, [1.0, 0.0]),
                make_result("b", 0.95, [0.95, 0.31]),
                make_result("c", 0.7, [0.6, 0.8]),
            ]
        )
        service = RetrievalService(settings, store, StubEmbedder())
        chunks = await service.retrieve("query")
        assert len(chunks) == 2
        assert {c.id for c in chunks} == {"a", "c"}
