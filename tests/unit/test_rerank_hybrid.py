"""Unit tests for cross-encoder reranking and BM25 hybrid fusion (offline)."""

from __future__ import annotations

from app.rag.vectorstore.base import StoredChunk, VectorSearchResult
from app.services.hybrid import fuse_bm25
from app.services.reranker import CrossEncoderReranker


def result(id_: str, content: str, score: float) -> VectorSearchResult:
    return VectorSearchResult(id=id_, content=content, metadata={"document_id": id_}, score=score)


class StubCrossEncoder:
    """Scores pairs by naive token overlap - deterministic, no model download."""

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores = []
        for query, passage in pairs:
            q_tokens = set(query.lower().split())
            p_tokens = set(passage.lower().split())
            scores.append(len(q_tokens & p_tokens))
        return scores


class TestCrossEncoderReranker:
    def test_reorders_by_model_score(self) -> None:
        reranker = CrossEncoderReranker("stub-model", model=StubCrossEncoder())
        results = [
            result("weak", "totally unrelated content here", 0.9),
            result("strong", "employees vacation days policy", 0.5),
        ]
        ranked = reranker.rerank("how many vacation days for employees", results)
        assert [r.id for r in ranked] == ["strong", "weak"]

    def test_single_result_passthrough_without_model(self) -> None:
        reranker = CrossEncoderReranker("stub-model", model=None)
        results = [result("only", "text", 0.5)]
        assert reranker.rerank("query", results) == results  # model never loaded


class StubStore:
    def __init__(self, chunks: list[StoredChunk]) -> None:
        self._chunks = chunks

    def get_all(self, where=None) -> list[StoredChunk]:
        return self._chunks


class TestHybridFusion:
    def test_bm25_boosts_exact_keyword_match(self) -> None:
        # NB: corpus needs >2 docs; with df=1 of N=2, BM25Okapi's idf is
        # log(1.5/1.5) = 0 and every score degenerates to zero.
        corpus = [
            StoredChunk(id="a", content="general information about benefits", metadata={}),
            StoredChunk(id="b", content="error code E4521 means disk failure", metadata={}),
            StoredChunk(id="c", content="office opening hours and parking", metadata={}),
            StoredChunk(id="d", content="travel booking guidelines for staff", metadata={}),
        ]
        # vector search ranked the generic chunk first, but the query contains
        # a rare exact token that BM25 catches
        results = [
            result("a", "general information about benefits", 0.8),
            result("b", "error code E4521 means disk failure", 0.75),
        ]
        fused = fuse_bm25("what does E4521 mean", results, StubStore(corpus), None, alpha=0.3)
        assert [r.id for r in fused] == ["b", "a"]

    def test_alpha_one_keeps_vector_order(self) -> None:
        corpus = [
            StoredChunk(id="a", content="alpha text", metadata={}),
            StoredChunk(id="b", content="beta text E4521", metadata={}),
        ]
        results = [
            result("a", "alpha text", 0.9),
            result("b", "beta text E4521", 0.2),
        ]
        fused = fuse_bm25("E4521", results, StubStore(corpus), None, alpha=1.0)
        assert [r.id for r in fused] == ["a", "b"]

    def test_empty_inputs_are_passthrough(self) -> None:
        assert fuse_bm25("q", [], StubStore([]), None, 0.5) == []
        results = [result("a", "text", 0.5)]
        assert fuse_bm25("q", results, StubStore([]), None, 0.5) == results


class WiringStore:
    """Vector store stub covering both query() and get_all()."""

    def __init__(self, results: list[VectorSearchResult]) -> None:
        self._results = results

    def query(self, embedding, n_results, where=None, include_embeddings=False):
        return self._results[:n_results]

    def get_all(self, where=None) -> list[StoredChunk]:
        return [StoredChunk(id=r.id, content=r.content, metadata=r.metadata) for r in self._results]

    def add(self, *args, **kwargs) -> None:  # pragma: no cover
        raise NotImplementedError

    def delete_by_document(self, document_id) -> None:  # pragma: no cover
        raise NotImplementedError

    def count(self) -> int:
        return len(self._results)

    def heartbeat(self) -> bool:
        return True


class WiringEmbedder:
    name = "stub"

    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


class TestRetrievalServiceFlags:
    """The config flags reach the fusion/rerank implementations correctly."""

    async def test_hybrid_flag_reranks_candidates(self, tmp_path) -> None:
        from app.services.retrieval import RetrievalService
        from tests.conftest import make_test_settings

        settings = make_test_settings(
            tmp_path,
            hybrid_search_enabled=True,
            hybrid_alpha=0.1,
            use_mmr=False,
            top_k=2,
            similarity_threshold=-1.0,
        )
        store = WiringStore(
            [
                result("generic", "general information about benefits", 0.9),
                result("exact", "error code E4521 means disk failure", 0.85),
                result("filler1", "office opening hours and parking", 0.3),
                result("filler2", "travel booking guidelines for staff", 0.2),
            ]
        )
        service = RetrievalService(settings, store, WiringEmbedder())
        chunks = await service.retrieve("what does E4521 mean")
        assert chunks[0].id == "exact"

    async def test_rerank_flag_uses_cross_encoder(self, tmp_path) -> None:
        from app.services import reranker as reranker_module
        from app.services.retrieval import RetrievalService
        from tests.conftest import make_test_settings

        reranker_module._cache["stub-model"] = CrossEncoderReranker(
            "stub-model", model=StubCrossEncoder()
        )
        settings = make_test_settings(
            tmp_path,
            rerank_enabled=True,
            rerank_model="stub-model",
            use_mmr=False,
            top_k=2,
            similarity_threshold=-1.0,
        )
        store = WiringStore(
            [
                result("weak", "totally unrelated content here", 0.9),
                result("strong", "employees vacation days policy", 0.5),
            ]
        )
        service = RetrievalService(settings, store, WiringEmbedder())
        chunks = await service.retrieve("how many vacation days for employees")
        assert chunks[0].id == "strong"
