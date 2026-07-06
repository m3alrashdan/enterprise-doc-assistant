"""Retrieval: semantic search with similarity threshold, MMR diversity,
optional metadata filtering, and hooks for reranking / hybrid search."""

from __future__ import annotations

import logging
import math
from functools import partial
from typing import Any

import anyio

from app.core.config import Settings
from app.models.document import RetrievedChunk
from app.rag.embeddings.base import EmbeddingProvider
from app.rag.vectorstore.base import VectorSearchResult, VectorStore

logger = logging.getLogger("app.retrieval")


def build_where_filter(
    document_ids: list[str] | None = None, tag: str | None = None
) -> dict[str, Any] | None:
    """Compose a Chroma metadata filter from the optional restrictions."""
    conditions: list[dict[str, Any]] = []
    if document_ids:
        if len(document_ids) == 1:
            conditions.append({"document_id": document_ids[0]})
        else:
            conditions.append({"document_id": {"$in": document_ids}})
    if tag:
        conditions.append({"tag": tag})
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def maximal_marginal_relevance(
    query_embedding: list[float],
    candidates: list[VectorSearchResult],
    k: int,
    lambda_mult: float = 0.5,
) -> list[VectorSearchResult]:
    """Greedy MMR: balance query relevance against redundancy among picks."""
    pool = [c for c in candidates if c.embedding is not None]
    if len(pool) <= k:
        return pool or candidates[:k]

    selected: list[VectorSearchResult] = []
    remaining = pool.copy()
    while remaining and len(selected) < k:
        best_score = -math.inf
        best_item = remaining[0]
        for candidate in remaining:
            relevance = _cosine(query_embedding, candidate.embedding or [])
            redundancy = max(
                (_cosine(candidate.embedding or [], s.embedding or []) for s in selected),
                default=0.0,
            )
            score = lambda_mult * relevance - (1.0 - lambda_mult) * redundancy
            if score > best_score:
                best_score = score
                best_item = candidate
        selected.append(best_item)
        remaining.remove(best_item)
    return selected


class RetrievalService:
    def __init__(
        self,
        settings: Settings,
        vector_store: VectorStore,
        embedder: EmbeddingProvider,
    ) -> None:
        self._settings = settings
        self._vector_store = vector_store
        self._embedder = embedder

    async def retrieve(
        self,
        query: str,
        *,
        document_ids: list[str] | None = None,
        tag: str | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Return the top chunks for a query, ordered by final rank."""
        settings = self._settings
        k = top_k or settings.top_k
        where = build_where_filter(document_ids, tag)

        query_embedding = await anyio.to_thread.run_sync(self._embedder.embed_query, query)
        fetch_k = max(settings.fetch_k, k)
        results = await anyio.to_thread.run_sync(
            partial(
                self._vector_store.query,
                query_embedding,
                fetch_k,
                where=where,
                include_embeddings=settings.use_mmr,
            )
        )

        results = [r for r in results if r.score >= settings.similarity_threshold]

        if settings.hybrid_search_enabled:
            results = await self._hybrid_rerank(query, results, where)

        if settings.use_mmr and len(results) > k:
            results = maximal_marginal_relevance(query_embedding, results, k, settings.mmr_lambda)
        else:
            results = results[:k]

        if settings.rerank_enabled and results:
            results = await self._cross_encoder_rerank(query, results)

        results = results[:k]
        logger.info(
            "retrieval completed",
            extra={"candidates": fetch_k, "returned": len(results), "filtered": where is not None},
        )
        return [
            RetrievedChunk(id=r.id, content=r.content, metadata=r.metadata, score=r.score)
            for r in results
        ]

    async def _hybrid_rerank(
        self,
        query: str,
        results: list[VectorSearchResult],
        where: dict[str, Any] | None,
    ) -> list[VectorSearchResult]:
        """BM25 + vector score fusion. Implemented in app/services/hybrid.py."""
        from app.services.hybrid import fuse_bm25

        return await anyio.to_thread.run_sync(
            partial(
                fuse_bm25,
                query,
                results,
                self._vector_store,
                where,
                self._settings.hybrid_alpha,
            )
        )

    async def _cross_encoder_rerank(
        self, query: str, results: list[VectorSearchResult]
    ) -> list[VectorSearchResult]:
        """Cross-encoder reranking. Implemented in app/services/reranker.py."""
        from app.services.reranker import get_reranker

        reranker = get_reranker(self._settings.rerank_model)
        return await anyio.to_thread.run_sync(partial(reranker.rerank, query, results))
