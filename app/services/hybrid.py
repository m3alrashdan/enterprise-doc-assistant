"""Hybrid search: BM25 + vector score fusion (HYBRID_SEARCH_ENABLED=true).

Re-scores the semantic candidate set with BM25 computed over the full
(filter-matching) corpus, then fuses both signals:

    combined = alpha * vector_score_norm + (1 - alpha) * bm25_score_norm

Scope note: fusion re-ranks the vector candidates rather than merging in
BM25-only candidates, which keeps embeddings available for the MMR stage.
The corpus is fetched per query, which is fine at internal-documents scale;
swap in a persistent BM25 index if the corpus grows large.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from rank_bm25 import BM25Okapi

from app.rag.vectorstore.base import VectorSearchResult, VectorStore

logger = logging.getLogger("app.hybrid")

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _normalise(scores: list[float]) -> list[float]:
    if not scores:
        return []
    low, high = min(scores), max(scores)
    if high == low:
        return [1.0 for _ in scores]
    return [(score - low) / (high - low) for score in scores]


def fuse_bm25(
    query: str,
    results: list[VectorSearchResult],
    vector_store: VectorStore,
    where: dict[str, Any] | None,
    alpha: float,
) -> list[VectorSearchResult]:
    """Reorder ``results`` by fused vector+BM25 score. Returns a new list."""
    if not results:
        return results

    corpus = vector_store.get_all(where=where)
    if not corpus:
        return results

    bm25 = BM25Okapi([_tokenize(chunk.content) for chunk in corpus])
    corpus_scores = bm25.get_scores(_tokenize(query))
    bm25_by_id = dict(zip((chunk.id for chunk in corpus), corpus_scores, strict=True))

    vector_norm = _normalise([result.score for result in results])
    bm25_norm = _normalise([float(bm25_by_id.get(result.id, 0.0)) for result in results])

    fused = sorted(
        zip(results, vector_norm, bm25_norm, strict=True),
        key=lambda item: -(alpha * item[1] + (1.0 - alpha) * item[2]),
    )
    logger.debug("hybrid fusion applied", extra={"candidates": len(results)})
    return [result for result, _, _ in fused]
