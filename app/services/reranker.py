"""Cross-encoder reranking (RERANK_ENABLED=true).

A cross-encoder scores (query, passage) pairs jointly and is far more accurate
than bi-encoder cosine similarity, at the cost of one forward pass per
candidate. It re-orders the shortlist; the original cosine scores are kept on
the results so citation scores stay comparable across requests.

The model is loaded lazily and cached per process.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Protocol

from app.rag.vectorstore.base import VectorSearchResult

logger = logging.getLogger("app.reranker")


class ScoringModel(Protocol):
    """Anything with a CrossEncoder-style ``predict`` (injectable for tests)."""

    def predict(self, pairs: list[tuple[str, str]]) -> Any: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str, model: ScoringModel | None = None) -> None:
        self._model_name = model_name
        self._model: ScoringModel | None = model
        self._lock = threading.Lock()

    def _get_model(self) -> ScoringModel:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import CrossEncoder

                    logger.info("loading rerank model", extra={"model": self._model_name})
                    self._model = CrossEncoder(self._model_name, device="cpu")
        return self._model

    def rerank(self, query: str, results: list[VectorSearchResult]) -> list[VectorSearchResult]:
        if len(results) < 2:
            return results
        scores = self._get_model().predict([(query, result.content) for result in results])
        ranked = sorted(zip(results, scores, strict=True), key=lambda pair: -float(pair[1]))
        return [result for result, _ in ranked]


_cache: dict[str, CrossEncoderReranker] = {}
_cache_lock = threading.Lock()


def get_reranker(model_name: str) -> CrossEncoderReranker:
    with _cache_lock:
        if model_name not in _cache:
            _cache[model_name] = CrossEncoderReranker(model_name)
        return _cache[model_name]
