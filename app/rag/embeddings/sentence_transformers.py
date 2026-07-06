"""Local embeddings via sentence-transformers (Hugging Face).

The model is loaded lazily on first use so application startup stays fast and
unit tests never touch torch. Embeddings are L2-normalised, making cosine
similarity equivalent to dot product.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from app.core.exceptions import ProviderError

logger = logging.getLogger("app.embeddings")

# BGE models are trained with an instruction prefix on the *query* side only.
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class SentenceTransformersEmbeddings:
    name = "sentence_transformers"

    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._model: Any = None
        self._lock = threading.Lock()
        self._query_prefix = _BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""

    def _get_model(self) -> Any:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        from sentence_transformers import SentenceTransformer
                    except ImportError as exc:  # pragma: no cover
                        raise ProviderError("sentence-transformers is not installed.") from exc
                    logger.info("loading embedding model", extra={"model": self._model_name})
                    self._model = SentenceTransformer(self._model_name, device="cpu")
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        vectors = model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([self._query_prefix + text])[0]
