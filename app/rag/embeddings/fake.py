"""Deterministic offline embeddings for tests and evaluation.

Uses hashed bag-of-words vectors: texts sharing vocabulary land close in
cosine space, so retrieval behaves sensibly (relevant chunks really do rank
first) without any model download or network access.
"""

from __future__ import annotations

import hashlib
import math
import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class FakeEmbeddings:
    name = "fake"

    def __init__(self, dimension: int = 256) -> None:
        self._dimension = dimension

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        for token in _TOKEN_RE.findall(text.lower()):
            digest = hashlib.md5(token.encode()).digest()
            index = int.from_bytes(digest[:4], "little") % self._dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            vector[0] = 1.0
            norm = 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
