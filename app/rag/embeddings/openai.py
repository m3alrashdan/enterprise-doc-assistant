"""OpenAI embeddings provider (switch with EMBEDDING_PROVIDER=openai)."""

from __future__ import annotations

from app.core.exceptions import ProviderError


class OpenAIEmbeddings:
    name = "openai"

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: str | None = None,
        batch_size: int = 64,
    ) -> None:
        if not api_key:
            raise ProviderError("OPENAI_API_KEY is required for OpenAI embeddings.")
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        try:
            for start in range(0, len(texts), self._batch_size):
                batch = texts[start : start + self._batch_size]
                response = self._client.embeddings.create(model=self._model, input=batch)
                ordered = sorted(response.data, key=lambda item: item.index)
                vectors.extend(item.embedding for item in ordered)
        except Exception as exc:
            raise ProviderError(f"OpenAI embeddings request failed: {exc}") from exc
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
