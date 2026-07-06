"""Embedding provider factory driven purely by configuration."""

from __future__ import annotations

from app.core.config import Settings
from app.core.exceptions import InvalidRequestError
from app.rag.embeddings.base import EmbeddingProvider
from app.rag.embeddings.fake import FakeEmbeddings


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider == "sentence_transformers":
        from app.rag.embeddings.sentence_transformers import SentenceTransformersEmbeddings

        return SentenceTransformersEmbeddings(
            model_name=settings.embedding_model,
            batch_size=settings.embedding_batch_size,
        )
    if settings.embedding_provider == "openai":
        from app.rag.embeddings.openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            api_key=settings.openai_api_key,
            model=settings.openai_embedding_model,
            base_url=settings.openai_base_url,
            batch_size=settings.embedding_batch_size,
        )
    if settings.embedding_provider == "fake":
        return FakeEmbeddings()
    raise InvalidRequestError(f"Unknown embedding provider '{settings.embedding_provider}'.")
