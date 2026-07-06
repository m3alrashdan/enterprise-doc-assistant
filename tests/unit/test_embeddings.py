"""Unit tests for embedding providers (offline: fake + factory)."""

from __future__ import annotations

import math

import pytest

from app.core.exceptions import ProviderError
from app.rag.embeddings.factory import build_embedding_provider
from app.rag.embeddings.fake import FakeEmbeddings
from tests.conftest import make_test_settings


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_fake_embeddings_are_deterministic_and_normalised() -> None:
    provider = FakeEmbeddings()
    first = provider.embed_query("vacation policy")
    second = provider.embed_query("vacation policy")
    assert first == second
    assert math.isclose(sum(v * v for v in first), 1.0, rel_tol=1e-6)


def test_fake_embeddings_respect_vocabulary_overlap() -> None:
    provider = FakeEmbeddings()
    query = provider.embed_query("how many vacation days do employees get")
    relevant = provider.embed_query("employees get twenty vacation days per year")
    unrelated = provider.embed_query("kubernetes cluster deployment pipeline")
    assert cosine(query, relevant) > cosine(query, unrelated)


def test_embed_documents_matches_query_for_same_text() -> None:
    provider = FakeEmbeddings()
    assert provider.embed_documents(["same text"])[0] == provider.embed_query("same text")


def test_factory_selects_fake(tmp_path) -> None:
    settings = make_test_settings(tmp_path, embedding_provider="fake")
    assert build_embedding_provider(settings).name == "fake"


def test_factory_openai_requires_key(tmp_path) -> None:
    settings = make_test_settings(tmp_path, embedding_provider="openai", openai_api_key="")
    with pytest.raises(ProviderError):
        build_embedding_provider(settings)


def test_factory_sentence_transformers_is_lazy(tmp_path) -> None:
    """Building the provider must not download or load any model."""
    settings = make_test_settings(tmp_path, embedding_provider="sentence_transformers")
    provider = build_embedding_provider(settings)
    assert provider.name == "sentence_transformers"
    assert provider._model is None  # model only loads on first embed call
