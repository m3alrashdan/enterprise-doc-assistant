"""Unit tests for LLM providers (offline: fake + factory + local failure paths)."""

from __future__ import annotations

import pytest

from app.core.exceptions import ProviderError
from app.rag.llm.base import ChatMessage
from app.rag.llm.factory import build_llm_provider
from app.rag.llm.fake import FakeLLM
from app.rag.llm.ollama_provider import OllamaLLM
from tests.conftest import make_test_settings


async def test_fake_llm_answers_with_citation_marker() -> None:
    llm = FakeLLM()
    answer = await llm.generate(
        [ChatMessage(role="user", content="Context sources:\n[1] ...\n\nQuestion: X")]
    )
    assert "[1]" in answer


async def test_fake_llm_condenses_follow_up_verbatim() -> None:
    llm = FakeLLM()
    answer = await llm.generate(
        [
            ChatMessage(
                role="user",
                content="Conversation so far:\n...\n\nFollow-up question: What about dental?",
            )
        ]
    )
    assert answer == "What about dental?"


async def test_fake_llm_streams_full_answer() -> None:
    llm = FakeLLM(canned_response="hello world, this is streaming")
    parts = [
        part async for part in llm.stream([ChatMessage(role="user", content="Context sources: x")])
    ]
    assert len(parts) > 1
    assert "".join(parts) == "hello world, this is streaming"


async def test_fake_llm_health() -> None:
    healthy, _ = await FakeLLM().health_check()
    assert healthy


def test_factory_selects_fake(tmp_path) -> None:
    settings = make_test_settings(tmp_path, llm_provider="fake")
    assert build_llm_provider(settings).name == "fake"


def test_factory_openai_requires_key(tmp_path) -> None:
    settings = make_test_settings(tmp_path, llm_provider="openai", openai_api_key="")
    with pytest.raises(ProviderError):
        build_llm_provider(settings)


def test_factory_builds_ollama_without_network(tmp_path) -> None:
    settings = make_test_settings(
        tmp_path, llm_provider="ollama", ollama_base_url="http://127.0.0.1:9"
    )
    provider = build_llm_provider(settings)
    assert provider.name == "ollama"


async def test_ollama_health_check_reports_unreachable() -> None:
    llm = OllamaLLM(base_url="http://127.0.0.1:9", model="whatever", timeout=1.0)
    healthy, detail = await llm.health_check()
    assert healthy is False
    assert "unreachable" in detail


async def test_ollama_generate_raises_provider_error_when_down() -> None:
    llm = OllamaLLM(base_url="http://127.0.0.1:9", model="whatever", timeout=1.0)
    with pytest.raises(ProviderError):
        await llm.generate([ChatMessage(role="user", content="hi")])
