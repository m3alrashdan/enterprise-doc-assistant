"""LLM provider factory driven purely by configuration."""

from __future__ import annotations

from app.core.config import Settings
from app.core.exceptions import InvalidRequestError
from app.rag.llm.base import LLMProvider
from app.rag.llm.fake import FakeLLM


def build_llm_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "openai":
        from app.rag.llm.openai_provider import OpenAILLM

        return OpenAILLM(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            base_url=settings.openai_base_url,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout_seconds,
        )
    if settings.llm_provider == "ollama":
        from app.rag.llm.ollama_provider import OllamaLLM

        return OllamaLLM(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout_seconds,
        )
    if settings.llm_provider == "fake":
        return FakeLLM()
    raise InvalidRequestError(f"Unknown LLM provider '{settings.llm_provider}'.")
