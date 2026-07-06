"""OpenAI chat provider (LLM_PROVIDER=openai)."""

from __future__ import annotations

from collections.abc import AsyncIterator

from app.core.exceptions import ProviderError
from app.rag.llm.base import ChatMessage


class OpenAILLM:
    name = "openai"

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        timeout: float = 120.0,
    ) -> None:
        if not api_key:
            raise ProviderError("OPENAI_API_KEY is required when LLM_PROVIDER=openai.")
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model_id = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    @staticmethod
    def _to_payload(messages: list[ChatMessage]) -> list[dict[str, str]]:
        return [{"role": message.role, "content": message.content} for message in messages]

    async def generate(self, messages: list[ChatMessage]) -> str:
        try:
            response = await self._client.chat.completions.create(
                model=self.model_id,
                messages=self._to_payload(messages),
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        except Exception as exc:
            raise ProviderError(f"OpenAI completion failed: {exc}") from exc
        return response.choices[0].message.content or ""

    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        try:
            stream = await self._client.chat.completions.create(
                model=self.model_id,
                messages=self._to_payload(messages),
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                stream=True,
            )
            async for event in stream:
                if event.choices and event.choices[0].delta.content:
                    yield event.choices[0].delta.content
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"OpenAI streaming failed: {exc}") from exc

    async def health_check(self) -> tuple[bool, str]:
        try:
            await self._client.models.retrieve(self.model_id)
            return True, f"model {self.model_id} available"
        except Exception as exc:
            return False, str(exc)
