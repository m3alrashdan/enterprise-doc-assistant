"""Local Llama-family models via Ollama (LLM_PROVIDER=ollama)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from app.core.exceptions import ProviderError
from app.rag.llm.base import ChatMessage


class OllamaLLM:
    name = "ollama"

    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        timeout: float = 120.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self.model_id = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout

    def _payload(self, messages: list[ChatMessage], stream: bool) -> dict:
        return {
            "model": self.model_id,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": stream,
            "options": {
                "temperature": self._temperature,
                "num_predict": self._max_tokens,
            },
        }

    async def generate(self, messages: list[ChatMessage]) -> str:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._base_url}/api/chat", json=self._payload(messages, stream=False)
                )
                response.raise_for_status()
                return response.json().get("message", {}).get("content", "")
        except httpx.HTTPError as exc:
            raise ProviderError(f"Ollama request failed: {exc}") from exc

    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        try:
            async with (
                httpx.AsyncClient(timeout=self._timeout) as client,
                client.stream(
                    "POST",
                    f"{self._base_url}/api/chat",
                    json=self._payload(messages, stream=True),
                ) as response,
            ):
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    event = json.loads(line)
                    delta = event.get("message", {}).get("content", "")
                    if delta:
                        yield delta
                    if event.get("done"):
                        break
        except httpx.HTTPError as exc:
            raise ProviderError(f"Ollama streaming failed: {exc}") from exc

    async def health_check(self) -> tuple[bool, str]:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self._base_url}/api/tags")
                response.raise_for_status()
                models = [m.get("name", "") for m in response.json().get("models", [])]
            base = self.model_id.split(":")[0]
            if any(name == self.model_id or name.split(":")[0] == base for name in models):
                return True, f"model {self.model_id} available"
            return False, f"model {self.model_id} not pulled (available: {models})"
        except Exception as exc:
            return False, f"ollama unreachable: {exc}"
