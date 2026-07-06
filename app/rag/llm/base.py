"""LLM provider interface.

Providers are async (network-bound). ``generate`` returns the full completion;
``stream`` yields text deltas for Server-Sent Events.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(slots=True)
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


@runtime_checkable
class LLMProvider(Protocol):
    name: str
    model_id: str

    async def generate(self, messages: list[ChatMessage]) -> str:
        """Return the complete assistant response."""
        ...

    def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        """Yield response text deltas as they are produced."""
        ...

    async def health_check(self) -> tuple[bool, str]:
        """(reachable, detail) - used by /health/ready."""
        ...
