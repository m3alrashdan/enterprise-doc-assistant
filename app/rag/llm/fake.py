"""Deterministic offline LLM for tests and evaluation.

Understands the two prompt shapes produced by ``app.rag.pipeline``:
- condense prompts: returns the follow-up question verbatim (standalone)
- answer prompts: returns a short grounded answer citing source [1]

A canned response can be injected for tests that need specific citation
patterns (e.g. "see [2] and [3]").
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from app.rag.llm.base import ChatMessage

_CONDENSE_MARKER = "Follow-up question:"
_CONTEXT_MARKER = "Context sources:"


class FakeLLM:
    name = "fake"
    model_id = "fake-model"

    def __init__(self, canned_response: str | None = None) -> None:
        self._canned = canned_response
        self.calls: list[list[ChatMessage]] = []  # inspected by tests

    async def generate(self, messages: list[ChatMessage]) -> str:
        self.calls.append(messages)
        user_content = messages[-1].content
        if _CONDENSE_MARKER in user_content:
            return user_content.rsplit(_CONDENSE_MARKER, 1)[1].strip().rstrip("?") + "?"
        if self._canned is not None:
            return self._canned
        if _CONTEXT_MARKER in user_content:
            return "Based on the provided documents, the answer is supported by the context [1]."
        return "OK."

    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        text = await self.generate(messages)
        for i in range(0, len(text), 12):
            yield text[i : i + 12]

    async def health_check(self) -> tuple[bool, str]:
        return True, "fake provider"
