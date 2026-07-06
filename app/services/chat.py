"""Chat orchestration: condense follow-ups, retrieve, generate, persist.

Both the blocking and the streaming paths share the same preparation logic;
the streaming path additionally emits retrieval sources before the first
token so clients can render provenance immediately.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from app.core.config import Settings
from app.db.repositories import ConversationRepository
from app.models.chat import ChatResult, Citation
from app.models.document import RetrievedChunk
from app.rag import pipeline
from app.rag.llm.base import LLMProvider
from app.services.retrieval import RetrievalService

logger = logging.getLogger("app.chat")

# SSE event payloads: (event_name, data)
StreamEvent = tuple[str, dict[str, Any]]

_HISTORY_LIMIT = 12


class ChatService:
    def __init__(
        self,
        settings: Settings,
        retrieval: RetrievalService,
        llm: LLMProvider,
        conversations: ConversationRepository,
    ) -> None:
        self._settings = settings
        self._retrieval = retrieval
        self._llm = llm
        self._conversations = conversations

    @property
    def model_used(self) -> str:
        return f"{self._llm.name}:{self._llm.model_id}"

    async def _prepare(
        self,
        question: str,
        conversation_id: str | None,
        document_ids: list[str] | None,
        tag: str | None,
        top_k: int | None,
    ) -> tuple[str, str, list[RetrievedChunk]]:
        """Resolve the conversation, condense follow-ups, run retrieval."""
        new_conversation = conversation_id is None
        resolved_id = conversation_id or uuid.uuid4().hex[:16]

        standalone = question
        if not new_conversation:
            history = await self._conversations.get_messages(resolved_id, limit=_HISTORY_LIMIT)
            if history:
                messages = pipeline.build_condense_messages(history, question)
                standalone = (await self._llm.generate(messages)).strip() or question
                if standalone != question:
                    logger.info(
                        "condensed follow-up question",
                        extra={"conversation_id": resolved_id},
                    )

        chunks = await self._retrieval.retrieve(
            standalone, document_ids=document_ids, tag=tag, top_k=top_k
        )
        return resolved_id, standalone, chunks

    async def _persist_turn(
        self, conversation_id: str, question: str, answer: str, citations: list[Citation]
    ) -> None:
        await self._conversations.ensure(conversation_id)
        await self._conversations.add_message(conversation_id, "user", question)
        await self._conversations.add_message(
            conversation_id, "assistant", answer, citations=[c.to_dict() for c in citations]
        )

    async def query(
        self,
        question: str,
        conversation_id: str | None = None,
        document_ids: list[str] | None = None,
        tag: str | None = None,
        top_k: int | None = None,
    ) -> ChatResult:
        started = time.perf_counter()
        resolved_id, standalone, chunks = await self._prepare(
            question, conversation_id, document_ids, tag, top_k
        )

        if not chunks:
            answer = pipeline.NOT_FOUND_ANSWER
            citations: list[Citation] = []
        else:
            messages = pipeline.build_answer_messages(standalone, chunks)
            answer = (await self._llm.generate(messages)).strip()
            citations = pipeline.assemble_citations(answer, chunks)

        await self._persist_turn(resolved_id, question, answer, citations)
        return ChatResult(
            answer=answer,
            citations=citations,
            conversation_id=resolved_id,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            model_used=self.model_used,
        )

    async def stream_query(
        self,
        question: str,
        conversation_id: str | None = None,
        document_ids: list[str] | None = None,
        tag: str | None = None,
        top_k: int | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Yield ``sources`` -> ``token``* -> ``done`` events for SSE."""
        started = time.perf_counter()
        resolved_id, standalone, chunks = await self._prepare(
            question, conversation_id, document_ids, tag, top_k
        )

        yield (
            "sources",
            {
                "conversation_id": resolved_id,
                "sources": [
                    pipeline.chunk_to_citation(i, chunk).to_dict()
                    for i, chunk in enumerate(chunks, start=1)
                ],
            },
        )

        if not chunks:
            answer = pipeline.NOT_FOUND_ANSWER
            yield ("token", {"text": answer})
        else:
            messages = pipeline.build_answer_messages(standalone, chunks)
            parts: list[str] = []
            async for delta in self._llm.stream(messages):
                parts.append(delta)
                yield ("token", {"text": delta})
            answer = "".join(parts).strip()

        citations = pipeline.assemble_citations(answer, chunks)
        await self._persist_turn(resolved_id, question, answer, citations)
        yield (
            "done",
            {
                "answer": answer,
                "citations": [c.to_dict() for c in citations],
                "conversation_id": resolved_id,
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                "model_used": self.model_used,
            },
        )
