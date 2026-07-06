"""RAG pipeline: prompt construction and citation assembly.

The pipeline turns retrieved chunks into numbered context sources, instructs
the model to answer strictly from them with ``[n]`` markers, and afterwards
resolves those markers back to the source chunks to build the citation list.

Prompts are LangChain ``ChatPromptTemplate`` objects; provider-agnostic
``ChatMessage`` lists come out, so any ``LLMProvider`` can execute them.
"""

from __future__ import annotations

import re

from langchain_core.prompts import ChatPromptTemplate

from app.models.chat import Citation
from app.models.document import RetrievedChunk
from app.rag.llm.base import ChatMessage

NOT_FOUND_ANSWER = "I could not find an answer to this question in the available documents."

_SNIPPET_LENGTH = 240

_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an enterprise document assistant. Answer employee questions "
            "using ONLY the numbered context sources provided.\n"
            "Rules:\n"
            "1. Base every statement strictly on the context sources. Never use outside "
            "knowledge and never invent information.\n"
            "2. Cite the supporting source inline after each claim using its bracketed "
            "number, e.g. [1] or [2][3]. Only cite numbers that exist in the context.\n"
            "3. If the context does not contain the information needed, reply exactly: "
            f'"{NOT_FOUND_ANSWER}"\n'
            "4. Be concise, factual and professional.",
        ),
        (
            "human",
            "Context sources:\n{context}\n\nQuestion: {question}\n\n"
            "Answer (with [n] citations):",
        ),
    ]
)

_CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the follow-up question as a single self-contained question that "
            "preserves all context from the conversation. Return ONLY the rewritten "
            "question, nothing else.",
        ),
        (
            "human",
            "Conversation so far:\n{history}\n\nFollow-up question: {question}",
        ),
    ]
)

_CITATION_RE = re.compile(r"\[(\d{1,2})\]")

_LC_ROLE_MAP = {"system": "system", "human": "user", "ai": "assistant"}


def _to_chat_messages(prompt: ChatPromptTemplate, **variables: str) -> list[ChatMessage]:
    return [
        ChatMessage(role=_LC_ROLE_MAP.get(message.type, "user"), content=str(message.content))
        for message in prompt.format_messages(**variables)
    ]


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render chunks as numbered sources with their provenance."""
    blocks: list[str] = []
    for number, chunk in enumerate(chunks, start=1):
        origin = [f"document: {chunk.metadata.get('document_name', 'unknown')}"]
        if chunk.metadata.get("page") is not None:
            origin.append(f"page: {chunk.metadata['page']}")
        if chunk.metadata.get("section"):
            origin.append(f"section: {chunk.metadata['section']}")
        blocks.append(f"[{number}] ({', '.join(origin)})\n{chunk.content}")
    return "\n\n".join(blocks)


def build_answer_messages(question: str, chunks: list[RetrievedChunk]) -> list[ChatMessage]:
    return _to_chat_messages(_ANSWER_PROMPT, context=format_context(chunks), question=question)


def build_condense_messages(history: list[tuple[str, str]], question: str) -> list[ChatMessage]:
    """History is a list of (role, content) pairs, oldest first."""
    rendered = "\n".join(f"{role}: {content}" for role, content in history)
    return _to_chat_messages(_CONDENSE_PROMPT, history=rendered, question=question)


def chunk_to_citation(index: int, chunk: RetrievedChunk) -> Citation:
    snippet = chunk.content[:_SNIPPET_LENGTH]
    if len(chunk.content) > _SNIPPET_LENGTH:
        snippet = snippet.rsplit(" ", 1)[0] + "…"
    page = chunk.metadata.get("page")
    return Citation(
        index=index,
        document_id=str(chunk.metadata.get("document_id", "")),
        document_name=str(chunk.metadata.get("document_name", "unknown")),
        page=int(page) if page is not None else None,
        section=chunk.metadata.get("section"),
        snippet=snippet,
        score=round(chunk.score, 4),
    )


def assemble_citations(answer: str, chunks: list[RetrievedChunk]) -> list[Citation]:
    """Map ``[n]`` markers in the answer back to their source chunks.

    - Markers pointing outside the source range are ignored.
    - A "not found" answer gets no citations.
    - An answer without any markers falls back to citing all retrieved chunks,
      so provenance is never silently dropped.
    """
    if not chunks or NOT_FOUND_ANSWER in answer:
        return []

    seen: list[int] = []
    for match in _CITATION_RE.finditer(answer):
        number = int(match.group(1))
        if 1 <= number <= len(chunks) and number not in seen:
            seen.append(number)

    if not seen:
        return [chunk_to_citation(i, chunk) for i, chunk in enumerate(chunks, start=1)]
    return [chunk_to_citation(number, chunks[number - 1]) for number in sorted(seen)]
