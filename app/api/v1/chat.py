"""Chat endpoints: blocking query and Server-Sent Events streaming."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.deps import get_container
from app.api.v1.schemas import (
    ChatQueryRequest,
    ChatQueryResponse,
    CitationResponse,
    ErrorEnvelope,
)
from app.core.container import Container
from app.core.exceptions import AppError

logger = logging.getLogger("app.chat.api")

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post(
    "/query",
    response_model=ChatQueryResponse,
    summary="Ask a question over the ingested documents",
    responses={502: {"model": ErrorEnvelope}},
)
async def chat_query(
    payload: ChatQueryRequest,
    container: Container = Depends(get_container),
) -> ChatQueryResponse:
    """Retrieves the most relevant chunks and generates a cited answer.

    Pass `conversation_id` from a previous response to ask follow-up
    questions; they are condensed into standalone queries before retrieval.
    If the answer is not present in the documents, the assistant says so.
    """
    result = await container.chat_service.query(
        question=payload.question,
        conversation_id=payload.conversation_id,
        document_ids=payload.document_ids,
        tag=payload.tag,
        top_k=payload.top_k,
    )
    return ChatQueryResponse(
        answer=result.answer,
        citations=[CitationResponse(**c.to_dict()) for c in result.citations],
        conversation_id=result.conversation_id,
        latency_ms=result.latency_ms,
        model_used=result.model_used,
    )


@router.post(
    "/stream",
    summary="Ask a question and stream the answer via Server-Sent Events",
    response_description=(
        "SSE stream: one `sources` event (retrieved chunks), then `token` events "
        "with text deltas, then a final `done` event with the full payload "
        "(answer, citations, latency_ms, model_used). Errors arrive as an "
        "`error` event carrying the standard error envelope."
    ),
)
async def chat_stream(
    payload: ChatQueryRequest,
    container: Container = Depends(get_container),
) -> StreamingResponse:
    async def event_source():
        try:
            events = container.chat_service.stream_query(
                question=payload.question,
                conversation_id=payload.conversation_id,
                document_ids=payload.document_ids,
                tag=payload.tag,
                top_k=payload.top_k,
            )
            async for event, data in events:
                yield f"event: {event}\ndata: {json.dumps(data)}\n\n"
        except AppError as exc:
            envelope = {"code": exc.code, "message": exc.message, "details": exc.details}
            yield f"event: error\ndata: {json.dumps(envelope)}\n\n"
        except Exception:
            logger.exception("unexpected error during stream")
            envelope = {"code": "internal_error", "message": "Streaming failed.", "details": {}}
            yield f"event: error\ndata: {json.dumps(envelope)}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
