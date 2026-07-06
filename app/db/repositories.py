"""Repositories encapsulating all database access.

Each method opens its own session, so repositories are safe to use from both
request handlers and background ingestion tasks.
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import ConversationRecord, DocumentRecord, MessageRecord
from app.models.document import DocumentStatus


class DocumentRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def create(self, record: DocumentRecord) -> DocumentRecord:
        async with self._session_factory() as session:
            session.add(record)
            await session.commit()
            return record

    async def get(self, document_id: str) -> DocumentRecord | None:
        async with self._session_factory() as session:
            return await session.get(DocumentRecord, document_id)

    async def get_by_hash(self, content_hash: str) -> DocumentRecord | None:
        async with self._session_factory() as session:
            result = await session.execute(
                select(DocumentRecord).where(DocumentRecord.content_hash == content_hash)
            )
            return result.scalar_one_or_none()

    async def list(self, offset: int = 0, limit: int = 20) -> tuple[list[DocumentRecord], int]:
        async with self._session_factory() as session:
            total = (
                await session.execute(select(func.count()).select_from(DocumentRecord))
            ).scalar_one()
            result = await session.execute(
                select(DocumentRecord)
                .order_by(DocumentRecord.created_at.desc())
                .offset(offset)
                .limit(limit)
            )
            return list(result.scalars().all()), total

    async def update_status(
        self,
        document_id: str,
        status: DocumentStatus,
        *,
        error: str | None = None,
        chunk_count: int | None = None,
        page_count: int | None = None,
    ) -> None:
        async with self._session_factory() as session:
            record = await session.get(DocumentRecord, document_id)
            if record is None:
                return
            record.status = status.value
            record.error = error
            if chunk_count is not None:
                record.chunk_count = chunk_count
            if page_count is not None:
                record.page_count = page_count
            await session.commit()

    async def delete(self, document_id: str) -> bool:
        async with self._session_factory() as session:
            record = await session.get(DocumentRecord, document_id)
            if record is None:
                return False
            await session.delete(record)
            await session.commit()
            return True


class ConversationRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def ensure(self, conversation_id: str) -> None:
        async with self._session_factory() as session:
            if await session.get(ConversationRecord, conversation_id) is None:
                session.add(ConversationRecord(id=conversation_id))
                await session.commit()

    async def exists(self, conversation_id: str) -> bool:
        async with self._session_factory() as session:
            return await session.get(ConversationRecord, conversation_id) is not None

    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        citations: list[dict[str, Any]] | None = None,
    ) -> None:
        async with self._session_factory() as session:
            session.add(
                MessageRecord(
                    conversation_id=conversation_id,
                    role=role,
                    content=content,
                    citations_json=json.dumps(citations) if citations else None,
                )
            )
            await session.commit()

    async def get_messages(self, conversation_id: str, limit: int = 20) -> list[tuple[str, str]]:
        """Return the last ``limit`` (role, content) pairs, oldest first."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(MessageRecord)
                .where(MessageRecord.conversation_id == conversation_id)
                .order_by(MessageRecord.created_at.desc(), MessageRecord.id.desc())
                .limit(limit)
            )
            messages = list(result.scalars().all())
        return [(m.role, m.content) for m in reversed(messages)]

    async def delete_conversation(self, conversation_id: str) -> None:
        async with self._session_factory() as session:
            await session.execute(
                delete(MessageRecord).where(MessageRecord.conversation_id == conversation_id)
            )
            record = await session.get(ConversationRecord, conversation_id)
            if record is not None:
                await session.delete(record)
            await session.commit()
