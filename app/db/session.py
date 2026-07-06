"""Async database engine and session factory."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.db.models import Base


def create_engine_and_factory(
    database_url: str,
) -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(database_url, echo=False)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    return engine, factory


async def init_db(engine: AsyncEngine) -> None:
    """Create tables if they do not exist.

    Deliberate trade-off: for a single-service SQLite/Postgres metadata store,
    ``create_all`` on startup keeps operations simple. Introduce Alembic when
    the schema starts evolving in production.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
