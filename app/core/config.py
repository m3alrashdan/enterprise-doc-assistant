"""Application configuration.

Single source of truth for every tunable in the system. All values are read
from the environment (or an ``.env`` file) via pydantic-settings; nothing is
hardcoded elsewhere. See ``.env.example`` for documentation of each variable.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration, grouped by concern."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Application ---------------------------------------------------------
    app_name: str = "Enterprise Document Assistant"
    app_version: str = "0.1.0"
    environment: Literal["development", "production", "test"] = "development"
    log_level: str = "INFO"
    log_json: bool = True
    api_v1_prefix: str = "/api/v1"

    # --- Security ------------------------------------------------------------
    # Empty list disables authentication (development only; a warning is logged).
    api_keys: Annotated[list[str], NoDecode] = []
    rate_limit_enabled: bool = True
    rate_limit: str = "60/minute"

    # --- Storage ---------------------------------------------------------------
    data_dir: Path = Path("data")
    database_url: str = "sqlite+aiosqlite:///data/app.db"
    max_upload_size_mb: int = 25
    allowed_extensions: Annotated[list[str], NoDecode] = [
        ".pdf",
        ".docx",
        ".txt",
        ".md",
        ".html",
        ".htm",
    ]

    # --- Vector store (ChromaDB) ------------------------------------------------
    chroma_mode: Literal["embedded", "http"] = "embedded"
    chroma_persist_dir: Path = Path("data/chroma")
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_collection: str = "documents"

    # --- Chunking ----------------------------------------------------------------
    chunking_strategy: str = "recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # --- Embeddings ----------------------------------------------------------------
    embedding_provider: Literal["sentence_transformers", "openai", "fake"] = "sentence_transformers"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_batch_size: int = 32

    # --- Retrieval ------------------------------------------------------------------
    top_k: int = 4
    fetch_k: int = 20
    similarity_threshold: float = 0.25
    use_mmr: bool = True
    mmr_lambda: float = 0.5
    rerank_enabled: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    hybrid_search_enabled: bool = False
    hybrid_alpha: float = 0.5

    # --- LLM ---------------------------------------------------------------------------
    llm_provider: Literal["openai", "ollama", "fake"] = "ollama"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024
    llm_timeout_seconds: float = 120.0

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str | None = None

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"

    @field_validator("api_keys", "allowed_extensions", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        """Allow comma-separated strings for list fields in env vars."""
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def auth_enabled(self) -> bool:
        return len(self.api_keys) > 0


@lru_cache
def load_settings() -> Settings:
    """Process-wide settings for entrypoints (uvicorn, scripts).

    Request handlers must use the ``get_settings`` dependency instead so tests
    can inject their own instance.
    """
    return Settings()
