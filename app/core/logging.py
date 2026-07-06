"""Structured JSON logging with request-ID correlation.

Every log line emitted while serving a request carries the request ID set by
``RequestContextMiddleware`` (see app/api/middleware.py) via a contextvar, so
logs from any layer of the stack can be correlated in an aggregator.
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

# Attributes present on every LogRecord; anything else was passed via `extra=`.
_STANDARD_ATTRS = frozenset(
    {
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "module", "msecs",
        "message", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "taskName", "thread", "threadName",
    }
)  # fmt: skip


class JsonFormatter(logging.Formatter):
    """Format records as single-line JSON objects for log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = request_id_var.get()
        if request_id:
            payload["request_id"] = request_id
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class ConsoleFormatter(logging.Formatter):
    """Human-readable formatter for local development."""

    def format(self, record: logging.LogRecord) -> str:
        request_id = request_id_var.get()
        prefix = f"[{request_id}] " if request_id else ""
        base = (
            f"{datetime.now(UTC).strftime('%H:%M:%S')} "
            f"{record.levelname:<8} {record.name}: {prefix}{record.getMessage()}"
        )
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def configure_logging(level: str = "INFO", json_logs: bool = True) -> None:
    """Configure the root logger to emit structured logs to stdout."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if json_logs else ConsoleFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # We emit our own request-completion logs with timing and request IDs.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    for noisy in ("httpx", "chromadb", "sentence_transformers", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
