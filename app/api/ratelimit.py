"""Fixed-window in-memory rate limiting middleware.

Keyed by API key when present, else client IP. Health probes and API docs are
exempt. The window state is per-process: for multi-replica deployments swap
the counter for a shared store (Redis) behind the same middleware.
"""

from __future__ import annotations

import re
import threading
import time

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.api.errors import error_envelope

_EXEMPT_PREFIXES = ("/health", "/docs", "/redoc", "/openapi")

_RATE_RE = re.compile(r"^\s*(\d+)\s*/\s*(second|minute|hour)\s*$")
_WINDOW_SECONDS = {"second": 1, "minute": 60, "hour": 3600}


def parse_rate(rate: str) -> tuple[int, int]:
    """Parse ``"60/minute"`` into ``(limit, window_seconds)``."""
    match = _RATE_RE.match(rate)
    if not match:
        raise ValueError(f"Invalid rate limit '{rate}'; expected e.g. '60/minute'.")
    return int(match.group(1)), _WINDOW_SECONDS[match.group(2)]


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, rate: str = "60/minute") -> None:
        super().__init__(app)
        self._limit, self._window = parse_rate(rate)
        self._counters: dict[str, tuple[int, int]] = {}  # key -> (window_id, count)
        self._lock = threading.Lock()

    @staticmethod
    def _client_key(request: Request) -> str:
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"key:{api_key}"
        return f"ip:{request.client.host if request.client else 'unknown'}"

    def _register_hit(self, key: str) -> tuple[bool, int]:
        """Count a request; returns (allowed, seconds_until_reset)."""
        now = time.time()
        window_id = int(now // self._window)
        reset_in = int(self._window - (now % self._window)) + 1
        with self._lock:
            current_window, count = self._counters.get(key, (window_id, 0))
            if current_window != window_id:
                count = 0
            count += 1
            self._counters[key] = (window_id, count)
            if len(self._counters) > 10_000:  # opportunistic cleanup
                self._counters = {k: v for k, v in self._counters.items() if v[0] == window_id}
        return count <= self._limit, reset_in

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.url.path.startswith(_EXEMPT_PREFIXES):
            return await call_next(request)

        allowed, reset_in = self._register_hit(self._client_key(request))
        if not allowed:
            return JSONResponse(
                status_code=429,
                content=error_envelope(
                    "rate_limited",
                    "Rate limit exceeded. Slow down and retry shortly.",
                    {"retry_after_seconds": reset_in},
                ),
                headers={"Retry-After": str(reset_in)},
            )
        return await call_next(request)
