"""Authentication.

Current scheme: static API keys sent via the ``X-API-Key`` header, validated
against the configured ``API_KEYS`` list. Simple, real, and sufficient for
service-to-service use behind a gateway.

Upgrade path to JWT/OAuth2: replace ``require_api_key`` with a dependency that
validates a Bearer token (issuer/audience/expiry against your IdP's JWKS) and
returns the authenticated principal. Because every protected router depends on
this single function, swapping the scheme is a one-file change -- see
``require_jwt_stub`` below.
"""

from __future__ import annotations

import hashlib
import logging

from fastapi import Request, Security
from fastapi.security import APIKeyHeader

from app.core.exceptions import AuthenticationError

logger = logging.getLogger("app.security")

api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    description="Static API key. Configure valid keys via the API_KEYS env var.",
)


def _principal_for(key: str) -> str:
    """Stable, non-reversible identifier for logging/rate limiting."""
    return "key-" + hashlib.sha256(key.encode()).hexdigest()[:12]


async def require_api_key(
    request: Request,
    api_key: str | None = Security(api_key_header),
) -> str:
    """Validate the API key and return the caller's principal identifier.

    When no keys are configured (development only), auth is disabled and the
    caller is 'anonymous'.
    """
    settings = request.app.state.settings
    if not settings.auth_enabled:
        return "anonymous"
    if api_key is not None and api_key in settings.api_keys:
        return _principal_for(api_key)
    raise AuthenticationError(
        "Missing or invalid API key. Provide it in the X-API-Key header.",
    )


async def require_jwt_stub(request: Request) -> str:  # pragma: no cover - stub
    """Where JWT/OAuth2 plugs in.

    Example shape (using e.g. pyjwt + your IdP's JWKS endpoint)::

        token = request.headers["Authorization"].removeprefix("Bearer ")
        claims = jwt.decode(token, key=jwks_key, audience=..., issuer=...)
        return claims["sub"]
    """
    raise NotImplementedError("JWT auth is not enabled; use API keys (X-API-Key).")
