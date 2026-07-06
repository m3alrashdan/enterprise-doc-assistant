"""Dependency container: builds and owns all providers for the app lifetime.

Constructed once at startup (see app/main.py lifespan) from ``Settings``.
Providers are behind Protocol interfaces, so tests build a container with
fakes and production swaps implementations purely through configuration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.core.config import Settings

logger = logging.getLogger("app.container")

# (healthy, detail) per component name.
ReadinessReport = dict[str, tuple[bool, str]]


@dataclass
class Container:
    """Holds singletons shared across requests."""

    settings: Settings

    async def check_readiness(self) -> ReadinessReport:
        """Probe critical dependencies for the readiness endpoint."""
        return {}

    async def shutdown(self) -> None:
        """Release resources on application shutdown."""


async def build_container(settings: Settings) -> Container:
    """Wire up all providers from configuration."""
    return Container(settings=settings)
