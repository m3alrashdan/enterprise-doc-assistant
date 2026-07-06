"""FastAPI dependencies. All state flows from app.state, never from globals."""

from __future__ import annotations

from fastapi import Request

from app.core.config import Settings
from app.core.container import Container


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_container(request: Request) -> Container:
    return request.app.state.container
