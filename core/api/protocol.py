"""The deliberately small in-process contract between MIRA Core and consumers."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from core.api.events import FrontendEvent
from core.api.requests import FrontendRequest


@runtime_checkable
class Frontend(Protocol):
    """Consume Core events and answer interactions that pause execution."""

    def emit(self, event: FrontendEvent) -> None:
        """Consume one ordered notification."""

    async def request(self, request: FrontendRequest) -> Any:
        """Return the frontend response to one blocking interaction."""


class NullFrontend:
    """Headless sink useful for non-interactive lifecycle operations."""

    def emit(self, event: FrontendEvent) -> None:
        del event

    async def request(self, request: FrontendRequest) -> Any:
        raise RuntimeError(f"frontend interaction is unavailable: {type(request).__name__}")


__all__ = ["Frontend", "NullFrontend"]
