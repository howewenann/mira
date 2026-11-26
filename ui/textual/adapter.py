"""Textual consumer of the supported MIRA API."""

from typing import Any

from mira.api import FrontendEvent, FrontendRequest
from ui.shared.adapter import RendererAdapter


class TextualFrontend(RendererAdapter):
    """Connect typed Core events and requests to the Textual application."""

    def emit(self, event: FrontendEvent) -> None:
        super().emit(event)
        # Keep the complete observation available to the widgets; callback
        # projection remains shared with the terminal and Workflow observers.
        received = getattr(self.renderer, "frontend_event_received", None)
        if callable(received):
            received(event)

    async def request(self, request: FrontendRequest) -> Any:
        """Give every Core request the app's serialized foreground ownership."""
        parent_request = super().request
        return await self.renderer.foreground_interrupt(
            lambda: parent_request(request)
        )


__all__ = ["TextualFrontend"]
