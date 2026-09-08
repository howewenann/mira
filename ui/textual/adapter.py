"""Textual consumer of the supported MIRA API."""

from typing import Any

from mira.api import FrontendRequest
from ui.shared.adapter import RendererAdapter


class TextualFrontend(RendererAdapter):
    """Connect typed Core events and requests to the Textual application."""

    async def request(self, request: FrontendRequest) -> Any:
        """Give every Core request the app's serialized foreground ownership."""
        parent_request = super().request
        return await self.renderer.foreground_interrupt(
            lambda: parent_request(request)
        )


__all__ = ["TextualFrontend"]
