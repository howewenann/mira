"""Textual consumer of the supported MIRA API."""

from typing import Any

from contextlib import suppress

from textual.css.query import NoMatches

from mira.api import FrontendEvent, FrontendRequest, MessageEvent, RuntimeEvent, ToolEvent
from ui.shared.adapter import RendererAdapter
from ui.textual.widgets.chat_log import ChatLog


class TextualFrontend(RendererAdapter):
    """Connect typed Core events and requests to the Textual application."""

    def emit(self, event: FrontendEvent) -> None:
        if isinstance(event, RuntimeEvent) and event.kind == "inspection":
            self.renderer.inspection_updated(event)
            return
        if isinstance(event, RuntimeEvent) and event.kind == "configuration":
            if self.renderer.application is not None:
                self.renderer._refresh_runtime_view()
            return
        super().emit(event)

    def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Apply presentation and its provenance together in one projection."""
        result = super()._call(method, *args, **kwargs)
        event = kwargs.get("frontend_event")
        if isinstance(event, (MessageEvent, ToolEvent)) and self.renderer.is_mounted:
            with suppress(NoMatches):
                self.renderer.query_one(ChatLog).retain_frontend_event(event)
        return result

    async def request(self, request: FrontendRequest) -> Any:
        """Give every Core request the app's serialized foreground ownership."""
        parent_request = super().request
        return await self.renderer.foreground_interrupt(
            lambda: parent_request(request)
        )


__all__ = ["TextualFrontend"]
