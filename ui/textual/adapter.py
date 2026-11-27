"""Textual consumer of the supported MIRA API."""

from typing import Any

from contextlib import suppress

from textual.css.query import NoMatches

from mira.api import FrontendEvent, FrontendRequest, MessageEvent, RuntimeEvent, SubagentEvent, ToolEvent
from ui.shared.adapter import RendererAdapter
from ui.textual.widgets.chat_log import ChatLog


class TextualFrontend(RendererAdapter):
    """Connect typed Core events and requests to the Textual application."""

    def _chat_log(self) -> ChatLog | None:
        if self.renderer.is_mounted and self.renderer.screen_stack:
            with suppress(NoMatches):
                return self.renderer.screen_stack[0].query_one("#chat-log", ChatLog)
        return None

    def emit(self, event: FrontendEvent) -> None:
        selected = self.renderer.core_session
        if event.session_id and selected is not None and event.session_id != selected.id:
            return
        if isinstance(event, RuntimeEvent) and event.kind == "inspection":
            self.renderer.inspection_updated(event)
            return
        if isinstance(event, RuntimeEvent) and event.kind == "configuration":
            if self.renderer.application is not None:
                self.renderer._refresh_runtime_view()
            return
        chat = self._chat_log()
        if chat is not None:
            chat.begin_frontend_event(event)
        try:
            super().emit(event)
            if chat is not None and isinstance(event, SubagentEvent):
                chat.subagent_event(event)
        finally:
            if chat is not None:
                chat.end_frontend_event(event)

    def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Apply presentation and its provenance together in one projection."""
        result = super()._call(method, *args, **kwargs)
        event = kwargs.get("frontend_event")
        if isinstance(event, (MessageEvent, ToolEvent)) and self.renderer.is_mounted:
            if chat := self._chat_log():
                chat.retain_frontend_event(event)
        return result

    async def request(self, request: FrontendRequest) -> Any:
        """Give every Core request the app's serialized foreground ownership."""
        parent_request = super().request
        chat = self._chat_log()
        turn = chat._turns.get(chat._active_turn_id) if chat is not None else None
        if turn is not None:
            turn.state = "waiting"
            turn.tick()
        try:
            return await self.renderer.foreground_interrupt(
                lambda: parent_request(request)
            )
        finally:
            if turn is not None and turn.state == "waiting":
                turn.state = "running"
                turn.tick()


__all__ = ["TextualFrontend"]
