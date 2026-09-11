"""Read-only viewport for process-local live inspection transcripts."""

from __future__ import annotations

from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Static

from core.execution.inspection.live import (
    InspectionEvent,
    InspectionUpdate,
    LiveInspectionStore,
)
from ui.textual.widgets.chat_log import ChatLog, DEFAULT_TOOL_OUTPUT_CHARS


class Inspector(Vertical):
    """Generic live inspection surface backed by a LiveInspectionStore."""

    can_focus = False

    class Closed(Message):
        """Request restoration of the normal chat viewport."""

    def __init__(
        self,
        store: LiveInspectionStore,
        *,
        tool_output_chars: int = DEFAULT_TOOL_OUTPUT_CHARS,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.store = store
        self.tool_output_chars = tool_output_chars
        self.inspection_id = ""

    def compose(self) -> ComposeResult:
        with Horizontal(id="inspector-header"):
            yield Static("Inspector", id="inspector-title")
            yield Button("x", id="inspector-close", compact=True)
        yield ChatLog(
            tool_output_chars=self.tool_output_chars,
            id="inspector-log",
            classes="inspection-transcript",
        )

    def open(self, inspection_id: str) -> bool:
        """Replay captured state and subscribe for subsequent live updates."""
        inspection = self.store.get(inspection_id)
        if inspection is None:
            return False
        self._unsubscribe()
        self.inspection_id = inspection_id
        self.query_one("#inspector-title", Static).update(
            f"Inspector  ·  {inspection.title}"
        )
        self._replay()
        self.store.subscribe(inspection_id, self._inspection_updated)
        return True

    def stop_inspection(self) -> None:
        """Detach from the current transcript without changing stored state."""
        self._unsubscribe()
        self.inspection_id = ""

    def on_unmount(self) -> None:
        self._unsubscribe()

    @on(Button.Pressed, "#inspector-close")
    def close_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(self.Closed())

    def _unsubscribe(self) -> None:
        if self.inspection_id:
            self.store.unsubscribe(self.inspection_id, self._inspection_updated)

    def _inspection_updated(
        self,
        inspection_id: str,
        update: InspectionUpdate,
    ) -> None:
        if inspection_id != self.inspection_id:
            return
        if update.operation == "reset":
            inspection = self.store.get(inspection_id)
            if inspection is not None:
                self.query_one("#inspector-title", Static).update(
                    f"Inspector  ·  {inspection.title}"
                )
            self._replay()
        elif update.event is not None:
            self._render_event(update.event)

    def _replay(self) -> None:
        log = self.query_one("#inspector-log", ChatLog)
        log.clear_log()
        inspection = self.store.get(self.inspection_id)
        if inspection is None:
            return
        for event in inspection.events:
            self._render_event(event)

    def _render_event(self, event: InspectionEvent) -> None:
        log = self.query_one("#inspector-log", ChatLog)
        if event.kind == "user":
            log.user_message(event.text)
        elif event.kind == "reasoning":
            log.reasoning_delta(event.text)
        elif event.kind == "assistant":
            log.text_delta(event.text)
        elif event.kind == "tool_call":
            log.tool_call(event.name or "tool", event.args, call_id=event.call_id)
        elif event.kind == "tool_result":
            log.tool_result(event.name or "tool", event.text, call_id=event.call_id)
        elif event.kind == "tool_error":
            log.tool_error(event.name or "tool", event.text, call_id=event.call_id)
        elif event.kind == "error":
            log.system_message(event.text, kind="error")


__all__ = ["Inspector"]
