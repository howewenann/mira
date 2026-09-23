"""Read-only viewport for process-local live inspection transcripts."""

from __future__ import annotations

from typing import Any

from rich.pretty import Pretty, pretty_repr
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.events import Click
from textual.message import Message
from textual.widgets import Button, Static

from core.execution.inspection.live import (
    InspectionEvent,
    InspectionUpdate,
    LiveInspection,
    LiveInspectionStore,
)
from ui.textual.widgets.chat_log import ChatLog, DEFAULT_TOOL_OUTPUT_CHARS


class Inspector(Vertical):
    """Generic live inspection surface backed by a LiveInspectionStore."""

    can_focus = False
    FEEDBACK_SECONDS = 1.5

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
        self._snapshot: LiveInspection | None = None
        self._final_state: Any = None
        self._final_state_text = ""
        self._copy_feedback_version = 0
        self._copy_button = Button(
            "Copy",
            id="inspector-final-state-copy",
            compact=True,
        )
        self._final_state_actions = Horizontal(
            self._copy_button,
            id="inspector-final-state-actions",
        )
        self._final_state_actions.display = False

    def compose(self) -> ComposeResult:
        with Horizontal(id="inspector-header"):
            yield Static("Inspector", id="inspector-title", markup=False)
            yield Button("x", id="inspector-close", compact=True)
        yield ChatLog(
            tool_output_chars=self.tool_output_chars,
            id="inspector-log",
            classes="inspection-transcript",
        )
        yield self._final_state_actions

    def open(self, inspection_id: str) -> bool:
        """Replay captured state and subscribe for subsequent live updates."""
        inspection = self.store.get(inspection_id)
        if inspection is None:
            return False
        self._unsubscribe()
        self._snapshot = None
        self._clear_final_state()
        self.inspection_id = inspection_id
        self._update_header(inspection)
        self._replay()
        self.store.subscribe(inspection_id, self._inspection_updated)
        return True

    def open_snapshot(self, inspection: LiveInspection) -> bool:
        """Open one immutable historical transcript without subscribing."""
        if not inspection.id:
            return False
        self._unsubscribe()
        self.inspection_id = inspection.id
        self._snapshot = inspection
        self._clear_final_state()
        self._update_header(inspection)
        self._replay()
        return True

    def open_workflow_state(self, name: str, final_state: Any) -> None:
        """Show one retained Python object in the existing Inspector viewport."""
        self._unsubscribe()
        self.inspection_id = ""
        self._snapshot = None
        self._final_state = final_state
        self._final_state_text = pretty_repr(final_state, expand_all=True)
        self.query_one("#inspector-title", Static).update(
            f"Workflow · {name} · Final state"
        )
        log = self.query_one("#inspector-log", ChatLog)
        log.clear_log()
        log.command_output(Pretty(final_state, expand_all=False))
        self._copy_button.label = "Copy"
        self._final_state_actions.display = True

    def stop_inspection(self) -> None:
        """Detach from the current transcript without changing stored state."""
        self._unsubscribe()
        self.inspection_id = ""
        self._snapshot = None
        self._clear_final_state()

    def on_unmount(self) -> None:
        self._unsubscribe()

    def on_click(self, event: Click) -> None:
        """Give Escape ownership to the Inspector after a non-button click."""
        if not isinstance(event.widget, Button):
            self.query_one("#inspector-log", ChatLog).focus()

    @on(Button.Pressed, "#inspector-close")
    def close_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(self.Closed())

    @on(Button.Pressed, "#inspector-final-state-copy")
    def copy_final_state(self, event: Button.Pressed) -> None:
        event.stop()
        self.app.copy_to_clipboard(self._final_state_text)
        self._copy_feedback_version += 1
        version = self._copy_feedback_version
        self._copy_button.label = "Copied"
        self.set_timer(
            self.FEEDBACK_SECONDS,
            lambda: self._restore_copy_label(version),
        )

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
                self._update_header(inspection)
            self._replay()
        elif update.event is not None:
            self._render_event(update.event)

    def _replay(self) -> None:
        log = self.query_one("#inspector-log", ChatLog)
        log.clear_log()
        inspection = self._snapshot or self.store.get(self.inspection_id)
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

    def _update_header(self, inspection: LiveInspection) -> None:
        self.query_one("#inspector-title", Static).update(
            f"Inspector · {inspection.inspection_type} · {inspection.title}"
        )

    def _clear_final_state(self) -> None:
        self._final_state = None
        self._final_state_text = ""
        self._copy_feedback_version += 1
        if self._final_state_actions.is_mounted:
            self._copy_button.label = "Copy"
            self._final_state_actions.display = False

    def _restore_copy_label(self, version: int) -> None:
        if self._copy_feedback_version == version and self._copy_button.is_mounted:
            self._copy_button.label = "Copy"


__all__ = ["Inspector"]
