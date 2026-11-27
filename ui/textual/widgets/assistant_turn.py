"""Presentation-only grouping of existing transcript widgets by outer turn ID."""

from __future__ import annotations

from collections import Counter, deque
import time
from itertools import count
from typing import Callable

from rich.text import Text
from textual import on
from textual.containers import Vertical, VerticalScroll
from textual.widget import Widget
from textual.widgets import Button, Collapsible, Static

from ui.shared.rubric import format_elapsed
from ui.textual.widgets.subagent_panel import SubagentSelected


CATEGORY_COLORS = {
    "Thinking": "#B8C2C9", "Tools": "#8FA3B8", "Messages": "#5BB8B1",
    "Subagents": "#B7A4E8", "Rubric": "#C58FD6", "Plans": "#A0B86B",
    "Goals": "#C58FD6", "Workflows": "#5BB8B1", "Activity": "#7D9BD1",
}


class ActivityDisclosure(Collapsible):
    """Native disclosure whose updates do not scroll the surrounding chat."""

    def scroll_visible(self, *args, **kwargs):
        # Collapsible's reactive watcher otherwise scrolls on every programmatic
        # collapse as well as user expansion. The history owns its own viewport.
        return None


class ActivityViewport(VerticalScroll):
    """A bounded history that follows new rows only while already at its tail."""

    def __init__(self, pause_chat: Callable[[], None]) -> None:
        super().__init__(classes="activity-viewport")
        self.follow_tail = True
        self._scrolling = False
        self._pause_chat = pause_chat

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        super().watch_scroll_y(old_value, new_value)
        if not self._scrolling and self.is_mounted and self.size.height:
            self.follow_tail = self.is_vertical_scroll_end
            if not self.follow_tail:
                self._pause_chat()

    def watch_scroll_target_y(self, old_value: float, new_value: float) -> None:
        if not self._scrolling and new_value < self.max_scroll_y:
            self.follow_tail = False
            self._pause_chat()

    def follow(self) -> None:
        if self.follow_tail:
            self.call_after_refresh(self._tail)

    def _tail(self) -> None:
        if self.follow_tail:
            self._scrolling = True
            try:
                self.scroll_end(animate=False, force=True, immediate=True)
            finally:
                self._scrolling = False

    def capture_anchor(self):
        if self.follow_tail or not self.is_mounted:
            return None
        top = self.content_region.y
        row = next((child for child in self.children if child.region.bottom > top), None)
        return (row, row.region.y, self.scroll_y) if row is not None else None

    def restore_anchor(self, anchor) -> None:
        if anchor is None or self.follow_tail:
            return
        row, screen_y, scroll_y = anchor
        if row.is_attached and self.scroll_y == scroll_y:
            self._scrolling = True
            try:
                self.scroll_to(y=scroll_y + row.region.y - screen_y, animate=False,
                               force=True, immediate=True)
            finally:
                self._scrolling = False


class ActivityInspectButton(Button):
    can_focus = False

    def __init__(self, inspection_id: str) -> None:
        super().__init__("Inspect", compact=True, classes="rubric-inspect")
        self.inspection_id = inspection_id
        self.display = True

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(SubagentSelected(self.inspection_id))


class ActivityRow(ActivityDisclosure):
    """One existing activity widget, expanded live and independently inspectable."""

    def __init__(self, widget: Widget, category: str, sequence: int, *, attention: bool) -> None:
        children = [widget]
        if category == "Subagents" and (inspection_id := getattr(widget, "inspection_id", "")):
            children.append(ActivityInspectButton(inspection_id))
        super().__init__(*children, title=str(widget.border_title or category),
                         collapsed=False, classes="activity-row active-activity")
        self.widget = widget
        self.category = category
        self.sequence = sequence
        self.attention = attention
        self.completed = False

    def refresh_title(self) -> None:
        title = Text.from_markup(str(self.widget.border_title or self.category),
                     style=CATEGORY_COLORS[self.category])
        status = getattr(self.widget, "status", None)
        if isinstance(status, Static) and isinstance(status.content, Text) and status.content.plain:
            title.append(" · ")
            title.append_text(status.content)
        self.title = title


class AssistantTurn(Vertical):
    """A turn header, completed history and the existing active/final widgets.

    Rows reference widgets, not a copied execution history. A widget crosses
    containers only when its lifecycle changes, using native remove/mount.
    Streaming continues to update the same widget and never rebuilds the turn.
    """

    def __init__(self, turn_id: str, pause_chat: Callable[[], None]) -> None:
        super().__init__(classes="message assistant-turn")
        self.turn_id = turn_id
        self.border_title = "Assistant"
        self.summary = Static(classes="turn-summary")
        self.counts = Static(classes="turn-counts")
        self.viewport = ActivityViewport(pause_chat)
        self.history = ActivityDisclosure(self.viewport, title="Completed activity",
                                          collapsed=True, classes="completed-activity")
        self.active = Vertical(classes="turn-active")
        self.rows: dict[Widget, ActivityRow] = {}
        self._sequence = count()
        self.completed_rows: list[ActivityRow] = []
        self.completed_counts: Counter[str] = Counter()
        self.candidate: ActivityRow | None = None
        self._candidate_viable = False
        self.state = "running"
        self._elapsed = 0.0
        self._duration_since_recovery = False
        self._started = time.monotonic()
        self._moves: deque[ActivityRow] = deque()
        self._moving = False
        self._composed = False
        self.tick()
        self._refresh_counts()

    def compose(self):
        yield self.summary
        yield self.counts
        yield self.history
        yield self.active

    def on_mount(self) -> None:
        self.call_after_refresh(self._mount_initial_rows)

    def _mount_initial_rows(self) -> None:
        if not self.is_attached:
            return
        # Historical replay can finish before composition. Mount those rows
        # directly in history, without a redundant active-to-history transfer.
        with self.app.batch_update():
            self._composed = True
            self.viewport.mount_all(row for row in self.rows.values() if row.completed and not row.attention)
            self.active.mount_all(row for row in self.rows.values() if not row.completed or row.attention)
        self.viewport.follow()

    def add(self, widget: Widget, category: str, *, attention: bool = False) -> None:
        # A later execution phase proves the previous assistant message was
        # intermediate. A model-message end alone does not prove that.
        if self.candidate is not None:
            self.complete(self.candidate.widget)
        row = ActivityRow(widget, category, next(self._sequence), attention=attention)
        self.rows[widget] = row
        if self._composed:
            self.active.mount(row)
        if category == "Messages":
            self.candidate = row
            self._candidate_viable = True
        elif category in {"Thinking", "Tools", "Plans", "Goals", "Workflows"}:
            self._candidate_viable = False

    def complete(self, widget: Widget) -> None:
        row = self.rows.get(widget)
        if row is None or row.attention:
            return
        if row.completed:
            row.refresh_title()
            if row.parent is self.active:
                row.remove_class("active-activity")
                row.collapsed = True
                self._queue_move(row)
            return
        row.completed = True
        row.refresh_title()
        row.collapsed = True
        row.remove_class("active-activity")
        self.completed_counts[row.category] += 1
        self.completed_rows.append(row)
        self.completed_rows.sort(key=lambda item: item.sequence)
        self._refresh_counts()
        self._queue_move(row)

    def resume(self, widget: Widget) -> None:
        row = self.rows.get(widget)
        if row is None or not row.completed:
            return
        if row.category == "Messages":
            if self.candidate is not None and self.candidate is not row:
                self.complete(self.candidate.widget)
            self.candidate = row
            self._candidate_viable = True
        elif row.category in {"Thinking", "Tools", "Plans", "Goals", "Workflows"}:
            if self.candidate is not None:
                self.complete(self.candidate.widget)
            self._candidate_viable = False
        row.completed = False
        self.completed_rows.remove(row)
        self.completed_counts[row.category] -= 1
        row.add_class("active-activity")
        row.collapsed = False
        self._refresh_counts()
        self._queue_move(row)

    def pin_completed(self, widget: Widget) -> None:
        """Keep a completed failure visible as well as counted."""
        row = self.rows[widget]
        self.complete(widget)
        row.attention = True
        row.collapsed = False
        row.add_class("active-activity")
        self._refresh_counts()
        self._queue_move(row)

    def forget(self, widget: Widget) -> None:
        row = self.rows.pop(widget, None)
        if row is None:
            return
        if row.completed:
            self.completed_rows.remove(row)
            self.completed_counts[row.category] -= 1
            self._refresh_counts()
        if self.candidate is row:
            self.candidate = None
        row.remove()

    def _queue_move(self, row: ActivityRow) -> None:
        if not self._composed:
            return
        if row not in self._moves:
            self._moves.append(row)
        if not self._moving:
            self._moving = True
            self.call_later(self._move_rows)

    async def _move_rows(self) -> None:
        try:
            while self._moves and self.is_attached:
                row = self._moves.popleft()
                if row.widget not in self.rows:
                    continue
                destination = self.viewport if row.completed and not row.attention else self.active
                if row.parent is destination:
                    continue
                focused = self.screen.focused
                restore_focus = focused is not None and row in focused.ancestors
                anchor = self.viewport.capture_anchor()
                with self.app.batch_update():
                    await row.remove()
                    if not self.is_attached or row.widget not in self.rows:
                        continue
                    # Recovery may have changed the state while removal yielded.
                    destination = self.viewport if row.completed and not row.attention else self.active
                    before = next((child for child in destination.children
                                   if isinstance(child, ActivityRow) and child.sequence > row.sequence), None)
                    await destination.mount(row, before=before)
                if restore_focus and focused.is_mounted and not row.collapsed:
                    focused.focus(scroll_visible=False)
                self.viewport.follow()
                self.viewport.call_after_refresh(self.viewport.restore_anchor, anchor)
        finally:
            self._moving = False

    def _refresh_counts(self) -> None:
        value = Text()
        for category, color in CATEGORY_COLORS.items():
            count = self.completed_counts[category]
            if count:
                if value.plain:
                    value.append(" · ", style="#b8c1c7")
                label = category[:-1] if count == 1 and category.endswith("s") else category
                if category == "Activity" and count != 1:
                    label = "Activities"
                value.append(f"{count} {label}", style=color)
        self.counts.update(value)
        self.counts.display = bool(value.plain)
        self.history.display = any(not row.attention for row in self.completed_rows)

    def start(self) -> None:
        if self._elapsed < 0:
            self._elapsed = 0.0
            self._duration_since_recovery = True
        if self.state != "running":
            self._started = time.monotonic()
        self.state = "running"
        self.tick()

    def finish(self, state: str = "completed", *, historical: bool = False) -> None:
        if self.state in {"running", "waiting", "cancelling"}:
            self._elapsed += time.monotonic() - self._started
        self.state = state
        if state in {"failed", "interrupted", "cancelled"}:
            for widget in self.rows:
                if widget.has_class("info"):
                    self.pin_completed(widget)
        # The last assistant message remains outside history. Artifacts and
        # warnings retain attention independently of the outer operation.
        final = self.candidate if self._candidate_viable else None
        if final is not None:
            self.resume(final.widget)
        for widget, row in tuple(self.rows.items()):
            if row is not final:
                self.complete(widget)
        if historical:
            self._elapsed = -1  # Saved records have no authoritative turn duration.
        self.tick()

    def tick(self) -> None:
        if self.state in {"running", "waiting", "cancelling"}:
            duration = self._elapsed + time.monotonic() - self._started
            label = {"running": "Working", "waiting": "Waiting for you", "cancelling": "Cancelling"}[self.state]
            value = f"{label} · {format_elapsed(round(duration * 1000))}"
        elif self._elapsed < 0:
            value = self.state.capitalize()
        else:
            duration = format_elapsed(round(self._elapsed * 1000))
            value = f"Worked for {duration}" if self.state == "completed" else f"{self.state.capitalize()} after {duration}"
        if self._duration_since_recovery:
            value += " since recovery"
        self.summary.update(value)

    @on(Collapsible.Expanded)
    def disclosure_expanded(self, event: Collapsible.Expanded) -> None:
        if event.collapsible is self.history:
            self.viewport.follow()


__all__ = ["AssistantTurn", "ActivityRow", "ActivityViewport"]
