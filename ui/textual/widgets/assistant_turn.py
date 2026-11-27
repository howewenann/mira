"""Presentation-only grouping of existing transcript widgets by outer turn ID."""

from __future__ import annotations

from collections import Counter, deque
from itertools import count
import time
from typing import Callable

from rich.table import Table
from rich.text import Text
from textual import on
from textual.containers import Horizontal, Vertical
from textual.events import Click
from textual.widget import Widget
from textual.widgets import Button, Label, ListItem, ListView, Static

from ui.shared.rubric import format_elapsed
from ui.textual.widgets.subagent_panel import SubagentSelected
from ui.textual.widgets.tool_bubble import ToolBubble, tool_argument_preview


# Existing bubble border/text colours from mira.tcss, shared by counts and rows.
CATEGORY_COLORS = {
    "Thinking": "#B8C2C9", "Tools": "#8FA3B8", "Messages": "#5BB8B1",
    "Subagents": "#B7A4E8", "Rubric": "#C58FD6", "Plans": "#A0B86B",
    "Goals": "#C58FD6", "Workflows": "#5BB8B1", "Activity": "#7D9BD1",
}


def activity_line(label: Text, duration: str, expanded: bool) -> Table:
    """Native Rich columns keep the glyph and duration independent of text width."""
    line = Table.grid(expand=True, padding=0)
    line.add_column(width=2, no_wrap=True)
    line.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
    line.add_column(width=max(6, len(duration)), justify="right", no_wrap=True)
    line.add_row("▾ " if expanded else "▸ ", label, Text(duration, style="dim"))
    return line


class TurnSummary(Static, can_focus=True):
    BINDINGS = [("enter,space", "toggle", "Toggle activity")]

    def __init__(self) -> None:
        super().__init__(classes="turn-summary")

    def on_click(self, event: Click) -> None:
        event.stop()
        self.focus(scroll_visible=False)
        self.action_toggle()

    def action_toggle(self) -> None:
        self.parent.toggle_history()


class ActivityRow(ListItem):
    """One native keyline row; its original detail widget stays detached."""

    def __init__(self, widget: Widget, category: str, sequence: int, *, attention: bool, historical: bool) -> None:
        self.glyph = Label("▸", classes="activity-glyph")
        self.label = Label(classes="activity-label")
        self.preview = Label(markup=False, classes="activity-preview")
        self.duration = Label(classes="activity-duration")
        super().__init__(Horizontal(self.glyph, self.label, self.preview, self.duration,
                                    classes="activity-content"), classes="activity-row")
        self.widget, self.category, self.sequence = widget, category, sequence
        self.attention = attention
        self.completed = False
        self.selected = False
        self.started_at = None if historical else time.monotonic()
        self.duration_ms = None

    def refresh_title(self) -> None:
        self.label.styles.color = self.glyph.styles.color = CATEGORY_COLORS[self.category]
        if isinstance(self.widget, ToolBubble):
            name = self.widget.tool_name
            self.preview.update(tool_argument_preview(name, self.widget.arguments))
        else:
            name = {"Thinking": "thinking", "Messages": "message", "Activity": "activity"}.get(
                self.category, self.category.rstrip("s").lower())
        self.label.update(name)
        duration = getattr(self.widget, "duration_ms", None)
        if duration is not None:
            self.duration_ms = duration
        duration = format_elapsed(self.duration_ms) if self.duration_ms is not None else ""
        self.duration.update(duration)
        self.duration.styles.width = max(5, len(duration))
        self.glyph.update("▾" if self.selected else "▸")
        self.set_class(self.selected, "selected")


class AssistantTurn(Vertical):
    """Compact history references existing widgets; only selected details mount.

    Active widgets mount directly, without a disclosure wrapper. Completion
    detaches them and mounts a lightweight row. Streaming keeps widget identity.
    """

    def __init__(self, turn_id: str, pause_chat: Callable[[], None]) -> None:
        super().__init__(classes="message assistant-turn")
        self.turn_id, self._pause_chat = turn_id, pause_chat
        self.border_title = "Assistant"
        self.summary = TurnSummary()
        self.history = ListView(initial_index=None, classes="completed-activity")
        self._history_opened = False
        self.details = Vertical(classes="activity-details")
        self.inspect_button = Button("Inspect", compact=True, classes="rubric-inspect")
        self.close_button = Button("Close", compact=True, classes="activity-close")
        self.detail_actions = Horizontal(self.inspect_button, self.close_button, classes="activity-detail-actions")
        self.active = Vertical(classes="turn-active")
        self.rows: dict[Widget, ActivityRow] = {}
        self._sequence = count()
        self.completed_rows: list[ActivityRow] = []
        self.completed_counts: Counter[str] = Counter()
        self.selected: ActivityRow | None = None
        self.expanded = False
        self.candidate: ActivityRow | None = None
        self._candidate_viable = False
        self.state = "running"
        self._elapsed = 0.0
        self._duration_since_recovery = False
        self._started = time.monotonic()
        self._moves: deque[ActivityRow] = deque()
        self._moving = self._composed = False
        self.tick()

    def compose(self):
        yield self.summary
        yield self.history
        yield self.details
        yield self.active

    def on_mount(self) -> None:
        self._composed = True
        for row in self.rows.values():
            self._queue_move(row)
        self.tick()

    def add(self, widget: Widget, category: str, *, attention: bool = False, historical: bool = False) -> None:
        if self.candidate is not None and category != "Rubric":
            self.complete(self.candidate.widget)
        row = ActivityRow(widget, category, next(self._sequence), attention=attention, historical=historical)
        self.rows[widget] = row
        if category == "Messages":
            self.candidate, self._candidate_viable = row, True
        elif category in {"Thinking", "Tools", "Plans", "Goals", "Workflows"}:
            self._candidate_viable = False
        self._queue_move(row)

    def complete(self, widget: Widget) -> None:
        row = self.rows.get(widget)
        if row is None or row.attention:
            return
        widget.remove_class("turn-final-response")
        if not row.completed:
            row.completed = True
            if row.started_at is not None:
                row.duration_ms = round((time.monotonic() - row.started_at) * 1000)
            self.completed_counts[row.category] += 1
            self.completed_rows.append(row)
            self.completed_rows.sort(key=lambda item: item.sequence)
        row.refresh_title()
        self.tick()
        self._queue_move(row)

    def resume(self, widget: Widget) -> None:
        row = self.rows.get(widget)
        if row is None or not row.completed:
            return
        if row.category == "Messages":
            if self.candidate is not None and self.candidate is not row:
                self.complete(self.candidate.widget)
            self.candidate, self._candidate_viable = row, True
        elif row.category in {"Thinking", "Tools", "Plans", "Goals", "Workflows"}:
            if self.candidate is not None:
                self.complete(self.candidate.widget)
            self._candidate_viable = False
        if self.selected is row:
            self.select_activity(row)
        row.completed = False
        self.completed_rows.remove(row)
        self.completed_counts[row.category] -= 1
        self.tick()
        self._queue_move(row)

    def pin_completed(self, widget: Widget) -> None:
        """Keep a completed failure visible as well as counted."""
        row = self.rows[widget]
        self.complete(widget)
        row.attention = True
        if self.selected is row:
            self.select_activity(row)
        self._queue_move(row)

    def forget(self, widget: Widget) -> None:
        row = self.rows.pop(widget, None)
        if row is None:
            return
        if self.selected is row:
            self.select_activity(row)
        if row.completed:
            self.completed_rows.remove(row)
            self.completed_counts[row.category] -= 1
        if self.candidate is row:
            self.candidate = None
        if row.parent is self.history:
            self.history.index = None
        row.remove()
        widget.remove()
        self.tick()

    def toggle_history(self) -> None:
        self._pause_chat()
        self.expanded = not self.expanded
        if self.expanded and not self._history_opened:
            self._history_opened = True
            for row in self.completed_rows:
                self._queue_move(row)
        if not self.expanded and self.selected is not None:
            self.select_activity(self.selected)
        self.tick()

    def select_activity(self, row: ActivityRow) -> None:
        self._pause_chat()
        previous = self.selected
        self.selected = None if previous is row else row
        for changed in (previous, self.selected):
            if changed is not None:
                changed.selected = changed is self.selected
                changed.refresh_title()
                self._queue_move(changed)
        self.details.display = self.selected is not None
        self.inspect_button.display = bool(self.selected and getattr(self.selected.widget, "inspection_id", ""))

    def on_click(self, event: Click) -> None:
        # ChatLog focuses itself on ordinary transcript clicks. Native ListView
        # owns focus and keyboard navigation within the activity list instead.
        if event.widget is self.history or self.history in event.widget.ancestors:
            event.stop()

    @on(ListView.Selected)
    def activity_selected(self, event: ListView.Selected) -> None:
        if event.list_view is self.history:
            event.stop()
            self.select_activity(event.item)

    @on(ListView.Highlighted)
    def reveal_keyboard_cursor(self, event: ListView.Highlighted) -> None:
        if event.list_view is self.history and event.item is not None and self.history.has_focus:
            event.item.scroll_visible(animate=False, immediate=True)

    @on(Button.Pressed)
    def detail_action(self, event: Button.Pressed) -> None:
        if event.button is self.close_button and self.selected is not None:
            event.stop()
            self.history.focus(scroll_visible=False)
            self.select_activity(self.selected)
        elif event.button is self.inspect_button and self.selected is not None:
            event.stop()
            self.post_message(SubagentSelected(self.selected.widget.inspection_id))

    def _queue_move(self, row: ActivityRow) -> None:
        if self._composed and row not in self._moves:
            self._moves.append(row)
            if not self._moving:
                self._moving = True
                self.call_later(self._move_rows)

    async def _move_rows(self) -> None:
        try:
            with self.app.batch_update():
                while self._moves and self.is_attached:
                    row = self._moves.popleft()
                    if row.widget not in self.rows:
                        continue
                    archived = row.completed and not row.attention
                    # Keep native cursor identity when parallel rows settle out
                    # of order, without an arrival scrolling the conversation.
                    with self.history.prevent(ListView.Highlighted):
                        highlighted = self.history.highlighted_child
                        if archived and self._history_opened and row.parent is None:
                            before = next((child for child in self.history.children if child.sequence > row.sequence), None)
                            await self.history.mount(row, before=before)
                        elif not archived and row.parent is self.history:
                            await self.history.pop(list(self.history.children).index(row))
                        if highlighted in self.history.children:
                            self.history.index = list(self.history.children).index(highlighted)
                    destination = self.details if archived and row is self.selected else None if archived else self.active
                    widget = row.widget
                    if widget.parent is not destination:
                        if widget.parent is not None:
                            await widget.remove()
                        # Updates can arrive while native removal yields.
                        if widget not in self.rows:
                            continue
                        archived = row.completed and not row.attention
                        destination = self.details if archived and row is self.selected else None if archived else self.active
                        if destination is not None:
                            if destination is self.details and self.detail_actions.parent is None:
                                await self.details.mount(self.detail_actions)
                            if isinstance(widget, ToolBubble):
                                widget.set_history_detail(destination is self.details)
                            await destination.mount(widget)
        finally:
            self._moving = False

    def start(self) -> None:
        if self.candidate is not None:
            self.candidate.widget.remove_class("turn-final-response")
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
        final = self.candidate if self._candidate_viable else None
        if final is not None:
            self.resume(final.widget)
            final.widget.add_class("turn-final-response")
        for widget, row in tuple(self.rows.items()):
            if row is not final:
                self.complete(widget)
        if historical:
            self._elapsed = -1
        self.tick()

    def tick(self) -> None:
        running = self.state in {"running", "waiting", "cancelling"}
        duration = self._elapsed + time.monotonic() - self._started if running else self._elapsed
        label = {"running": "Working", "waiting": "Waiting for you", "cancelling": "Cancelling",
                 "completed": "Worked" if duration >= 0 else "Completed"}.get(self.state, self.state.capitalize())
        steps = sum(self.completed_counts.values())
        if self._duration_since_recovery:
            label += " since recovery"
        header = activity_line(Text(f"{label} · {steps} {'step' if steps == 1 else 'steps'}"),
                               format_elapsed(round(duration * 1000)) if duration >= 0 else "", self.expanded)
        counts = Text()
        for category, color in CATEGORY_COLORS.items():
            count = self.completed_counts[category]
            if count:
                if counts.plain:
                    counts.append(" · ", style="#b8c1c7")
                name = category[:-1] if count == 1 and category.endswith("s") else category
                if category == "Activity" and count != 1:
                    name = "Activities"
                counts.append(f"{count} {name}", style=color)
        header.add_row("", counts, "")
        self.summary.update(header, layout=False)
        self.history.display = self.expanded
        self.details.display = self.expanded and self.selected is not None


__all__ = ["AssistantTurn", "ActivityRow"]
