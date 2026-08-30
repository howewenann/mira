"""Status bar for the MIRA TUI."""

from __future__ import annotations

from typing import Any

from rich.table import Table
from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.message import Message
from textual.containers import Horizontal
from textual.css.query import NoMatches
from textual.widgets import Button, Static

from ui.spinners import SPINNER_FRAMES
from ui.terminal_colors import (
    TOOL_CANCELLED_COLOR,
    TOOL_COMPLETED_COLOR,
    TOOL_FAILED_COLOR,
    TOOL_PREPARING_COLOR,
    TOOL_RUNNING_COLOR,
)

STATUS_STARTING_COLOR = TOOL_PREPARING_COLOR
ANIMATED_STATES = {"starting", "running", "cancelling"}


class ContextStatus(Static):
    """Mouse-only context status control with no keyboard focus behavior."""

    can_focus = False

    class Pressed(Message):
        """Posted when the idle context group is clicked."""

    def on_click(self, event: events.Click) -> None:
        if self.disabled:
            return
        event.stop()
        self.post_message(self.Pressed())


class StatusBar(Horizontal):
    """Top operational session and activity status."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._mode = "ACT"
        self._state = "starting"
        self._dashboard: dict[str, Any] = {}
        self._spinner_index = 0
        self._busy = False
        self._combined_renderable = Text()

    @property
    def renderable(self) -> Text:
        """Return the combined header projection for tests and inspection."""
        return self._combined_renderable

    def compose(self) -> ComposeResult:
        yield Static(id="status-primary")
        yield ContextStatus(id="context-status")

    def set_state(
        self,
        *,
        mode: str,
        model_name: str,
        state: str,
        dashboard: dict[str, Any] | None = None,
        turns: int = 0,
        busy: bool = False,
    ) -> None:
        """Update the status bar text."""
        normalized_state = str(state or "").strip().lower()
        if normalized_state != self._state:
            self._spinner_index = 0
        self._mode = mode
        self._state = normalized_state
        self._dashboard = dashboard if isinstance(dashboard, dict) else {}
        self._busy = bool(busy)
        self._render_status()

    def tick(self) -> None:
        """Advance the fixed header badge while its state is transitional."""
        if self._state not in ANIMATED_STATES:
            return
        self._spinner_index = (self._spinner_index + 1) % len(SPINNER_FRAMES)
        self._render_status()

    def _render_status(self) -> None:
        """Render the current operational state without refreshing telemetry."""
        primary = Text()
        append_part(primary, "MIRA", "bold #d6fff6")
        append_part(primary, self._mode)
        append_operational_state(primary, self._state, self._spinner_index)
        primary.append(" | ", style="#6f8389")
        context = Text()
        append_context(context, self._dashboard.get("context"))
        self._combined_renderable = primary.copy()
        self._combined_renderable.append_text(context)
        try:
            self.query_one("#status-primary", Static).update(primary)
            context_status = self.query_one("#context-status", ContextStatus)
        except NoMatches:
            return
        context_status.update(context)
        context_status.disabled = self._busy


class TelemetryBar(Horizontal):
    """Bottom model shortcut and compact session telemetry."""

    def compose(self) -> ComposeResult:
        yield Button("model: unset", id="model-settings-button")
        yield Static(id="telemetry-values")

    def set_state(
        self,
        *,
        model_name: str,
        dashboard: dict[str, Any] | None = None,
        turns: int = 0,
    ) -> None:
        button = self.query_one("#model-settings-button", Button)
        button.label = Text(f"model: {model_name or 'unset'}")
        button.refresh(layout=True)
        self.query_one("#telemetry-values", Static).update(telemetry_values(dashboard, turns))


def telemetry_values(dashboard: dict[str, Any] | None, turns: int) -> Text:
    dashboard = dashboard or {}
    text = Text()
    append_part(text, token_part(dashboard.get("tokens") if isinstance(dashboard, dict) else {}))
    append_part(text, f"Turns {max(0, int(turns or 0))}")
    append_part(text, duration_text(dashboard.get("duration_seconds", 0)))
    return text


def telemetry_row(model_name: str, dashboard: dict[str, Any] | None, turns: int) -> Table:
    """Build the legacy renderable projection without passive model text."""
    row = Table.grid(expand=True, padding=0)
    row.add_column(justify="right", no_wrap=True)
    row.add_row(telemetry_values(dashboard, turns))
    return row


def append_part(text: Text, value: str, style: str = "#d7dee2") -> None:
    """Append one pipe-separated status part."""
    if len(text):
        text.append(" | ", style="#6f8389")
    text.append(str(value), style=style)


def append_operational_state(text: Text, state: str, spinner_index: int = 0) -> None:
    """Append one prominent state badge using MIRA's lifecycle palette."""
    normalized = str(state or "").strip().lower()
    if normalized == "starting":
        symbol = SPINNER_FRAMES[spinner_index % len(SPINNER_FRAMES)]
        style = f"bold {STATUS_STARTING_COLOR}"
    elif normalized == "running":
        symbol = SPINNER_FRAMES[spinner_index % len(SPINNER_FRAMES)]
        style = f"bold {TOOL_RUNNING_COLOR}"
    elif normalized == "cancelling":
        symbol = SPINNER_FRAMES[spinner_index % len(SPINNER_FRAMES)]
        style = f"bold {TOOL_CANCELLED_COLOR}"
    elif normalized == "ready":
        symbol = "●"
        style = f"bold {TOOL_COMPLETED_COLOR}"
    elif normalized == "error":
        symbol = "×"
        style = f"bold {TOOL_FAILED_COLOR}"
    else:
        symbol = "•"
        style = "bold #d7dee2"
    append_part(text, f"{symbol} {normalized.upper() or 'UNKNOWN'}", style)


def append_context(text: Text, context: Any) -> None:
    """Append context usage with a colored bar."""
    context = context if isinstance(context, dict) else {}
    used = positive_int(context.get("used_tokens"))
    limit = positive_int(context.get("limit_tokens"))
    percent = percent_value(context.get("percent"), used, limit)
    style = context_style(percent)

    if len(text):
        text.append(" | ", style="#6f8389")
    text.append("Ctx ", style="#9fb0b6")
    if not used:
        pending_style = "bold #7D9BD1"
        text.append(context_bar(0), style=pending_style)
        text.append(" pending ", style=pending_style)
        text.append(f"(?/{compact_count(limit) if limit else '?'})", style="#b8c3c7")
        return

    text.append(context_bar(percent), style=style)
    text.append(f" {percent:.0f}% ", style=style)
    text.append(f"({compact_count(used)}/{compact_count(limit) if limit else '?'})", style="#b8c3c7")


def token_part(tokens: Any) -> str:
    """Return compact input/output token totals."""
    tokens = tokens if isinstance(tokens, dict) else {}
    return f"In {compact_count(tokens.get('in'))} Out {compact_count(tokens.get('out'))}"


def short_model(model_name: str) -> str:
    """Return a status-line model label."""
    text = str(model_name or "loading")
    provider, sep, model = text.partition(":")
    if sep:
        model = model.rsplit("/", 1)[-1]
        text = f"{provider}:{model}"
    return truncate(text, 28)


def context_bar(percent: float) -> str:
    """Return a compact 10-cell context bar."""
    cells = 10
    filled = max(0, min(cells, round((percent / 100) * cells)))
    return "█" * filled + "░" * (cells - filled)


def context_style(percent: float) -> str:
    """Return a color for context pressure."""
    if percent >= 85:
        return "bold #ff6b6b"
    if percent >= 60:
        return "bold #f0c95a"
    return "bold #70d77a"


def duration_text(value: Any) -> str:
    """Return mm:ss or h:mm duration text."""
    seconds = positive_int(value)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def compact_count(value: Any) -> str:
    """Return compact token counts for a narrow status line."""
    number = positive_int(value)
    if number >= 1_000_000:
        return f"{number / 1_000_000:.1f}m"
    if number >= 1_000:
        return f"{number / 1_000:.1f}k"
    return str(number)


def percent_value(value: Any, used: int, limit: int) -> float:
    """Return a percent from stored data or used/limit counts."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = 0.0
    if parsed:
        return parsed
    return (used / limit) * 100 if limit else 0.0


def positive_int(value: Any) -> int:
    """Return a non-negative integer."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed > 0 else 0


def truncate(text: str, limit: int) -> str:
    """Shorten text for the status line."""
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"
