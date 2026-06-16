"""Status bar for the MIRA TUI."""

from __future__ import annotations

from typing import Any

from rich.text import Text
from textual.widgets import Static


class StatusBar(Static):
    """One-line session and activity status."""

    def set_state(
        self,
        *,
        mode: str,
        model_name: str,
        state: str,
        dashboard: dict[str, Any] | None = None,
        turns: int = 0,
        detail: str = "",
    ) -> None:
        """Update the status bar text."""
        dashboard = dashboard or {}
        text = Text()
        append_part(text, "MIRA", "bold #d6fff6")
        append_part(text, mode)
        append_part(text, state.title())
        append_part(text, short_model(model_name))
        append_context(text, dashboard.get("context") if isinstance(dashboard, dict) else {})
        append_part(text, token_part(dashboard.get("tokens") if isinstance(dashboard, dict) else {}))
        append_part(text, f"Turns {max(0, int(turns or 0))}")
        append_part(text, duration_text(dashboard.get("duration_seconds", 0)))
        if detail:
            append_part(text, detail)
        self.update(text)


def append_part(text: Text, value: str, style: str = "#d7dee2") -> None:
    """Append one pipe-separated status part."""
    if len(text):
        text.append(" | ", style="#6f8389")
    text.append(str(value), style=style)


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
