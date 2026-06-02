"""Session history list for the Textual TUI."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from textual.widgets import ListItem, ListView, Static


class SessionHistory(ListView):
    """Selectable list of saved sessions."""

    def refresh_sessions(self, store: Any, *, current_id: str = "") -> None:
        """Reload session rows from the backing store."""
        records = session_records(store)
        items: list[SessionItem] = []
        active_index = 0

        for index, record in enumerate(records):
            active = str(record.get("id") or "") == current_id
            if active:
                active_index = index
            items.append(SessionItem(record, active=active))

        self.clear()
        if items:
            self.extend(items)
            self.index = active_index
        else:
            self.append(ListItem(Static("No sessions yet", classes="session-empty"), disabled=True))


class SessionItem(ListItem):
    """List item carrying its session id for selection events."""

    def __init__(self, record: dict[str, Any], *, active: bool = False) -> None:
        self.session_id = str(record.get("id") or "")
        classes = "session-row active" if active else "session-row"
        super().__init__(Static(session_label(record), classes="session-label"), classes=classes)


def session_records(store: Any) -> list[dict[str, Any]]:
    """Return sessions sorted by most recently updated."""
    root = getattr(store, "root", None)
    if not isinstance(root, Path):
        return []

    paths = sorted(root.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    records = []
    for path in paths:
        try:
            records.append(store.read(path))
        except Exception:
            continue
    return records


def session_label(record: dict[str, Any]) -> str:
    """Return a compact two-line session row."""
    title = truncate(str(record.get("title") or "Untitled session"), 30)
    timestamp = timestamp_text(record.get("updated_at") or record.get("created_at"))
    turns = int(record.get("turns") or 0)
    return f"{title}\n{timestamp}  {turns} turns"


def timestamp_text(value: Any) -> str:
    """Format a persisted session timestamp."""
    text = str(value or "")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return "unknown time"
    return parsed.astimezone().strftime("%b %d %H:%M")


def truncate(value: str, limit: int) -> str:
    """Hard truncate long sidebar labels."""
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"
