"""Session history list for the Textual TUI."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from textwrap import wrap
from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.events import Click, Key
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Input, ListItem, ListView, OptionList, Static
from textual.widgets.option_list import Option

SESSION_ROW_PREVIEW_WIDTH = 27


class HistoryToggleButton(Button):
    """Mouse toggle that never retains keyboard focus."""

    can_focus = False


class SessionActionButton(Button):
    """Mouse row action that never retains keyboard focus."""

    can_focus = False


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

    class PinRequested(Message):
        def __init__(self, item: SessionItem) -> None:
            super().__init__()
            self.item = item

    class MenuRequested(Message):
        def __init__(self, item: SessionItem) -> None:
            super().__init__()
            self.item = item

    class RenameSubmitted(Message):
        def __init__(self, item: SessionItem, value: str) -> None:
            super().__init__()
            self.item = item
            self.value = value

    def __init__(self, record: dict[str, Any], *, active: bool = False) -> None:
        self.record = record
        self.session_id = str(record.get("id") or "")
        classes = "session-row active" if active else "session-row"
        super().__init__(classes=classes)

    @property
    def pinned(self) -> bool:
        return self.record.get("pinned") is True

    def compose(self) -> ComposeResult:
        with Horizontal(classes="session-row-layout"):
            with Vertical(classes="session-text-column"):
                yield Static(
                    session_label(
                        self.record,
                        width=SESSION_ROW_PREVIEW_WIDTH,
                        pad_preview=True,
                    ),
                    classes="session-label",
                )
                with Vertical(classes="session-rename-editor") as editor:
                    editor.display = False
                    yield Input(classes="session-rename-input")
                    yield Static("", classes="session-rename-spacer")
                    yield Static(
                        timestamp_text(
                            self.record.get("updated_at") or self.record.get("created_at")
                        ),
                        classes="session-rename-timestamp",
                    )
            with Vertical(classes="session-actions"):
                pin_button = SessionActionButton(
                    "[!]",
                    classes=f"session-pin-action{' pinned' if self.pinned else ''}",
                    compact=True,
                )
                pin_button.styles.line_pad = 0
                yield pin_button
                menu_button = SessionActionButton(
                    "...", classes="session-menu-action", compact=True
                )
                menu_button.styles.line_pad = 0
                yield menu_button

    def on_click(self, event: Click) -> None:
        if isinstance(event.widget, Button) or event.widget.has_class("session-rename-input"):
            event.stop()

    @on(Button.Pressed, ".session-pin-action")
    def request_pin(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(self.PinRequested(self))

    @on(Button.Pressed, ".session-menu-action")
    def request_menu(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(self.MenuRequested(self))

    @on(Input.Submitted, ".session-rename-input")
    def submit_rename(self, event: Input.Submitted) -> None:
        event.stop()
        self.post_message(self.RenameSubmitted(self, event.value))

    def on_key(self, event: Key) -> None:
        editor = self.query_one(".session-rename-input", Input)
        if event.key != "escape" or not editor.has_focus:
            return
        event.stop()
        event.prevent_default()
        self.cancel_rename()

    def begin_rename(self) -> None:
        """Swap the preview for an inline editor in this row."""
        self.query_one(".session-label", Static).display = False
        self.query_one(".session-rename-editor").display = True
        editor = self.query_one(".session-rename-input", Input)
        editor.value = session_display_title(self.record)
        self.call_after_refresh(editor.focus)

    def cancel_rename(self) -> None:
        """Restore the normal preview without changing persisted metadata."""
        self.query_one(".session-rename-editor").display = False
        self.query_one(".session-label", Static).display = True
        self.focus()


class SessionActionMenuScreen(ModalScreen[str | None]):
    """Compact native action menu for one session row."""

    POPUP_WIDTH = 18
    POPUP_HEIGHT = 6
    BINDINGS = [("escape", "cancel", "Cancel")]
    AUTO_FOCUS = "#session-action-options"

    def __init__(self, *, pinned: bool, anchor_x: int, anchor_y: int) -> None:
        super().__init__()
        self.pinned = pinned
        self.anchor_x = anchor_x
        self.anchor_y = anchor_y

    def compose(self) -> ComposeResult:
        with Vertical(id="session-action-popup"):
            yield OptionList(
                Option("Rename", id="rename"),
                Option("Unpin" if self.pinned else "Pin", id="pin"),
                Option("Delete", id="delete"),
                Option("Cancel", id="cancel"),
                id="session-action-options",
            )

    def on_mount(self) -> None:
        """Place the menu beside its originating History row."""
        x = max(0, min(self.anchor_x, self.size.width - self.POPUP_WIDTH))
        y = max(0, min(self.anchor_y, self.size.height - self.POPUP_HEIGHT))
        self.query_one("#session-action-popup").styles.offset = (x, y)

    @on(OptionList.OptionSelected, "#session-action-options")
    def choose_action(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        action = str(event.option.id or "")
        self.dismiss(None if action == "cancel" else action)

    def action_cancel(self) -> None:
        self.dismiss(None)


def session_records(store: Any) -> list[dict[str, Any]]:
    """Return sessions sorted by most recently updated."""
    root = getattr(store, "root", None)
    if not isinstance(root, Path):
        return []

    records = []
    for path in root.glob("*.json"):
        try:
            records.append(store.read(path))
        except Exception:
            continue
    records.sort(
        key=lambda record: (
            record.get("pinned") is True,
            session_updated_timestamp(record),
        ),
        reverse=True,
    )
    return records


def session_label(
    record: dict[str, Any],
    *,
    width: int = 34,
    pad_preview: bool = False,
) -> str:
    """Return a compact session row with a prompt preview."""
    preview = preview_lines(session_display_title(record), width=width)
    if pad_preview:
        preview = [*preview[:2], *([""] * max(0, 2 - len(preview)))]
    timestamp = timestamp_text(record.get("updated_at") or record.get("created_at"))
    return "\n".join([*preview, timestamp])


def session_display_title(record: dict[str, Any]) -> str:
    """Apply custom, recent-user, persisted, and untitled precedence."""
    custom = compact_line(record.get("custom_title")).strip("\"'` ")
    if custom:
        return custom
    prompt = latest_user_prompt(record)
    if prompt:
        return prompt
    persisted = compact_line(record.get("title")).strip("\"'` ")
    return persisted or "Untitled session"


def latest_user_prompt(record: dict[str, Any]) -> str:
    """Return the newest visible user prompt from a session record."""
    events = record.get("events")
    if not isinstance(events, list):
        return ""

    for event in reversed(events):
        if not isinstance(event, dict) or event.get("type") != "user":
            continue
        text = compact_line(event.get("text"))
        if text:
            return text
    return ""


def preview_lines(value: str, *, width: int = 34, max_lines: int = 2) -> list[str]:
    """Return one or two ordered preview lines for the sidebar."""
    text = compact_line(value).strip("\"'` ")
    if not text:
        return ["Untitled session"]

    lines = wrap(text, width=width, max_lines=max_lines, placeholder="...")
    return lines or ["Untitled session"]


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


def session_updated_timestamp(record: dict[str, Any]) -> float:
    """Return a sortable timestamp without relying on metadata-write mtimes."""
    text = str(record.get("updated_at") or record.get("created_at") or "")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return 0.0


def compact_line(value: Any) -> str:
    return " ".join(str(value or "").split())
