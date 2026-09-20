"""Prompt input widget for the MIRA TUI."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from textual import events
from textual.message import Message
from textual.events import Key
from textual.widgets import TextArea

from ui.textual.platform.windows.clipboard import get_windows_clipboard_files


class PromptBox(TextArea):
    """Multiline prompt entry."""

    class Submitted(Message):
        def __init__(self, prompt: "PromptBox", value: str) -> None:
            super().__init__()
            self.prompt = prompt
            self.value = value

    class LocalFilesPasted(Message):
        """Explorer files received through an empty Windows paste event."""

        def __init__(self, prompt: "PromptBox", paths: list[Path]) -> None:
            super().__init__()
            self.prompt = prompt
            self.paths = paths

    def __init__(self, **kwargs: object) -> None:
        super().__init__("", placeholder="prompt", show_line_numbers=False, id="prompt", **kwargs)
        self._history: list[str] = []
        self._history_index: int | None = None
        self._history_draft = ""
        self._untouched_history_entry = False

    @property
    def value(self) -> str:
        return self.text

    @value.setter
    def value(self, text: str) -> None:
        self.text = text
        self._move_cursor_to_end()

    def set_history(self, entries: list[str]) -> None:
        """Replace the prompt history used by Up/Down navigation."""
        self._history = [entry for entry in entries if entry]
        self._history_index = len(self._history)
        self._history_draft = ""
        self._untouched_history_entry = False

    def remember(self, text: str) -> None:
        """Add a submitted prompt to in-memory history."""
        entry = text.strip()
        if not entry:
            return
        if not self._history or self._history[-1] != entry:
            self._history.append(entry)
        self._history_index = len(self._history)
        self._history_draft = ""
        self._untouched_history_entry = False

    @property
    def displaying_untouched_history_entry(self) -> bool:
        """Return whether the prompt still contains a recalled history entry."""
        if not self._untouched_history_entry:
            return False
        if (
            self._history_index is None
            or self._history_index >= len(self._history)
            or self.value != self._history[self._history_index]
        ):
            self._untouched_history_entry = False
        return self._untouched_history_entry

    def on_key(self, event: Key) -> None:
        """Submit prompts and navigate history."""
        completion_handler = getattr(self.parent, "handle_prompt_key", None)
        if callable(completion_handler) and completion_handler(event):
            return

        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self.post_message(self.Submitted(self, self.value))
            return

        if event.key == "shift+enter":
            event.stop()
            event.prevent_default()
            self.insert("\n")
            return

        if event.key == "up" and (self.document.line_count <= 1 or self.cursor_at_start_of_text):
            event.stop()
            self._previous_history()
            return

        if event.key == "down" and (self.document.line_count <= 1 or self.cursor_at_end_of_text):
            event.stop()
            self._next_history()

    async def _on_paste(self, event: events.Paste) -> None:
        """Promote empty Windows CF_HDROP pastes without affecting text paste."""
        if event.text or sys.platform != "win32" or self.disabled or self.read_only:
            await super()._on_paste(event)
            return

        paths = await asyncio.to_thread(get_windows_clipboard_files)
        if not paths:
            await super()._on_paste(event)
            return

        event.stop()
        event.prevent_default()
        self.post_message(self.LocalFilesPasted(self, paths))

    def insert_file_references(self, references: list[str]) -> None:
        """Insert normal file references at the selection with clean boundaries."""
        if not references:
            return
        start, end = sorted((self.selection.start, self.selection.end))
        start_offset = _offset_from_location(self.value, start)
        end_offset = _offset_from_location(self.value, end)
        before = self.value[:start_offset]
        after = self.value[end_offset:]
        prefix = " " if before and not before[-1].isspace() else ""
        suffix = " " if after and not after[0].isspace() else ""
        result = self.replace(
            f"{prefix}{' '.join(references)}{suffix}",
            start,
            end,
            maintain_selection_offset=False,
        )
        self.move_cursor(result.end_location)

    def _previous_history(self) -> None:
        """Move to the previous prompt history entry."""
        if not self._history:
            return

        if self._history_index is None or self._history_index >= len(self._history):
            self._history_draft = self.value
            self._history_index = len(self._history) - 1
        elif self._history_index > 0:
            self._history_index -= 1

        self._show_history_value()

    def _next_history(self) -> None:
        """Move to the next prompt history entry or restore the draft."""
        if not self._history or self._history_index is None:
            return

        if self._history_index < len(self._history) - 1:
            self._history_index += 1
            self._show_history_value()
            return

        self._history_index = len(self._history)
        self._untouched_history_entry = False
        self.value = self._history_draft

    def _show_history_value(self) -> None:
        """Render the current history entry in the prompt."""
        if self._history_index is None or self._history_index >= len(self._history):
            return
        self._untouched_history_entry = True
        self.value = self._history[self._history_index]

    def _move_cursor_to_end(self) -> None:
        self.cursor_location = self.document.end

    def watch_disabled(self, disabled: bool) -> None:
        """Notify the prompt wrapper whenever availability changes."""
        callback = getattr(self.parent, "prompt_disabled_changed", None)
        if callable(callback):
            callback(disabled)


def _offset_from_location(text: str, location: tuple[int, int]) -> int:
    """Translate a TextArea row/column location into a string offset."""
    row, column = location
    lines = text.splitlines(keepends=True)
    return sum(len(line) for line in lines[:row]) + column
