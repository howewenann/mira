"""Prompt input widget for the MIRA TUI."""

from __future__ import annotations

from textual.message import Message
from textual.events import Key
from textual.widgets import TextArea


class PromptBox(TextArea):
    """Multiline prompt entry."""

    class Submitted(Message):
        def __init__(self, prompt: "PromptBox", value: str) -> None:
            super().__init__()
            self.prompt = prompt
            self.value = value

    def __init__(self, **kwargs: object) -> None:
        super().__init__("", placeholder="prompt", show_line_numbers=False, id="prompt", **kwargs)
        self._history: list[str] = []
        self._history_index: int | None = None
        self._history_draft = ""

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

    def remember(self, text: str) -> None:
        """Add a submitted prompt to in-memory history."""
        entry = text.strip()
        if not entry:
            return
        if not self._history or self._history[-1] != entry:
            self._history.append(entry)
        self._history_index = len(self._history)
        self._history_draft = ""

    def on_key(self, event: Key) -> None:
        """Submit prompts and navigate history."""
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
        self.value = self._history_draft

    def _show_history_value(self) -> None:
        """Render the current history entry in the prompt."""
        if self._history_index is None or self._history_index >= len(self._history):
            return
        self.value = self._history[self._history_index]

    def _move_cursor_to_end(self) -> None:
        self.cursor_location = self.document.end
