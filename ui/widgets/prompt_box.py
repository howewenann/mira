"""Prompt input widget for the MIRA TUI."""

from __future__ import annotations

from textual.events import Key
from textual.widgets import Input


class PromptBox(Input):
    """Single-line prompt entry."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(placeholder="prompt", id="prompt", **kwargs)
        self._history: list[str] = []
        self._history_index: int | None = None
        self._history_draft = ""

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
        """Navigate prompt history with Up and Down."""
        if event.key == "up":
            event.stop()
            self._previous_history()
            return

        if event.key == "down":
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
        self.cursor_position = len(self.value)

    def _show_history_value(self) -> None:
        """Render the current history entry in the prompt."""
        if self._history_index is None or self._history_index >= len(self._history):
            return
        self.value = self._history[self._history_index]
        self.cursor_position = len(self.value)
