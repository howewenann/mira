"""Prompt input widget for the MIRA TUI."""

from __future__ import annotations

from textual.widgets import Input


class PromptBox(Input):
    """Single-line prompt entry."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(placeholder="prompt", id="prompt", **kwargs)
