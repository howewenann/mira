"""Windows-safe rendering for Textual scrollbars."""

from __future__ import annotations

from textual.scrollbar import ScrollBar, ScrollBarRender


class SolidScrollBarRender(ScrollBarRender):
    """Render scrollbar thumbs without fractional block glyphs."""

    VERTICAL_BARS = [" "] * 8
    HORIZONTAL_BARS = [" "] * 8


def configure_scrollbars_for_platform(platform: str) -> None:
    """Use solid-cell scrollbars on Windows and leave other platforms alone."""
    if platform == "win32":
        ScrollBar.renderer = SolidScrollBarRender
