"""Startup splash rendering helpers for the Textual UI."""

from __future__ import annotations

from pathlib import Path

from rich.text import Text

from config.branding import (
    MIRA_CYAN,
    MIRA_HINT,
    MIRA_LABEL,
    MIRA_TITLE,
    MIRA_VALUE,
    VERSION,
    append_label,
    blocky_wordmark,
    branded_header,
)

HINTS = "Hints: /help commands | /goal outcome goal | /plan plan safely | /act action mode | Ctrl+C copy | Alt+Q cancel/quit"


def splash_text(*, model_name: str, session_id: str, workspace: str | Path) -> Text:
    """Build the Rich text used for the Textual startup splash."""
    text = branded_header()
    append_label(text, "session", session_id)
    append_label(text, "model", model_name)
    append_label(text, "workspace", workspace)
    text.append("\n")
    text.append(HINTS, style=MIRA_HINT)
    return text


def loading_splash_text(*, workspace: str | Path, state: str, frame: str = "-") -> Text:
    """Build the startup splash shown while agents are still loading."""
    text = splash_text(model_name="loading", session_id="starting", workspace=workspace)
    text.append("\n\n")
    text.append(f"{frame} {state}", style="bold #d2a957")
    return text


__all__ = [
    "HINTS",
    "MIRA_CYAN",
    "MIRA_HINT",
    "MIRA_LABEL",
    "MIRA_TITLE",
    "MIRA_VALUE",
    "VERSION",
    "append_label",
    "blocky_wordmark",
    "loading_splash_text",
    "splash_text",
]
