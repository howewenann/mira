"""Startup splash rendering helpers for the Textual UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pyfiglet import Figlet
from rich.text import Text

from config.version import display_version

MIRA_CYAN = "#5bb8b1"
MIRA_TITLE = "bold #eef7f8"
MIRA_LABEL = "bold #d2a957"
MIRA_VALUE = "#e8edef"
MIRA_HINT = "#b8c1c7"
VERSION = display_version()
HINTS = "Hints: /help commands | /goal graded goal | /plan plan safely | /act action mode | Ctrl+C copy | Alt+Q cancel/quit"


def blocky_wordmark() -> str:
    """Return the restored blocky MIRA wordmark."""
    return Figlet(font="blocky").renderText("MIRA").rstrip()


def splash_text(*, model_name: str, session_id: str, workspace: str | Path) -> Text:
    """Build the Rich text used for the Textual startup splash."""
    wordmark = blocky_wordmark()
    logo_width = max((len(line.rstrip()) for line in wordmark.splitlines()), default=0)
    border = "=" * logo_width
    divider = "-" * logo_width

    text = Text()
    text.append(border + "\n", style=MIRA_CYAN)
    text.append(wordmark + "\n\n", style=MIRA_CYAN)
    text.append(VERSION + "\n", style=MIRA_TITLE)
    text.append(divider + "\n", style=MIRA_CYAN)
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


def append_label(text: Text, label: str, value: Any) -> None:
    """Append one aligned metadata label."""
    text.append(f"{label:<10}", style=MIRA_LABEL)
    text.append(str(value), style=MIRA_VALUE)
    text.append("\n")
