"""Shared MIRA branding primitives for terminal-facing surfaces."""

from __future__ import annotations

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


def blocky_wordmark() -> str:
    """Return MIRA's blocky wordmark."""
    return Figlet(font="blocky").renderText("MIRA").rstrip()


def branded_header() -> Text:
    """Build the wordmark, version, border, and divider shared by splashes."""
    wordmark = blocky_wordmark()
    logo_width = max((len(line.rstrip()) for line in wordmark.splitlines()), default=0)

    text = Text()
    text.append("=" * logo_width + "\n", style=MIRA_CYAN)
    text.append(wordmark + "\n\n", style=MIRA_CYAN)
    text.append(VERSION + "\n", style=MIRA_TITLE)
    text.append("-" * logo_width + "\n", style=MIRA_CYAN)
    return text


def append_label(text: Text, label: str, value: Any) -> None:
    """Append one aligned metadata label using MIRA's shared palette."""
    text.append(f"{label:<10}", style=MIRA_LABEL)
    text.append(str(value), style=MIRA_VALUE)
    text.append("\n")


__all__ = [
    "MIRA_CYAN",
    "MIRA_HINT",
    "MIRA_LABEL",
    "MIRA_TITLE",
    "MIRA_VALUE",
    "VERSION",
    "append_label",
    "blocky_wordmark",
    "branded_header",
]
