"""Shared display colors for terminal transcripts and Textual widgets."""

from __future__ import annotations

import re
import sys
from contextlib import suppress

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
THINKING_BODY = "\033[38;2;184;194;201m"
RUBRIC_HEADER_COLOR = "#C58FD6"
RUBRIC_BODY_COLOR = "#F1DCF5"
RUBRIC_HEADER = "\033[38;2;197;143;214m"
RUBRIC_BODY = "\033[38;2;241;220;245m"
ANSI_RE = re.compile(r"\033\[[0-9;]*m")

COLORS = {
    "user": "\033[38;2;214;179;90m",
    "user planning": "\033[38;2;214;179;90m",
    "mira": "\033[38;2;91;184;177m",
    "startup": "\033[38;2;91;184;177m",
    "thinking": "\033[38;2;130;144;154m",
    "status": "\033[38;2;125;155;209m",
    "info": "\033[38;2;108;182;255m",
    "warning": "\033[38;2;224;168;79m",
    "error": "\033[38;2;217;107;102m",
    "command": "\033[38;2;122;133;140m",
    "task": "\033[38;2;125;155;209m",
    "task request": "\033[38;2;125;155;209m",
    "system": "\033[38;2;125;155;209m",
    "rubric review": RUBRIC_HEADER,
    "goal": RUBRIC_HEADER,
}


class TerminalColorizer:
    """Apply display-only color to complete transcript blocks."""

    def __init__(self) -> None:
        self.current_body_color = ""

    def colorize(self, text: str) -> str:
        """Return colored terminal text without changing plain content."""
        return "".join(self.colorize_line(line) for line in str(text).splitlines(keepends=True))

    def colorize_line(self, line: str) -> str:
        """Return one colored terminal line."""
        label, separator, rest = line.partition(":")
        if separator and len(label) > 1:
            color = color_for_label(label)
            if color:
                self.current_body_color = body_color_for_label(label, color)
                return f"{BOLD}{color}{label}:{RESET}{self.current_body_color}{rest}{RESET}"

        if self.current_body_color:
            return f"{self.current_body_color}{line}{RESET}"
        return line


def strip_ansi(text: str) -> str:
    """Remove ANSI color escapes from text."""
    return ANSI_RE.sub("", text)


def terminal_header(title: str, detail: str = "") -> str:
    """Return a small colored terminal header."""
    header = f"{COLORS['mira']}{title}{RESET}\n"
    if detail:
        header += f"{DIM}{detail}{RESET}\n"
    return header


def colorize_line(line: str) -> str:
    """Return one colored line using a fresh colorizer."""
    return TerminalColorizer().colorize_line(line)


def color_for_label(label: str) -> str:
    """Return the terminal color for a block or line label."""
    normalized = label.strip().lower()
    if normalized in COLORS:
        return COLORS[normalized]
    if normalized.startswith("subagent -"):
        return COLORS["task"]
    if normalized.endswith(" output"):
        return COLORS["command"]
    if is_tool_label(normalized):
        return COLORS["command"]
    return ""


def body_color_for_label(label: str, header_color: str) -> str:
    """Return the body color for a block label."""
    if label.strip().lower() == "thinking":
        return THINKING_BODY
    if label.strip().lower() in {"rubric review", "goal"}:
        return RUBRIC_BODY
    return header_color


def is_tool_label(label: str) -> bool:
    """Return whether a label looks like a tool name rather than prose."""
    return bool(label) and " " not in label and any(character in label for character in ("_", "-"))


def enable_console_colors() -> None:
    """Enable ANSI color support on Windows consoles when possible."""
    if sys.platform != "win32":
        return
    with suppress(Exception):
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
