"""Small module used by the optional MIRA trace sidecar window."""

from __future__ import annotations

import sys
import time
from collections import deque
from contextlib import suppress
from pathlib import Path

BACKLOG_LINES = 40
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
THINKING_BODY = "\033[38;2;184;194;201m"

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
}


def main(argv: list[str] | None = None) -> int:
    """Tail a diagnostics log until the console is closed."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("Usage: python -m runtime.trace_tail <log-path>")
        return 2

    enable_console_colors()
    path = Path(args[0])
    print(f"{COLORS['mira']}MIRA Trace{RESET}")
    print(f"{DIM}{path}{RESET}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    colorizer = TraceColorizer()

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in deque(handle, maxlen=BACKLOG_LINES):
            print(colorizer.colorize(line), end="", flush=True)
        while True:
            line = handle.readline()
            if line:
                print(colorizer.colorize(line), end="", flush=True)
            else:
                time.sleep(0.25)


class TraceColorizer:
    """Apply sidecar-only color to complete trace blocks."""

    def __init__(self) -> None:
        self.current_color = ""
        self.current_body_color = ""

    def colorize(self, line: str) -> str:
        """Return one console-colored line without changing log content."""
        label, separator, rest = line.partition(":")
        if separator and len(label) > 1:
            color = color_for_label(label)
            if color:
                self.current_color = color
                self.current_body_color = body_color_for_label(label, color)
                return f"{BOLD}{color}{label}:{RESET}{self.current_body_color}{rest}{RESET}"

        if self.current_body_color:
            return f"{self.current_body_color}{line}{RESET}"
        return line


def colorize_line(line: str) -> str:
    """Return one console-colored trace line without changing log content."""
    return TraceColorizer().colorize(line)


def color_for_label(label: str) -> str:
    """Return the trace sidecar color for a block or line label."""
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
    """Return the sidecar body color for a block label."""
    if label.strip().lower() == "thinking":
        return THINKING_BODY
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


if __name__ == "__main__":
    raise SystemExit(main())
