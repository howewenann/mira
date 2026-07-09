"""Small module used by the optional MIRA trace sidecar window."""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path

from ui.terminal_colors import TerminalColorizer, enable_console_colors, terminal_header

BACKLOG_LINES = 40


def main(argv: list[str] | None = None) -> int:
    """Tail a diagnostics log until the console is closed."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("Usage: python -m runtime.trace_tail <log-path>")
        return 2

    enable_console_colors()
    path = Path(args[0])
    print(terminal_header("MIRA Trace", str(path)), end="")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    colorizer = TerminalColorizer()

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in deque(handle, maxlen=BACKLOG_LINES):
            print(colorizer.colorize_line(line), end="", flush=True)
        while True:
            line = handle.readline()
            if line:
                print(colorizer.colorize_line(line), end="", flush=True)
            else:
                time.sleep(0.25)


if __name__ == "__main__":
    raise SystemExit(main())
