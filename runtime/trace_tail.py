"""Small module used by the optional MIRA trace sidecar window."""

from __future__ import annotations

import sys
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """Tail a diagnostics log until the console is closed."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("Usage: python -m runtime.trace_tail <log-path>")
        return 2

    path = Path(args[0])
    print("MIRA Trace")
    print(str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        handle.seek(0, 2)
        while True:
            line = handle.readline()
            if line:
                print(line, end="", flush=True)
            else:
                time.sleep(0.25)


if __name__ == "__main__":
    raise SystemExit(main())
