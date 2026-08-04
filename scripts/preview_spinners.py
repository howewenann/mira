"""Animate MIRA's spinner and ASCII alternatives in the current terminal."""

from __future__ import annotations

import argparse
import itertools
import sys
import time


SPINNERS = (
    ("Braille", ("\u280b", "\u2819", "\u2839", "\u2838", "\u283c", "\u2834", "\u2826", "\u2827", "\u2807", "\u280f")),
    ("Classic rotor (MIRA)", ("|", "/", "-", "\\")),
    ("Bracketed rotor", ("[|]", "[/]", "[-]", "[\\]")),
    ("Pulse", (".", "o", "O", "o")),
    ("Moving dot", ("[.  ]", "[ . ]", "[  .]", "[ . ]")),
    ("Bouncing bar", ("[=  ]", "[== ]", "[ ==]", "[  =]", "[ ==]", "[== ]")),
    ("Ellipsis", ("[   ]", "[.  ]", "[.. ]", "[...]")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview animated spinner styles in your actual terminal."
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="seconds to animate each spinner (default: 2)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.1,
        help="seconds between frames (default: 0.1)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.duration <= 0:
        raise SystemExit("--duration must be greater than zero")
    if args.interval <= 0:
        raise SystemExit("--interval must be greater than zero")
    if not sys.stdout.isatty():
        print(
            "This preview needs direct terminal output.\n"
            "Run `python scripts\\preview_spinners.py` from the activated environment,\n"
            "or use `conda run --no-capture-output -n ai_agents python "
            "scripts\\preview_spinners.py`."
        )
        return 2

    print("Spinner preview (Ctrl+C to stop)\n")
    label_width = max(len(name) for name, _frames in SPINNERS)

    try:
        for name, frames in SPINNERS:
            deadline = time.monotonic() + args.duration
            for frame_number in itertools.count():
                frame = frames[frame_number % len(frames)]
                print(f"\r{name:<{label_width}}  {frame:<5}", end="", flush=True)
                if time.monotonic() >= deadline:
                    break
                time.sleep(args.interval)
            print()
    except KeyboardInterrupt:
        print()

    print("\n\nPreview finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
