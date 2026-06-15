"""Readable generated names for UI labels."""

from __future__ import annotations

from collections.abc import Iterator

try:
    import coolname
except Exception:  # pragma: no cover - dependency fallback
    coolname = None

IGNORE_LIST = {
    "beaver",
    "booby",
    "chubby",
    "curvy",
    "demonic",
    "fat",
    "flashy",
    "flat",
    "funky",
    "gay",
    "godlike",
    "heretic",
    "juicy",
    "kickass",
    "nippy",
    "sexy",
    "sloppy",
    "thick",
}


def generate_slug(n_words: int = 2, *, fallback: Iterator[int] | None = None) -> str:
    """Return a short hyphenated slug, falling back to numbers if unavailable."""
    if coolname is not None:
        for _ in range(8):
            try:
                words = coolname.generate(n_words)
            except Exception:
                words = coolname.generate(max(2, n_words))[:n_words]
            if words and not IGNORE_LIST.intersection(words):
                return "-".join(words)
    return str(next(fallback)) if fallback is not None else "1"
