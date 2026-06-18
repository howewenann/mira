"""Live state for DeepAgents compaction model calls."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from collections.abc import Iterator

_COMPACTION_DEPTH: ContextVar[int] = ContextVar("mira_compaction_depth", default=0)


def compaction_active() -> bool:
    """Return whether the current context is running a compaction model call."""
    return _COMPACTION_DEPTH.get() > 0


@contextmanager
def compaction_scope() -> Iterator[None]:
    """Mark the current context as a DeepAgents compaction call."""
    token = _COMPACTION_DEPTH.set(_COMPACTION_DEPTH.get() + 1)
    try:
        yield
    finally:
        _COMPACTION_DEPTH.reset(token)


__all__ = ["compaction_active", "compaction_scope"]
