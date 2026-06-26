"""Observed context usage from DeepAgents runtime internals."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

ContextUsageCallback = Callable[[dict[str, Any]], None]

DEEPAGENTS_CONTEXT_SOURCE = "deepagents.summarization._count_tokens"

_context_usage_callback: ContextVar[ContextUsageCallback | None] = ContextVar(
    "mira_context_usage_callback",
    default=None,
)


@contextmanager
def context_usage_scope(callback: ContextUsageCallback | None) -> Iterator[None]:
    """Install a callback for observed DeepAgents context counts."""
    token = _context_usage_callback.set(callback)
    try:
        yield
    finally:
        _context_usage_callback.reset(token)


def record_deepagents_context_tokens(tokens: int) -> None:
    """Record a token count already computed by DeepAgents."""
    callback = _context_usage_callback.get()
    if callback is None or tokens <= 0:
        return

    callback(
        {
            "context_tokens": int(tokens),
            "context_source": DEEPAGENTS_CONTEXT_SOURCE,
            "source": DEEPAGENTS_CONTEXT_SOURCE,
        }
    )
