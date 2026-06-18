"""MIRA wrappers for DeepAgents compaction middleware."""

from __future__ import annotations

from functools import wraps
from typing import Any

from deepagents.middleware.summarization import create_summarization_tool_middleware

from runtime.compaction_state import compaction_scope


def create_mira_summarization_tool_middleware(model: Any, backend: Any) -> Any:
    """Create DeepAgents summarization middleware with an explicit live marker."""
    middleware = create_summarization_tool_middleware(model=model, backend=backend)
    mark_summarization_engine(getattr(middleware, "_summarization", None))
    return middleware


def mark_summarization_engine(summarization: Any) -> None:
    """Wrap DeepAgents summary generation methods so MIRA can filter their stream."""
    if summarization is None or getattr(summarization, "_mira_compaction_marked", False):
        return

    create_summary = getattr(summarization, "_create_summary", None)
    if callable(create_summary):

        @wraps(create_summary)
        def wrapped_create_summary(*args: Any, **kwargs: Any) -> Any:
            with compaction_scope():
                return create_summary(*args, **kwargs)

        setattr(summarization, "_create_summary", wrapped_create_summary)

    acreate_summary = getattr(summarization, "_acreate_summary", None)
    if callable(acreate_summary):

        @wraps(acreate_summary)
        async def wrapped_acreate_summary(*args: Any, **kwargs: Any) -> Any:
            with compaction_scope():
                return await acreate_summary(*args, **kwargs)

        setattr(summarization, "_acreate_summary", wrapped_acreate_summary)

    setattr(summarization, "_mira_compaction_marked", True)


__all__ = ["create_mira_summarization_tool_middleware", "mark_summarization_engine"]
