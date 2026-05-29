from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver


def make_checkpointer() -> MemorySaver:
    """Create the in-memory LangGraph checkpointer used by both agents."""
    return MemorySaver()
