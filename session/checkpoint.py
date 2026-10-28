"""LangGraph checkpointer construction."""

from langgraph.checkpoint.memory import InMemorySaver


def make_checkpointer() -> InMemorySaver:
    """Create an in-memory LangGraph checkpointer for agents or Workflows."""
    return InMemorySaver()
