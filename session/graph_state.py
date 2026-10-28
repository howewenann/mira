"""Checkpointed MIRA session semantics beside native Act and Plan graph state."""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import START, StateGraph


class SessionGraphState(TypedDict):
    mira_session: dict[str, Any]


def make_session_graph(checkpointer: Any) -> Any:
    """Build a terminal graph for durable application-owned transitions."""
    builder = StateGraph(SessionGraphState)
    builder.add_node("save", lambda state: {})
    builder.add_edge(START, "save")
    return builder.compile(checkpointer=checkpointer)
