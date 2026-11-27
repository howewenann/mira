"""Native agent and HITL graph used by the hidden Workflow Phase 2 demo."""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import interrupt


def build_workflow_demo_phase2(agent: Any):
    """Build a two-node graph that exercises inspection and native resume."""

    def approval(_state: MessagesState) -> dict[str, object]:
        interrupt(
            {
                "type": "ask_user",
                "question": "Approve the Workflow Phase 2 demo?",
                "options": ["Approve"],
            }
        )
        return {}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("approval", approval)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", "approval")
    graph.add_edge("approval", END)
    return graph.compile(checkpointer=InMemorySaver())


__all__ = ["build_workflow_demo_phase2"]
