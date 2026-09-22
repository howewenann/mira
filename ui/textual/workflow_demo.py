"""Deterministic native LangGraph workflow used by the hidden TUI demo."""

from __future__ import annotations

import asyncio
import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send


class WorkflowDemoState(TypedDict):
    events: Annotated[list[str], operator.add]
    review_count: int


class WorkflowDemoWorker(TypedDict):
    item: str


def build_workflow_demo(*, delay_scale: float = 1.0):
    """Build a four-step graph with same-node fan-out and a one-pass loop."""

    async def prepare(_state: WorkflowDemoState) -> dict[str, object]:
        await asyncio.sleep(0.35 * delay_scale)
        return {"events": ["prepared"], "review_count": 0}

    def fan_out_workers(_state: WorkflowDemoState) -> list[Send]:
        return [
            Send("worker", {"item": "alpha"}),
            Send("worker", {"item": "beta"}),
            Send("worker", {"item": "gamma"}),
        ]

    async def worker(state: WorkflowDemoWorker) -> dict[str, list[str]]:
        delays = {"alpha": 1.2, "beta": 0.35, "gamma": 0.75}
        item = state["item"]
        await asyncio.sleep(delays[item] * delay_scale)
        return {"events": [f"{item} complete"]}

    async def review(state: WorkflowDemoState) -> dict[str, object]:
        await asyncio.sleep(0.45 * delay_scale)
        review_count = state.get("review_count", 0) + 1
        return {
            "events": [f"review {review_count} complete"],
            "review_count": review_count,
        }

    def continue_review(state: WorkflowDemoState) -> str:
        return "again" if state["review_count"] < 2 else "done"

    graph = StateGraph(WorkflowDemoState)
    graph.add_node("prepare", prepare)
    graph.add_node("worker", worker)
    graph.add_node("review", review)
    graph.add_edge(START, "prepare")
    graph.add_conditional_edges("prepare", fan_out_workers, ["worker"])
    graph.add_edge("worker", "review")
    graph.add_conditional_edges(
        "review",
        continue_review,
        {"again": "review", "done": END},
    )
    return graph.compile()


__all__ = ["WorkflowDemoState", "build_workflow_demo"]
