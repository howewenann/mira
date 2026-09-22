"""Deterministic native LangGraph workflow used by the hidden TUI demo."""

from __future__ import annotations

import asyncio
import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph


class WorkflowDemoState(TypedDict):
    events: Annotated[list[str], operator.add]


def build_workflow_demo(*, delay_scale: float = 1.0):
    """Build a three-step graph with a visibly parallel middle step."""

    async def prepare(_state: WorkflowDemoState) -> dict[str, list[str]]:
        await asyncio.sleep(0.35 * delay_scale)
        return {"events": ["prepared"]}

    async def inspect_code(_state: WorkflowDemoState) -> dict[str, list[str]]:
        await asyncio.sleep(0.85 * delay_scale)
        return {"events": ["code inspected"]}

    async def inspect_tests(_state: WorkflowDemoState) -> dict[str, list[str]]:
        await asyncio.sleep(1.2 * delay_scale)
        return {"events": ["tests inspected"]}

    async def summarize(_state: WorkflowDemoState) -> dict[str, list[str]]:
        await asyncio.sleep(0.45 * delay_scale)
        return {"events": ["summarized"]}

    graph = StateGraph(WorkflowDemoState)
    graph.add_node("prepare", prepare)
    graph.add_node("inspect_code", inspect_code)
    graph.add_node("inspect_tests", inspect_tests)
    graph.add_node("summarize", summarize)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "inspect_code")
    graph.add_edge("prepare", "inspect_tests")
    graph.add_edge(["inspect_code", "inspect_tests"], "summarize")
    graph.add_edge("summarize", END)
    return graph.compile()


__all__ = ["WorkflowDemoState", "build_workflow_demo"]
