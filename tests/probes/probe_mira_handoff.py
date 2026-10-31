#!/usr/bin/env python
"""MIRA context-handoff diagnostic. No LLM calls, no MIRA repo writes.

Run from MIRA's existing Python environment:
    python probe_mira_handoff.py

Creates a temporary SQLite checkpoint database and two small LangGraph graphs
that model Act/Plan's overlapping message channels and independent other state.
Tests the exact DeepAgents effective-message calculation installed locally.
This is a primitive-level probe, not a replacement for MIRA integration tests.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Annotated, Any, TypedDict

try:
    from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
    from deepagents.middleware.summarization import (
        _DeepAgentsSummarizationMiddleware,
    )
except ImportError as exc:
    raise SystemExit(
        f"Missing MIRA dependency: {exc}\n"
        "Use the Python interpreter from your MIRA conda environment."
    ) from exc


class ActState(TypedDict, total=False):
    messages: Annotated[list[Any], add_messages]
    _summarization_event: dict[str, Any] | None
    todos: list[str]
    act_marker: str


class PlanState(TypedDict, total=False):
    messages: Annotated[list[Any], add_messages]
    _summarization_event: dict[str, Any] | None
    plan_marker: str


def graph_for(schema: type, saver: AsyncSqliteSaver):
    builder = StateGraph(schema)
    # The node never runs. It exists only to identify a legitimate writer
    # for aupdate_state(), exactly as an agent graph does.
    builder.add_node("turn", lambda state: {})
    builder.add_edge(START, "turn")
    builder.add_edge("turn", END)
    return builder.compile(checkpointer=saver)


def config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def msg(label: str):
    cls = AIMessage if label[0] in {"A", "P"} and label[-1].isdigit() and int(label[-1]) % 2 == 0 else HumanMessage
    return cls(content=label, id=label)


def labels(messages: list[Any]) -> list[str]:
    return [m.content for m in messages]


def check(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(f"FAIL: {label}")
    print(f"PASS: {label}")


async def write_state(graph, thread: str, payload: dict[str, Any]) -> None:
    await graph.aupdate_state(config(thread), payload, as_node="turn")


async def state(graph, thread: str) -> dict[str, Any]:
    snapshot = await graph.aget_state(config(thread))
    if snapshot.next or snapshot.tasks:
        raise AssertionError(f"Thread {thread} has pending graph work; refusing handoff")
    return snapshot.values


def effective(value: dict[str, Any]) -> list[Any]:
    return _DeepAgentsSummarizationMiddleware._apply_event_to_messages(
        list(value.get("messages") or []),
        value.get("_summarization_event"),
    )


async def handoff(source_graph, source_id: str, target_graph, target_id: str):
    source = await state(source_graph, source_id)
    before = await state(target_graph, target_id)
    inherited = effective(source)
    # NOTE: This tests the candidate strategy; it is NOT production code.
    # Explicit reset is necessary: the old cutoff indexes refer to old messages.
    await write_state(
        target_graph,
        target_id,
        {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *inherited,
            ],
            "_summarization_event": None,
        },
    )
    after = await state(target_graph, target_id)
    check(
        labels(effective(after)) == labels(inherited),
        f"{source_id} -> {target_id}: effective messages match",
    )
    check(
        len(after["messages"]) == len(inherited),
        f"{source_id} -> {target_id}: no duplicate messages",
    )
    check(
        after.get("_summarization_event") is None,
        f"{source_id} -> {target_id}: stale compaction cleared",
    )
    for key in ("todos", "act_marker", "plan_marker"):
        if key in before:
            check(after.get(key) == before[key],
                  f"{target_id}: preserves existing {key}")


async def probe(saver: AsyncSqliteSaver):
    act = graph_for(ActState, saver)
    plan = graph_for(PlanState, saver)
    await write_state(
        act, "probe:act",
        {
            "messages": [msg("A1"), msg("A2"), msg("A3")],
            "todos": ["finish task", "run tests"],
            "act_marker": "keep ACT-only state",
        },
    )
    await write_state(
        plan, "probe:plan",
        {
            "messages": [msg("OLD1")],
            "plan_marker": "keep PLAN-only state",
            "_summarization_event": {
                "cutoff_index": 1,
                "summary_message": HumanMessage(
                    content="obsolete plan summary", id="obsolete-summary"
                ),
                "file_path": None,
            },
        },
    )

    print("\n--- 1. Act -> Plan (uncompacted) ---")
    await handoff(act, "probe:act", plan, "probe:plan")
    await write_state(plan, "probe:plan", {"messages": [msg("P1"), msg("P2")]})

    print("\n--- 2. Plan compacts, then Plan -> Act ---")
    await write_state(
        plan, "probe:plan",
        {
            "_summarization_event": {
                "cutoff_index": 4,  # A1 A2 A3 P1 summarized; P2 retained
                "summary_message": HumanMessage(
                    content="SUMMARY_OF_A1_TO_P1", id="summary-plan"
                ),
                "file_path": None,
            }
        },
    )
    plan_now = await state(plan, "probe:plan")
    check(
        labels(effective(plan_now)) == ["SUMMARY_OF_A1_TO_P1", "P2"],
        "DeepAgents reconstructs Plan's compacted effective context",
    )
    await handoff(plan, "probe:plan", act, "probe:act")
    await write_state(act, "probe:act", {"messages": [msg("A4"), msg("A5")]})

    print("\n--- 3. Act compacts, then Act -> Plan ---")
    await write_state(
        act, "probe:act",
        {
            "_summarization_event": {
                "cutoff_index": 3,  # prior summary, P2, A4 summarized; A5 kept
                "summary_message": HumanMessage(
                    content="SUMMARY_THROUGH_A4", id="summary-act"
                ),
                "file_path": None,
            }
        },
    )
    act_now = await state(act, "probe:act")
    check(
        labels(effective(act_now)) == ["SUMMARY_THROUGH_A4", "A5"],
        "DeepAgents reconstructs Act's compacted effective context",
    )
    await handoff(act, "probe:act", plan, "probe:plan")

    print("\n--- 4. Cold restart checks performed by caller ---")
    return ["SUMMARY_THROUGH_A4", "A5"]


async def main():
    print("MIRA message-handoff probe (no external model calls)")
    for package in ("langgraph", "deepagents", "langchain-core", "langgraph-checkpoint-sqlite"):
        try:
            print(f"  {package}: {version(package)}")
        except PackageNotFoundError:
            print(f"  {package}: version unavailable")
    with tempfile.TemporaryDirectory(prefix="mira-handoff-probe-") as d:
        db_path = str(Path(d) / "probe-checkpoints.db")
        async with AsyncSqliteSaver.from_conn_string(db_path) as saver:
            expected = await probe(saver)
        # Reopen a new SQLite connection and two freshly compiled graphs.
        async with AsyncSqliteSaver.from_conn_string(db_path) as saver:
            act = graph_for(ActState, saver)
            plan = graph_for(PlanState, saver)
            act_state = await state(act, "probe:act")
            plan_state = await state(plan, "probe:plan")
            check(
                labels(effective(plan_state)) == expected,
                "Plan context survives SQLite cold restart",
            )
            check(
                act_state["todos"] == ["finish task", "run tests"],
                "Act todos survive SQLite cold restart",
            )
            check(
                plan_state["plan_marker"] == "keep PLAN-only state",
                "Plan-only state survives SQLite cold restart",
            )
    print("\nALL PROBE CHECKS PASSED")
    print("Next: verify the same behavior against MIRA's compiled agent graphs.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        print(f"\nPROBE FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
