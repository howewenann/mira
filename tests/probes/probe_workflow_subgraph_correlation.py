"""Probe LangGraph subgraph -> parent task correlation for MIRA Workflow UI.

Run directly from ``tests/probes``:

    python probe_workflow_subgraph_correlation.py

Purpose
-------
Before designing MIRA's WorkflowCoordinator / Inspector plumbing, prove that
multiple concurrent nested subgraphs with the SAME node name can each be joined
deterministically back to the correct root Workflow task.

This deliberately exercises the hard case:

    Send("worker", ...)
    Send("worker", ...)
    Send("worker", ...)

where ``worker`` itself is a compiled LangGraph subgraph.

We compare three native surfaces:

1. Root ``TasksTransformer`` projection
       root task id + node name

2. ``run.subgraphs`` handles
       handle.path
       handle.graph_name
       handle.trigger_call_id

3. Root ``run.lifecycle`` projection
       namespace
       graph_name
       trigger_call_id

Required contract
-----------------
For every root ``worker`` task id T, exactly one direct child subgraph must
identify itself as that same T.

The probe accepts correlation only if ALL of these agree:

    handle.path[-1] == f"worker:{T}"
    handle.trigger_call_id == T

and the matching lifecycle ``started`` event agrees:

    lifecycle.namespace[-1] == f"worker:{T}"
    lifecycle.trigger_call_id == T

This is a pure LangGraph probe:
- no MIRA runtime
- no model
- no external services
"""

from __future__ import annotations

import asyncio
import json
import operator
import os
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.stream.transformers import TasksTransformer
from langgraph.types import Send


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "<unknown>"


def compact(value: Any, limit: int = 1200) -> str:
    try:
        text = json.dumps(value, default=str, ensure_ascii=False)
    except Exception:
        text = repr(value)

    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "..."


def report(label: str, passed: bool, detail: Any = None) -> bool:
    status = "PASS" if passed else "FAIL"
    print(f"\n[{status}] {label}")

    if detail is not None:
        print(compact(detail, 2200))

    return passed


def task_id_from_path_segment(segment: str) -> str | None:
    """Parse LangGraph's ``node_name:task_id`` namespace segment."""

    _name, separator, task_id = segment.partition(":")
    if not separator or not task_id:
        return None
    return task_id


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


@dataclass
class RootTask:
    phase: str
    task_id: str
    name: str


@dataclass
class SubgraphHandle:
    index: int
    graph_name: str | None
    path: tuple[str, ...]
    path_task_id: str | None
    trigger_call_id: str | None
    initial_status: str | None
    final_status: str | None = None
    final_error: str | None = None


@dataclass
class LifecycleEvent:
    event: str
    namespace: tuple[str, ...]
    graph_name: str | None
    trigger_call_id: str | None
    error: str | None


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


class State(TypedDict):
    item: Annotated[list[str], operator.add]
    results: Annotated[list[str], operator.add]


async def child_prepare(state: State) -> dict[str, Any]:
    """First node inside each worker subgraph."""

    item = state["item"][-1]

    # Different timings force the three concurrent worker subgraphs to
    # overlap and finish in a different order from their launch order.
    delays = {
        "alpha": 0.09,
        "beta": 0.03,
        "gamma": 0.06,
    }
    await asyncio.sleep(delays[item])

    return {"results": [f"{item}:prepare"]}


async def child_finish(state: State) -> dict[str, Any]:
    """Second node inside each worker subgraph."""

    item = state["item"][-1]
    await asyncio.sleep(0.02)

    return {"results": [f"{item}:finish"]}


def build_worker_subgraph():
    graph = StateGraph(State)
    graph.add_node("prepare", child_prepare)
    graph.add_node("finish", child_finish)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "finish")
    graph.add_edge("finish", END)
    return graph.compile()


def fan_out(_state: State) -> list[Send]:
    """Launch three concurrent copies of the same nested graph node."""

    return [
        Send("worker", {"item": ["alpha"], "results": []}),
        Send("worker", {"item": ["beta"], "results": []}),
        Send("worker", {"item": ["gamma"], "results": []}),
    ]


def build_root_graph():
    worker = build_worker_subgraph()

    graph = StateGraph(State)
    graph.add_node("worker", worker)
    graph.add_conditional_edges(START, fan_out, ["worker"])
    graph.add_edge("worker", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Stream consumers
# ---------------------------------------------------------------------------


async def consume_root_tasks(
    channel: Any,
    observations: list[RootTask],
) -> None:
    async for payload in channel:
        if not isinstance(payload, dict):
            raise TypeError(f"Unexpected task payload: {payload!r}")

        phase = "START" if "input" in payload else "RESULT"

        observation = RootTask(
            phase=phase,
            task_id=str(payload.get("id") or ""),
            name=str(payload.get("name") or ""),
        )
        observations.append(observation)

        print(
            f"ROOT TASK  "
            f"phase={phase:<6} "
            f"name={observation.name:<12} "
            f"id={observation.task_id}"
        )


async def consume_subgraphs(
    channel: Any,
    observations: list[SubgraphHandle],
    raw_handles: list[Any],
) -> None:
    index = 0

    async for handle in channel:
        index += 1
        raw_handles.append(handle)

        path = tuple(str(part) for part in getattr(handle, "path", ()) or ())
        graph_name = getattr(handle, "graph_name", None)
        trigger_call_id = getattr(handle, "trigger_call_id", None)

        path_task_id = (
            task_id_from_path_segment(path[-1])
            if path
            else None
        )

        observation = SubgraphHandle(
            index=index,
            graph_name=str(graph_name) if graph_name is not None else None,
            path=path,
            path_task_id=path_task_id,
            trigger_call_id=(
                str(trigger_call_id)
                if trigger_call_id is not None
                else None
            ),
            initial_status=getattr(handle, "status", None),
        )
        observations.append(observation)

        print(
            f"SUBGRAPH   "
            f"#{index} "
            f"graph_name={observation.graph_name!r} "
            f"path={observation.path!r} "
            f"path_task_id={observation.path_task_id!r} "
            f"trigger_call_id={observation.trigger_call_id!r} "
            f"status={observation.initial_status!r}"
        )


async def consume_lifecycle(
    channel: Any,
    observations: list[LifecycleEvent],
) -> None:
    async for payload in channel:
        if not isinstance(payload, dict):
            raise TypeError(f"Unexpected lifecycle payload: {payload!r}")

        observation = LifecycleEvent(
            event=str(payload.get("event") or ""),
            namespace=tuple(
                str(part)
                for part in (payload.get("namespace") or ())
            ),
            graph_name=(
                str(payload["graph_name"])
                if payload.get("graph_name") is not None
                else None
            ),
            trigger_call_id=(
                str(payload["trigger_call_id"])
                if payload.get("trigger_call_id") is not None
                else None
            ),
            error=(
                str(payload["error"])
                if payload.get("error") is not None
                else None
            ),
        )
        observations.append(observation)

        print(
            f"LIFECYCLE  "
            f"event={observation.event:<11} "
            f"graph_name={observation.graph_name!r} "
            f"namespace={observation.namespace!r} "
            f"trigger_call_id={observation.trigger_call_id!r}"
        )


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify(
    root_tasks: list[RootTask],
    handles: list[SubgraphHandle],
    lifecycle: list[LifecycleEvent],
) -> int:
    failures = 0

    root_starts = [
        item
        for item in root_tasks
        if item.phase == "START" and item.name == "worker"
    ]
    root_results = [
        item
        for item in root_tasks
        if item.phase == "RESULT" and item.name == "worker"
    ]

    root_ids = {item.task_id for item in root_starts}
    result_ids = {item.task_id for item in root_results}

    handle_path_ids = {
        item.path_task_id
        for item in handles
        if item.path_task_id is not None
    }
    handle_trigger_ids = {
        item.trigger_call_id
        for item in handles
        if item.trigger_call_id is not None
    }

    lifecycle_started = [
        item
        for item in lifecycle
        if item.event == "started"
        and item.graph_name == "worker"
    ]

    lifecycle_started_path_ids = {
        task_id_from_path_segment(item.namespace[-1])
        for item in lifecycle_started
        if item.namespace
    }
    lifecycle_started_path_ids.discard(None)

    lifecycle_started_trigger_ids = {
        item.trigger_call_id
        for item in lifecycle_started
        if item.trigger_call_id is not None
    }

    lifecycle_terminal = [
        item
        for item in lifecycle
        if item.event in {"completed", "failed", "interrupted", "drained"}
        and item.namespace
        and item.namespace[-1].startswith("worker:")
    ]

    lifecycle_terminal_path_ids = {
        task_id_from_path_segment(item.namespace[-1])
        for item in lifecycle_terminal
    }
    lifecycle_terminal_path_ids.discard(None)

    # 1. Hard case actually happened: three same-name concurrent root tasks.
    ok = (
        len(root_starts) == 3
        and len(root_ids) == 3
        and result_ids == root_ids
    )
    if not report(
        "Three concurrent same-name worker tasks have three distinct root task IDs",
        ok,
        {
            "starts": [asdict(item) for item in root_starts],
            "results": [asdict(item) for item in root_results],
        },
    ):
        failures += 1

    # 2. One direct child handle per root task.
    ok = (
        len(handles) == 3
        and len(handle_path_ids) == 3
        and handle_path_ids == root_ids
    )
    if not report(
        "handle.path uniquely joins every subgraph to its root worker task",
        ok,
        {
            "root_ids": sorted(root_ids),
            "handles": [asdict(item) for item in handles],
        },
    ):
        failures += 1

    # 3. Public trigger_call_id independently agrees.
    ok = (
        len(handle_trigger_ids) == 3
        and handle_trigger_ids == root_ids
    )
    if not report(
        "handle.trigger_call_id independently equals the parent root task ID",
        ok,
        {
            "root_ids": sorted(root_ids),
            "handle_trigger_ids": sorted(handle_trigger_ids),
        },
    ):
        failures += 1

    # 4. Each individual handle has both correlation signals agreeing.
    per_handle = [
        {
            "path": item.path,
            "path_task_id": item.path_task_id,
            "trigger_call_id": item.trigger_call_id,
            "agree": (
                item.path_task_id is not None
                and item.path_task_id == item.trigger_call_id
                and item.path_task_id in root_ids
            ),
        }
        for item in handles
    ]

    ok = len(per_handle) == 3 and all(item["agree"] for item in per_handle)

    if not report(
        "Each handle has two agreeing native correlation signals",
        ok,
        per_handle,
    ):
        failures += 1

    # 5. Lifecycle start events provide the same mapping.
    ok = (
        len(lifecycle_started) == 3
        and lifecycle_started_path_ids == root_ids
        and lifecycle_started_trigger_ids == root_ids
    )

    if not report(
        "Lifecycle started events agree with the same root task IDs",
        ok,
        {
            "started": [asdict(item) for item in lifecycle_started],
            "path_ids": sorted(lifecycle_started_path_ids),
            "trigger_ids": sorted(lifecycle_started_trigger_ids),
        },
    ):
        failures += 1

    # 6. Lifecycle terminal events retain the path join key even though terminal
    # payloads intentionally do not repeat trigger_call_id.
    ok = (
        len(lifecycle_terminal) == 3
        and lifecycle_terminal_path_ids == root_ids
    )

    if not report(
        "Lifecycle terminal events retain the namespace/path correlation",
        ok,
        {
            "terminal": [asdict(item) for item in lifecycle_terminal],
            "terminal_path_ids": sorted(lifecycle_terminal_path_ids),
        },
    ):
        failures += 1

    # 7. The same display name is insufficient by itself.
    ok = (
        len(handles) == 3
        and {item.graph_name for item in handles} == {"worker"}
        and len(handle_path_ids) == 3
    )

    if not report(
        "graph_name is intentionally non-unique; task identity comes from path / trigger_call_id",
        ok,
        {
            "graph_names": [item.graph_name for item in handles],
            "path_task_ids": [item.path_task_id for item in handles],
        },
    ):
        failures += 1

    return failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> int:
    print("=" * 110)
    print("MIRA Workflow UI · subgraph-to-parent correlation probe")
    print("=" * 110)
    print(f"cwd:               {os.getcwd()}")
    print(f"langgraph version: {package_version('langgraph')}")
    print(f"langchain version: {package_version('langchain')}")

    graph = build_root_graph()

    run = await graph.astream_events(
        {"item": [], "results": []},
        version="v3",
        transformers=[TasksTransformer],
    )

    print(f"v3 extensions: {sorted(run.extensions.keys())}")

    task_channel = run.extensions.get("tasks")
    if task_channel is None:
        raise RuntimeError(
            "TasksTransformer was registered but no 'tasks' projection exists."
        )

    root_tasks: list[RootTask] = []
    handles: list[SubgraphHandle] = []
    lifecycle: list[LifecycleEvent] = []
    raw_handles: list[Any] = []

    async with run:
        await asyncio.gather(
            consume_root_tasks(task_channel, root_tasks),
            consume_subgraphs(run.subgraphs, handles, raw_handles),
            consume_lifecycle(run.lifecycle, lifecycle),
        )

        interrupted = await run.interrupted()
        output = await run.output()

    # Read terminal public handle state only after the whole run has drained.
    for observation, handle in zip(handles, raw_handles, strict=True):
        observation.final_status = getattr(handle, "status", None)
        observation.final_error = getattr(handle, "error", None)

    print()
    print("=" * 110)
    print("RUN RESULT")
    print("=" * 110)
    print(f"interrupted: {interrupted}")
    print(f"output:      {compact(output, 1600)}")

    print()
    print("Final subgraph handle states:")
    for item in handles:
        print(
            f"  #{item.index} "
            f"path={item.path!r} "
            f"trigger_call_id={item.trigger_call_id!r} "
            f"status={item.final_status!r} "
            f"error={item.final_error!r}"
        )

    failures = verify(root_tasks, handles, lifecycle)

    print()
    print("=" * 110)
    print("FINAL INTERPRETATION")
    print("=" * 110)

    if failures:
        print(f"FAILED: {failures} assertion(s)")
        print()
        print(
            "Do NOT finalize WorkflowCoordinator -> Inspector correlation yet. "
            "Use the printed native fields above to understand the mismatch."
        )
        return 1

    print(
        """All correlation assertions passed.

The production join can be deterministic:

    root Workflow row
        task_id = task["id"]

    direct child subgraph
        handle.path[-1] == f"{node_name}:{task_id}"

    independent public cross-check
        handle.trigger_call_id == task_id

Therefore repeated same-name nodes from Send() are not ambiguous:

    worker  task=A  <->  path (..., "worker:A")
    worker  task=B  <->  path (..., "worker:B")
    worker  task=C  <->  path (..., "worker:C")

Recommended production rule:

    parent_task_id = handle.trigger_call_id

Optionally validate in debug/tests that:

    task_id_from_path_segment(handle.path[-1]) == parent_task_id

The Workflow panel should still use only root task events.
The correlated subgraph handle can feed the Inspector for that exact row.
"""
    )

    print("=" * 110)
    print("PASS: subgraph -> Workflow row correlation is safe to design against")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
