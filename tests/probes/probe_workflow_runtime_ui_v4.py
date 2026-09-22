"""Final probe for MIRA Workflow UI runtime correlation.

Run from ``tests/probes``:

    python probe_workflow_runtime_ui_v4.py

This probe compares two LangGraph v3 projections:

- ``TasksTransformer``: candidate production signal for MIRA Workflow UI.
- ``DebugTransformer``: PROBE-ONLY oracle carrying LangGraph's native ``step``.

Why this exists
---------------
LangGraph 1.2.11 intentionally strips framework metadata such as
``langgraph_step`` from the normal task payload. Therefore MIRA should not
expect ``task["metadata"]["langgraph_step"]`` from ``TasksTransformer``.

Instead, this probe asks whether a tiny batch tracker over normal task events
produces exactly the same grouping as LangGraph's native debug ``step``.

If it does, production MIRA can stay on the lightweight task projection while
the debug projection remains probe-only.

The probe also verifies:

1. Parallel nodes belong to one inferred/native step.
2. ``Send()`` fan-out copies belong to one inferred/native step.
3. Loops create a later inferred/native step.
4. Root tasks remain first-level only when a nested graph runs.
5. HITL resume reuses the same native task ID and therefore the same UI row.
"""

from __future__ import annotations

import asyncio
import json
import operator
import os
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Annotated, Any, NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.stream.transformers import DebugTransformer, TasksTransformer
from langgraph.types import Command, Send, interrupt

try:
    from langgraph.checkpoint.memory import InMemorySaver
except ImportError:
    from langgraph.checkpoint.memory import MemorySaver as InMemorySaver


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "<unknown>"


def compact(value: Any, limit: int = 800) -> str:
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
        print(compact(detail, 1600))

    return passed


@dataclass
class TaskObservation:
    case: str
    pass_number: int
    phase: str
    task_id: str
    name: str
    inferred_step: int
    interrupts: list[Any]
    error: Any


@dataclass
class DebugObservation:
    case: str
    pass_number: int
    phase: str
    task_id: str
    name: str
    native_step: int


@dataclass
class SubgraphObservation:
    case: str
    pass_number: int
    graph_name: str | None
    namespace: Any


class StepTracker:
    """Infer Pregel superstep groups using only normal task events.

    LangGraph emits all task-start events for one Pregel batch before the next
    batch can begin. A new task therefore starts a new UI step only when:

    - it has never been seen before, and
    - no tasks from the current batch remain active.

    Interrupted tasks keep their task_id -> step assignment. If the same native
    task ID starts again after ``Command(resume=...)``, it resumes the existing
    UI row and step rather than creating a new one.
    """

    def __init__(self) -> None:
        self._next_step = 0
        self._current_step = 0
        self._active: set[str] = set()
        self._task_steps: dict[str, int] = {}

    def start(self, task_id: str) -> int:
        existing = self._task_steps.get(task_id)

        if existing is not None:
            self._active.add(task_id)
            return existing

        if not self._active:
            self._next_step += 1
            self._current_step = self._next_step

        self._task_steps[task_id] = self._current_step
        self._active.add(task_id)
        return self._current_step

    def finish(self, task_id: str) -> int:
        step = self._task_steps[task_id]
        self._active.discard(task_id)
        return step

    def step_for(self, task_id: str) -> int | None:
        return self._task_steps.get(task_id)


async def consume_tasks(
    channel: Any,
    *,
    case: str,
    pass_number: int,
    tracker: StepTracker,
    observations: list[TaskObservation],
) -> None:
    async for data in channel:
        if not isinstance(data, dict):
            raise TypeError(f"Unexpected task payload: {data!r}")

        task_id = str(data.get("id") or "")
        name = str(data.get("name") or "")

        if "input" in data:
            phase = "START"
            inferred_step = tracker.start(task_id)
        else:
            phase = "RESULT"
            inferred_step = tracker.finish(task_id)

        observation = TaskObservation(
            case=case,
            pass_number=pass_number,
            phase=phase,
            task_id=task_id,
            name=name,
            inferred_step=inferred_step,
            interrupts=list(data.get("interrupts") or []),
            error=data.get("error"),
        )
        observations.append(observation)

        print(
            f"TASK  "
            f"phase={phase:<6} "
            f"ui_step={inferred_step:<3} "
            f"name={name:<20} "
            f"id={task_id}"
        )

        if observation.interrupts:
            print(f"      interrupts={compact(observation.interrupts)}")

        if observation.error:
            print(f"      error={compact(observation.error)}")


async def consume_debug(
    channel: Any,
    *,
    case: str,
    pass_number: int,
    observations: list[DebugObservation],
) -> None:
    async for event in channel:
        if not isinstance(event, dict):
            continue

        event_type = event.get("type")
        if event_type not in {"task", "task_result"}:
            # Checkpoint debug events are intentionally ignored. This projection
            # exists only as an oracle for native task step numbers.
            continue

        payload = event.get("payload") or {}

        observation = DebugObservation(
            case=case,
            pass_number=pass_number,
            phase="START" if event_type == "task" else "RESULT",
            task_id=str(payload.get("id") or ""),
            name=str(payload.get("name") or ""),
            native_step=int(event["step"]),
        )
        observations.append(observation)

        print(
            f"DEBUG "
            f"phase={observation.phase:<6} "
            f"native_step={observation.native_step:<3} "
            f"name={observation.name:<20} "
            f"id={observation.task_id}"
        )


async def consume_subgraphs(
    channel: Any,
    *,
    case: str,
    pass_number: int,
    observations: list[SubgraphObservation],
) -> None:
    async for subgraph in channel:
        observation = SubgraphObservation(
            case=case,
            pass_number=pass_number,
            graph_name=getattr(subgraph, "graph_name", None),
            namespace=getattr(subgraph, "namespace", None),
        )
        observations.append(observation)

        print(
            "SUBGRAPH "
            f"name={observation.graph_name!r} "
            f"namespace={observation.namespace!r}"
        )


def starts(
    observations: list[TaskObservation],
    *,
    name: str | None = None,
) -> list[TaskObservation]:
    rows = [item for item in observations if item.phase == "START"]

    if name is not None:
        rows = [item for item in rows if item.name == name]

    return rows


def debug_starts(
    observations: list[DebugObservation],
    *,
    name: str | None = None,
) -> list[DebugObservation]:
    rows = [item for item in observations if item.phase == "START"]

    if name is not None:
        rows = [item for item in rows if item.name == name]

    return rows


def native_step_by_occurrence(
    debug: list[DebugObservation],
) -> dict[tuple[str, int], int]:
    """Map (task_id, pass_number) -> native step for start events."""

    return {
        (item.task_id, item.pass_number): item.native_step
        for item in debug
        if item.phase == "START"
    }


def partitions_match(
    tasks: list[TaskObservation],
    debug: list[DebugObservation],
) -> tuple[bool, list[dict[str, Any]]]:
    """Compare inferred grouping with native grouping.

    We compare equivalence classes, not numeric labels. MIRA's UI can display
    Step 1/2/3 even if LangGraph's internal step numbers begin elsewhere.

    Two task executions must share an inferred step iff they share a native
    debug step.
    """

    task_starts = [item for item in tasks if item.phase == "START"]
    native = native_step_by_occurrence(debug)

    rows: list[dict[str, Any]] = []

    for item in task_starts:
        key = (item.task_id, item.pass_number)
        rows.append(
            {
                "pass": item.pass_number,
                "id": item.task_id,
                "name": item.name,
                "ui_step": item.inferred_step,
                "native_step": native.get(key),
            }
        )

    if any(row["native_step"] is None for row in rows):
        return False, rows

    for i, left in enumerate(rows):
        for right in rows[i + 1 :]:
            same_inferred = left["ui_step"] == right["ui_step"]
            same_native = left["native_step"] == right["native_step"]

            if same_inferred != same_native:
                return False, rows

    return True, rows


async def run_once(
    graph: Any,
    payload: Any,
    *,
    case: str,
    tracker: StepTracker,
    pass_number: int = 1,
    config: dict[str, Any] | None = None,
    inspect_subgraphs: bool = False,
) -> tuple[
    list[TaskObservation],
    list[DebugObservation],
    list[SubgraphObservation],
    Any,
    bool,
    list[Any],
]:
    print()
    print("-" * 100)
    print(f"{case} · PASS {pass_number}")
    print("-" * 100)

    run = await graph.astream_events(
        payload,
        config=config,
        version="v3",
        transformers=[TasksTransformer, DebugTransformer],
    )

    print(f"v3 extensions: {sorted(run.extensions.keys())}")

    task_channel = run.extensions.get("tasks")
    debug_channel = run.extensions.get("debug")

    if task_channel is None:
        raise RuntimeError("Missing TasksTransformer projection")

    if debug_channel is None:
        raise RuntimeError("Missing DebugTransformer projection")

    task_observations: list[TaskObservation] = []
    debug_observations: list[DebugObservation] = []
    subgraph_observations: list[SubgraphObservation] = []

    async with run:
        consumers = [
            asyncio.create_task(
                consume_tasks(
                    task_channel,
                    case=case,
                    pass_number=pass_number,
                    tracker=tracker,
                    observations=task_observations,
                )
            ),
            asyncio.create_task(
                consume_debug(
                    debug_channel,
                    case=case,
                    pass_number=pass_number,
                    observations=debug_observations,
                )
            ),
        ]

        if inspect_subgraphs:
            consumers.append(
                asyncio.create_task(
                    consume_subgraphs(
                        run.subgraphs,
                        case=case,
                        pass_number=pass_number,
                        observations=subgraph_observations,
                    )
                )
            )

        await asyncio.gather(*consumers)

        interrupted = await run.interrupted()
        interrupts = list(await run.interrupts() or [])
        output = await run.output()

    print(f"interrupted={interrupted}")

    if interrupts:
        print(f"interrupts={compact(interrupts)}")

    print(f"output={compact(output)}")

    return (
        task_observations,
        debug_observations,
        subgraph_observations,
        output,
        interrupted,
        interrupts,
    )


# ---------------------------------------------------------------------------
# 1. Parallel
# ---------------------------------------------------------------------------


class ParallelState(TypedDict):
    events: Annotated[list[str], operator.add]


async def parallel_left(_state: ParallelState) -> dict[str, Any]:
    await asyncio.sleep(0.04)
    return {"events": ["left"]}


async def parallel_right(_state: ParallelState) -> dict[str, Any]:
    await asyncio.sleep(0.08)
    return {"events": ["right"]}


def build_parallel_graph():
    graph = StateGraph(ParallelState)
    graph.add_node("left", parallel_left)
    graph.add_node("right", parallel_right)
    graph.add_edge(START, "left")
    graph.add_edge(START, "right")
    graph.add_edge("left", END)
    graph.add_edge("right", END)
    return graph.compile()


async def probe_parallel() -> int:
    tracker = StepTracker()

    tasks, debug, _, _, _, _ = await run_once(
        build_parallel_graph(),
        {"events": []},
        case="PARALLEL",
        tracker=tracker,
    )

    left = starts(tasks, name="left")
    right = starts(tasks, name="right")
    partition_ok, comparison = partitions_match(tasks, debug)

    passed = (
        partition_ok
        and len(left) == 1
        and len(right) == 1
        and left[0].task_id != right[0].task_id
        and left[0].inferred_step == right[0].inferred_step
    )

    return 0 if report(
        "Parallel nodes: lightweight task grouping matches native LangGraph step",
        passed,
        comparison,
    ) else 1


# ---------------------------------------------------------------------------
# 2. Send() fan-out
# ---------------------------------------------------------------------------


class SendState(TypedDict):
    items: list[str]
    results: Annotated[list[str], operator.add]
    item: NotRequired[str]


def fan_out(state: SendState) -> list[Send]:
    return [
        Send("worker", {"item": item})
        for item in state["items"]
    ]


async def send_worker(state: SendState) -> dict[str, Any]:
    await asyncio.sleep(0.03)
    return {"results": [state["item"]]}


def build_send_graph():
    graph = StateGraph(SendState)
    graph.add_node("worker", send_worker)
    graph.add_conditional_edges(START, fan_out, ["worker"])
    graph.add_edge("worker", END)
    return graph.compile()


async def probe_send() -> int:
    tracker = StepTracker()

    tasks, debug, _, output, _, _ = await run_once(
        build_send_graph(),
        {
            "items": ["alpha", "beta", "gamma"],
            "results": [],
        },
        case="SEND",
        tracker=tracker,
    )

    workers = starts(tasks, name="worker")
    partition_ok, comparison = partitions_match(tasks, debug)

    passed = (
        partition_ok
        and len(workers) == 3
        and len({item.task_id for item in workers}) == 3
        and len({item.inferred_step for item in workers}) == 1
    )

    return 0 if report(
        "Send(): lightweight task grouping matches native LangGraph step",
        passed,
        {
            "comparison": comparison,
            "output": output,
        },
    ) else 1


# ---------------------------------------------------------------------------
# 3. Loop
# ---------------------------------------------------------------------------


class LoopState(TypedDict):
    count: int


async def loop_review(state: LoopState) -> dict[str, Any]:
    await asyncio.sleep(0.02)
    return {"count": state["count"] + 1}


def loop_route(state: LoopState) -> str:
    return "again" if state["count"] < 2 else "done"


def build_loop_graph():
    graph = StateGraph(LoopState)
    graph.add_node("review", loop_review)
    graph.add_edge(START, "review")
    graph.add_conditional_edges(
        "review",
        loop_route,
        {
            "again": "review",
            "done": END,
        },
    )
    return graph.compile()


async def probe_loop() -> int:
    tracker = StepTracker()

    tasks, debug, _, _, _, _ = await run_once(
        build_loop_graph(),
        {"count": 0},
        case="LOOP",
        tracker=tracker,
    )

    reviews = starts(tasks, name="review")
    partition_ok, comparison = partitions_match(tasks, debug)

    passed = (
        partition_ok
        and len(reviews) == 2
        and len({item.task_id for item in reviews}) == 2
        and len({item.inferred_step for item in reviews}) == 2
    )

    return 0 if report(
        "Loop: lightweight task grouping matches later native LangGraph step",
        passed,
        comparison,
    ) else 1


# ---------------------------------------------------------------------------
# 4. Nested graph boundary
# ---------------------------------------------------------------------------


class NestedState(TypedDict):
    events: Annotated[list[str], operator.add]


async def internal_model(_state: NestedState) -> dict[str, Any]:
    await asyncio.sleep(0.02)
    return {"events": ["internal_model"]}


async def internal_tool(_state: NestedState) -> dict[str, Any]:
    await asyncio.sleep(0.02)
    return {"events": ["internal_tool"]}


def build_child_graph():
    graph = StateGraph(NestedState)
    graph.add_node("internal_model", internal_model)
    graph.add_node("internal_tool", internal_tool)
    graph.add_edge(START, "internal_model")
    graph.add_edge("internal_model", "internal_tool")
    graph.add_edge("internal_tool", END)
    return graph.compile()


def build_nested_graph():
    child = build_child_graph()

    graph = StateGraph(NestedState)
    graph.add_node("nested_agent", child)
    graph.add_edge(START, "nested_agent")
    graph.add_edge("nested_agent", END)
    return graph.compile()


async def probe_nested() -> int:
    tracker = StepTracker()

    tasks, debug, subgraphs, _, _, _ = await run_once(
        build_nested_graph(),
        {"events": []},
        case="NESTED",
        tracker=tracker,
        inspect_subgraphs=True,
    )

    root_names = [item.name for item in starts(tasks)]
    partition_ok, comparison = partitions_match(tasks, debug)

    passed = (
        partition_ok
        and root_names == ["nested_agent"]
        and len(subgraphs) == 1
    )

    return 0 if report(
        "Nested graph: root task projection stays first-level and exposes a subgraph handle",
        passed,
        {
            "root_names": root_names,
            "subgraphs": [asdict(item) for item in subgraphs],
            "comparison": comparison,
            "note": (
                "Do not require a child task projection to replay its first "
                "START event. The subgraph handle itself is the reliable "
                "boundary; nested Inspector capture should use its normal "
                "message/tool/lifecycle surfaces."
            ),
        },
    ) else 1


# ---------------------------------------------------------------------------
# 5. HITL
# ---------------------------------------------------------------------------


class InterruptState(TypedDict):
    answer: str


async def approval_node(_state: InterruptState) -> dict[str, Any]:
    answer = interrupt(
        {
            "type": "workflow_probe",
            "question": "Approve this probe?",
        }
    )
    return {"answer": str(answer)}


def build_interrupt_graph():
    graph = StateGraph(InterruptState)
    graph.add_node("approval", approval_node)
    graph.add_edge(START, "approval")
    graph.add_edge("approval", END)
    return graph.compile(checkpointer=InMemorySaver())


async def probe_interrupt() -> int:
    graph = build_interrupt_graph()
    tracker = StepTracker()

    config = {
        "configurable": {
            "thread_id": "mira-workflow-runtime-ui-probe-v4",
        }
    }

    first_tasks, first_debug, _, _, interrupted1, interrupts1 = await run_once(
        graph,
        {"answer": ""},
        case="INTERRUPT",
        pass_number=1,
        config=config,
        tracker=tracker,
    )

    second_tasks, second_debug, _, output2, interrupted2, interrupts2 = await run_once(
        graph,
        Command(resume="approved"),
        case="INTERRUPT",
        pass_number=2,
        config=config,
        tracker=tracker,
    )

    all_tasks = [*first_tasks, *second_tasks]
    all_debug = [*first_debug, *second_debug]

    before = starts(first_tasks, name="approval")
    after = starts(second_tasks, name="approval")

    partition_ok, comparison = partitions_match(all_tasks, all_debug)

    passed = (
        interrupted1
        and bool(interrupts1)
        and not interrupted2
        and not interrupts2
        and len(before) == 1
        and len(after) == 1
        and before[0].task_id == after[0].task_id
        and before[0].inferred_step == after[0].inferred_step
        and partition_ok
    )

    if before and after:
        print()
        print("=" * 100)
        print("HITL IDENTITY")
        print("=" * 100)
        print(f"before task id : {before[0].task_id}")
        print(f"after task id  : {after[0].task_id}")
        print(f"before UI step : {before[0].inferred_step}")
        print(f"after UI step  : {after[0].inferred_step}")
        print(f"same row       : {before[0].task_id == after[0].task_id}")
        print(f"same UI step   : {before[0].inferred_step == after[0].inferred_step}")

    return 0 if report(
        "HITL: resume keeps the same native task ID and same inferred/native step",
        passed,
        {
            "comparison": comparison,
            "final_output": output2,
        },
    ) else 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> int:
    print("=" * 100)
    print("MIRA Workflow UI · final runtime correlation probe")
    print("=" * 100)
    print(f"cwd:               {os.getcwd()}")
    print(f"langgraph version: {package_version('langgraph')}")
    print(f"langchain version: {package_version('langchain')}")

    failures = 0
    failures += await probe_parallel()
    failures += await probe_send()
    failures += await probe_loop()
    failures += await probe_nested()
    failures += await probe_interrupt()

    print()
    print("=" * 100)
    print("FINAL INTERPRETATION")
    print("=" * 100)

    if failures:
        print(f"FAILED: {failures} assertion(s)")
        print("Keep the Workflow grouping model open until the mismatch is understood.")
        return 1

    print(
        """All assertions passed.

Production UI plumbing can use TasksTransformer only.

Recommended identity/grouping:

  Workflow row identity
      task["id"]

  Workflow node label
      task["name"]

  Workflow Step N
      lightweight task-batch tracker:
        - first unseen task while no tasks are active -> next Step N
        - additional starts while the batch is active -> same Step N
        - results remove tasks from the active batch
        - a previously seen task ID after HITL resume -> reuse its existing row/step

  Parallel nodes
      distinct task IDs in one Step N

  Send() fan-out
      same node name + distinct task IDs in one Step N

  Loops
      same node name + new task ID in a later Step N

  HITL
      same task ID is re-emitted after resume, so RUNNING -> WAITING -> RUNNING
      can stay on one Workflow row

  Nested graph
      root task projection remains first-level only
      subgraph handle is the boundary for nested Inspector capture

DebugTransformer was used only as an oracle in this probe and is NOT needed in
the production Workflow runtime.
"""
    )

    print("=" * 100)
    print("PASS: Workflow runtime identity/grouping contract is ready to implement")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
