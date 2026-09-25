"""Probe LangGraph node Input state / Result / Output state semantics.

Run:
    python tests/probes/probe_workflow_state_boundaries.py

Production candidates:
    TasksTransformer -> node START input + node RESULT
    native run.values projection -> full graph state snapshots

Probe-only oracle:
    DebugTransformer -> native step number + ordered task/checkpoint events

Cases:
    1. sequential nodes
    2. parallel nodes
    3. Send() fan-out
    4. in-place state mutation vs returned update
"""

from __future__ import annotations

import asyncio
import copy
import json
import operator
from importlib.metadata import PackageNotFoundError, version
from typing import Annotated, Any, NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.stream.transformers import (
    DebugTransformer,
    TasksTransformer,
)
from langgraph.types import Send


def ver(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "<unknown>"


def snap(value: Any) -> Any:
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def show(value: Any) -> str:
    try:
        return json.dumps(value, default=str, ensure_ascii=False, sort_keys=True)
    except Exception:
        return repr(value)


def get_projection(run: Any, name: str) -> Any:
    value = getattr(run, name, None)
    if value is not None:
        return value
    return run.extensions.get(name)


async def consume_tasks(channel: Any, rows: list[dict[str, Any]]) -> None:
    async for event in channel:
        phase = "START" if "input" in event else "RESULT"
        value = event.get("input") if phase == "START" else event.get("result")

        row = {
            "phase": phase,
            "id": str(event.get("id") or ""),
            "name": str(event.get("name") or ""),
            "value": snap(value),
        }
        rows.append(row)

        print(
            f"TASKS  {phase:<6} "
            f"{row['name']:<12} "
            f"id={row['id']} "
            f"value={show(row['value'])}"
        )


async def consume_values(channel: Any, rows: list[Any]) -> None:
    index = 0

    async for state in channel:
        index += 1
        state = snap(state)
        rows.append(state)
        print(f"VALUES #{index:<2} state={show(state)}")


async def consume_debug(
    channel: Any,
    rows: list[dict[str, Any]],
    states_by_step: dict[int, Any],
) -> None:
    """Single ordered oracle stream. Use this for event ordering."""
    index = 0

    async for event in channel:
        index += 1
        kind = str(event.get("type") or "")
        step = int(event.get("step", -999))
        payload = event.get("payload") or {}

        if kind == "task":
            row = {
                "order": index,
                "phase": "START",
                "step": step,
                "id": str(payload.get("id") or ""),
                "name": str(payload.get("name") or ""),
                "value": snap(payload.get("input")),
            }
            rows.append(row)
            print(
                f"DEBUG #{index:<2} step={step:<2} START  "
                f"{row['name']:<12} "
                f"id={row['id']} "
                f"input={show(row['value'])}"
            )
            continue

        if kind == "task_result":
            row = {
                "order": index,
                "phase": "RESULT",
                "step": step,
                "id": str(payload.get("id") or ""),
                "name": str(payload.get("name") or ""),
                "value": snap(payload.get("result")),
            }
            rows.append(row)
            print(
                f"DEBUG #{index:<2} step={step:<2} RESULT "
                f"{row['name']:<12} "
                f"id={row['id']} "
                f"result={show(row['value'])}"
            )
            continue

        if kind == "checkpoint":
            state = snap(payload.get("values"))
            states_by_step[step] = state
            print(
                f"DEBUG #{index:<2} step={step:<2} STATE   "
                f"next={tuple(payload.get('next') or ())!r} "
                f"state={show(state)}"
            )


def print_node_mapping(
    debug_rows: list[dict[str, Any]],
    states_by_step: dict[int, Any],
) -> None:
    results: dict[str, dict[str, Any]] = {}

    for row in debug_rows:
        if row["phase"] == "RESULT":
            results[row["id"]] = row

    print()
    print("NODE MAPPING")
    print("-" * 100)

    for row in debug_rows:
        if row["phase"] != "START":
            continue

        result = results.get(row["id"])
        output_state = states_by_step.get(row["step"], "<NO SAME-STEP STATE>")

        print(f"{row['name']}  step={row['step']}  id={row['id']}")
        print(f"  Input state : {show(row['value'])}")
        print(
            "  Result      : "
            + (show(result["value"]) if result else "<MISSING>")
        )
        print(f"  Output state: {show(output_state)}")
        print()


async def run_case(name: str, graph: Any, payload: Any) -> dict[str, Any]:
    print()
    print("=" * 100)
    print(name)
    print("=" * 100)
    print(f"graph input: {show(payload)}")
    print()

    run = await graph.astream_events(
        payload,
        version="v3",
        transformers=[
            TasksTransformer,
            DebugTransformer,
        ],
    )

    tasks_rows: list[dict[str, Any]] = []
    values_rows: list[Any] = []
    debug_rows: list[dict[str, Any]] = []
    states_by_step: dict[int, Any] = {}

    async with run:
        await asyncio.gather(
            consume_tasks(get_projection(run, "tasks"), tasks_rows),
            consume_values(get_projection(run, "values"), values_rows),
            consume_debug(
                get_projection(run, "debug"),
                debug_rows,
                states_by_step,
            ),
        )
        output = await run.output()

    print(f"\nrun.output() = {show(output)}")
    print_node_mapping(debug_rows, states_by_step)

    return {
        "tasks": tasks_rows,
        "values": values_rows,
        "debug": debug_rows,
        "states_by_step": states_by_step,
        "output": output,
    }


# ---------------------------------------------------------------------------
# 1. Sequential
# ---------------------------------------------------------------------------


class SequentialState(TypedDict):
    seed: str
    first: NotRequired[str]
    second: NotRequired[str]


async def first(state: SequentialState) -> dict[str, Any]:
    await asyncio.sleep(0.02)
    return {"first": f"first({state['seed']})"}


async def second(state: SequentialState) -> dict[str, Any]:
    await asyncio.sleep(0.02)
    return {"second": f"second({state['first']})"}


def sequential_graph():
    graph = StateGraph(SequentialState)
    graph.add_node("first", first)
    graph.add_node("second", second)
    graph.add_edge(START, "first")
    graph.add_edge("first", "second")
    graph.add_edge("second", END)
    return graph.compile()


# ---------------------------------------------------------------------------
# 2. Parallel
# ---------------------------------------------------------------------------


class ParallelState(TypedDict):
    seed: str
    left: NotRequired[str]
    right: NotRequired[str]


async def left(state: ParallelState) -> dict[str, Any]:
    await asyncio.sleep(0.03)
    return {"left": f"left({state['seed']})"}


async def right(state: ParallelState) -> dict[str, Any]:
    # Slower on purpose so RESULT timing differs.
    await asyncio.sleep(0.08)
    return {"right": f"right({state['seed']})"}


def parallel_graph():
    graph = StateGraph(ParallelState)
    graph.add_node("left", left)
    graph.add_node("right", right)
    graph.add_edge(START, "left")
    graph.add_edge(START, "right")
    graph.add_edge("left", END)
    graph.add_edge("right", END)
    return graph.compile()


# ---------------------------------------------------------------------------
# 3. Send() fan-out
# ---------------------------------------------------------------------------


class SendState(TypedDict):
    items: list[str]
    results: Annotated[list[str], operator.add]
    item: NotRequired[str]


def route_items(state: SendState) -> list[Send]:
    sends: list[Send] = []

    for item in state["items"]:
        sends.append(Send("worker", {"item": item}))

    return sends


async def worker(state: SendState) -> dict[str, Any]:
    await asyncio.sleep(0.02)
    return {"results": [f"done:{state['item']}"]}


def send_graph():
    graph = StateGraph(SendState)
    graph.add_node("worker", worker)
    graph.add_conditional_edges(START, route_items, ["worker"])
    graph.add_edge("worker", END)
    return graph.compile()


# ---------------------------------------------------------------------------
# 4. In-place mutation
# ---------------------------------------------------------------------------


class MutationState(TypedDict):
    original: str
    mutated_only: NotRequired[str]
    returned: NotRequired[str]


async def mutate(state: MutationState) -> dict[str, Any]:
    # Deliberately mutate a field that is NOT returned.
    state["mutated_only"] = "MUTATED_IN_PLACE"

    await asyncio.sleep(0.02)

    return {"returned": "RETURNED_UPDATE"}


def mutation_graph():
    graph = StateGraph(MutationState)
    graph.add_node("mutate", mutate)
    graph.add_edge(START, "mutate")
    graph.add_edge("mutate", END)
    return graph.compile()


# ---------------------------------------------------------------------------
# Focused interpretations
# ---------------------------------------------------------------------------


def starts_for_step(
    rows: list[dict[str, Any]],
    names: set[str],
) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}

    for row in rows:
        if row["phase"] != "START":
            continue
        grouped.setdefault(row["step"], []).append(row)

    for step in sorted(grouped):
        group = grouped[step]
        group_names: set[str] = set()

        for row in group:
            group_names.add(row["name"])

        if group_names == names:
            return group

    return []


def result_for(
    rows: list[dict[str, Any]],
    task_id: str,
) -> dict[str, Any] | None:
    for row in rows:
        if row["phase"] == "RESULT" and row["id"] == task_id:
            return row
    return None


def analyze_parallel(case: dict[str, Any]) -> None:
    rows = case["debug"]
    group = starts_for_step(rows, {"left", "right"})

    print()
    print("=" * 100)
    print("PARALLEL CHECK")
    print("=" * 100)

    if len(group) != 2:
        print("FAIL: did not find left/right in one native step")
        return

    step = group[0]["step"]
    output_state = case["states_by_step"].get(step)

    for row in group:
        result = result_for(rows, row["id"])
        print(row["name"])
        print(f"  Input state : {show(row['value'])}")
        print(
            "  Result      : "
            + (show(result["value"]) if result else "<MISSING>")
        )
        print(f"  Output state: {show(output_state)}")

    same_input = group[0]["value"] == group[1]["value"]
    print(f"\nsame Input state : {same_input}")
    print(
        "same Output state: True "
        "(both rows point at the one full state checkpoint for this step)"
    )


def analyze_send(case: dict[str, Any]) -> None:
    rows = case["debug"]

    print()
    print("=" * 100)
    print("SEND() CHECK")
    print("=" * 100)

    grouped: dict[int, list[dict[str, Any]]] = {}

    for row in rows:
        if row["phase"] == "START" and row["name"] == "worker":
            grouped.setdefault(row["step"], []).append(row)

    workers: list[dict[str, Any]] = []

    for step in sorted(grouped):
        if len(grouped[step]) > 1:
            workers = grouped[step]
            break

    if not workers:
        print("FAIL: did not find the Send() workers in one native step")
        return

    step = workers[0]["step"]
    output_state = case["states_by_step"].get(step)

    for row in workers:
        result = result_for(rows, row["id"])
        print(f"worker id={row['id']}")
        print(f"  Input state : {show(row['value'])}")
        print(
            "  Result      : "
            + (show(result["value"]) if result else "<MISSING>")
        )

    print(f"shared Output state: {show(output_state)}")


def analyze_mutation(case: dict[str, Any]) -> None:
    output = case["output"]
    survived = (
        isinstance(output, dict)
        and output.get("mutated_only") == "MUTATED_IN_PLACE"
    )

    print()
    print("=" * 100)
    print("IN-PLACE MUTATION CHECK")
    print("=" * 100)
    print(f"final output: {show(output)}")
    print(f"mutated-only field survived: {survived}")


async def main() -> int:
    print("=" * 100)
    print("MIRA Workflow state-boundary probe")
    print("=" * 100)
    print(f"langgraph: {ver('langgraph')}")
    print(f"langchain: {ver('langchain')}")

    sequential = await run_case(
        "CASE 1 · SEQUENTIAL",
        sequential_graph(),
        {"seed": "alpha"},
    )

    parallel = await run_case(
        "CASE 2 · PARALLEL",
        parallel_graph(),
        {"seed": "alpha"},
    )

    send = await run_case(
        "CASE 3 · SEND FAN-OUT",
        send_graph(),
        {
            "items": ["alpha", "beta", "gamma"],
            "results": [],
        },
    )

    mutation = await run_case(
        "CASE 4 · IN-PLACE MUTATION",
        mutation_graph(),
        {"original": "keep-me"},
    )

    analyze_parallel(parallel)
    analyze_send(send)
    analyze_mutation(mutation)

    print()
    print("=" * 100)
    print("UI QUESTIONS THIS OUTPUT SETTLES")
    print("=" * 100)
    print(
        """\
1. Can Input state come directly from TasksTransformer START?
2. Can Result come directly from TasksTransformer RESULT?
3. Which full state snapshot is the correct Output state?
4. Do parallel nodes map to one shared Output state?
5. Do Send() task instances have distinct Input states?
6. Does in-place mutation survive if it is not in the returned update?

Do not use DebugTransformer in production merely because this probe uses it.
It is here only to make the runtime boundaries explicit.
"""
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
