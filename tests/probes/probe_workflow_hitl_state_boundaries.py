"""Probe Workflow state boundaries across LangGraph HITL resume.

Purpose
-------
Verify the exact runtime behavior needed for MIRA's Workflow node UI:

    [Input state]  [Result]  [Output state]

for one node that interrupts and later resumes.

Questions
---------
1. What Input state does TasksTransformer START expose?
2. Does the same native task ID reappear after Command(resume=...)?
3. Does the values stream emit an intermediate state while waiting?
4. What Result does the resumed node finally return?
5. What full Output state is emitted after the node completes?

Run:
    python tests/probes/probe_workflow_hitl_state_boundaries.py
"""

from __future__ import annotations

import asyncio
import copy
import json
from importlib.metadata import PackageNotFoundError, version
from typing import Any, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.stream.transformers import TasksTransformer
from langgraph.types import Command, interrupt


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
        return json.dumps(
            value,
            ensure_ascii=False,
            default=str,
            sort_keys=True,
        )
    except Exception:
        return repr(value)


def get_projection(run: Any, name: str) -> Any:
    value = getattr(run, name, None)
    if value is not None:
        return value
    return run.extensions.get(name)


class State(TypedDict):
    request: str
    answer: str


def approval_node(state: State) -> dict[str, Any]:
    answer = interrupt(
        {
            "type": "workflow_probe",
            "question": f"Approve request: {state['request']}?",
        }
    )
    return {"answer": str(answer)}


def build_graph():
    graph = StateGraph(State)
    graph.add_node("approval", approval_node)
    graph.add_edge(START, "approval")
    graph.add_edge("approval", END)
    return graph.compile(checkpointer=InMemorySaver())


async def consume_tasks(
    channel: Any,
    *,
    pass_number: int,
    rows: list[dict[str, Any]],
) -> None:
    async for event in channel:
        phase = "START" if "input" in event else "RESULT"
        value = event.get("input") if phase == "START" else event.get("result")

        row = {
            "pass": pass_number,
            "phase": phase,
            "id": str(event.get("id") or ""),
            "name": str(event.get("name") or ""),
            "value": snap(value),
            "interrupts": snap(event.get("interrupts") or []),
            "error": snap(event.get("error")),
        }
        rows.append(row)

        print(
            f"TASKS  pass={pass_number} "
            f"{phase:<6} "
            f"name={row['name']:<12} "
            f"id={row['id']} "
            f"value={show(row['value'])}"
        )

        if row["interrupts"]:
            print(f"       interrupts={show(row['interrupts'])}")

        if row["error"]:
            print(f"       error={show(row['error'])}")


async def consume_values(
    channel: Any,
    *,
    pass_number: int,
    rows: list[dict[str, Any]],
) -> None:
    index = 0

    async for state in channel:
        index += 1
        row = {
            "pass": pass_number,
            "index": index,
            "state": snap(state),
        }
        rows.append(row)

        print(
            f"VALUES pass={pass_number} "
            f"#{index:<2} "
            f"state={show(row['state'])}"
        )


async def run_pass(
    graph: Any,
    payload: Any,
    *,
    config: dict[str, Any],
    pass_number: int,
) -> dict[str, Any]:
    print()
    print("=" * 100)
    print(f"PASS {pass_number}")
    print("=" * 100)
    print(f"payload={show(payload)}")
    print()

    run = await graph.astream_events(
        payload,
        config=config,
        version="v3",
        transformers=[TasksTransformer],
    )

    tasks = get_projection(run, "tasks")
    values = get_projection(run, "values")

    if tasks is None:
        raise RuntimeError("TasksTransformer projection is unavailable.")

    if values is None:
        raise RuntimeError("Native values projection is unavailable.")

    task_rows: list[dict[str, Any]] = []
    value_rows: list[dict[str, Any]] = []

    async with run:
        await asyncio.gather(
            consume_tasks(
                tasks,
                pass_number=pass_number,
                rows=task_rows,
            ),
            consume_values(
                values,
                pass_number=pass_number,
                rows=value_rows,
            ),
        )

        interrupted = await run.interrupted()
        interrupts = list(await run.interrupts() or [])
        output = await run.output()

    print()
    print(f"interrupted={interrupted}")
    print(f"interrupts={show(interrupts)}")
    print(f"run.output()={show(output)}")

    return {
        "tasks": task_rows,
        "values": value_rows,
        "interrupted": interrupted,
        "interrupts": interrupts,
        "output": output,
    }


def first_task(
    rows: list[dict[str, Any]],
    *,
    phase: str,
) -> dict[str, Any] | None:
    for row in rows:
        if row["phase"] == phase and row["name"] == "approval":
            return row
    return None


async def main() -> int:
    print("=" * 100)
    print("MIRA Workflow HITL state-boundary probe")
    print("=" * 100)
    print(f"langgraph: {ver('langgraph')}")
    print(f"langchain: {ver('langchain')}")

    graph = build_graph()

    config = {
        "configurable": {
            "thread_id": "mira-workflow-hitl-state-boundary-probe",
        }
    }

    first = await run_pass(
        graph,
        {
            "request": "deploy",
            "answer": "",
        },
        config=config,
        pass_number=1,
    )

    second = await run_pass(
        graph,
        Command(resume="approved"),
        config=config,
        pass_number=2,
    )

    first_start = first_task(first["tasks"], phase="START")
    first_result = first_task(first["tasks"], phase="RESULT")
    second_start = first_task(second["tasks"], phase="START")
    second_result = first_task(second["tasks"], phase="RESULT")

    print()
    print("=" * 100)
    print("INTERPRETATION")
    print("=" * 100)

    print(
        "Pass 1 Input state : "
        + (
            show(first_start["value"])
            if first_start is not None
            else "<MISSING>"
        )
    )

    print(
        "Pass 1 Result      : "
        + (
            show(first_result["value"])
            if first_result is not None
            else "<NONE BEFORE INTERRUPT>"
        )
    )

    print(
        "Pass 1 values      : "
        + show([row["state"] for row in first["values"]])
    )

    print(
        "Pass 2 Input state : "
        + (
            show(second_start["value"])
            if second_start is not None
            else "<MISSING>"
        )
    )

    print(
        "Pass 2 Result      : "
        + (
            show(second_result["value"])
            if second_result is not None
            else "<MISSING>"
        )
    )

    print(
        "Pass 2 values      : "
        + show([row["state"] for row in second["values"]])
    )

    same_task_id = (
        first_start is not None
        and second_start is not None
        and first_start["id"] == second_start["id"]
    )

    print(f"same task id after resume: {same_task_id}")

    print()
    print("Expected UI questions:")
    print("  [Input state]  -> which START input should the node retain?")
    print("  [Result]       -> is there no result until the resumed node completes?")
    print("  [Output state] -> which values snapshot represents the committed state?")
    print("  WAITING        -> does pass 1 emit any new full state while interrupted?")

    failures = 0

    if not first["interrupted"]:
        print("\n[FAIL] Pass 1 did not interrupt.")
        failures += 1

    if second["interrupted"]:
        print("\n[FAIL] Pass 2 remained interrupted.")
        failures += 1

    if first_start is None:
        print("\n[FAIL] Pass 1 START was not observed.")
        failures += 1

    if second_start is None:
        print("\n[FAIL] Pass 2 START was not observed.")
        failures += 1

    if second_result is None:
        print("\n[FAIL] Pass 2 RESULT was not observed.")
        failures += 1

    if not same_task_id:
        print("\n[FAIL] Task ID changed across resume.")
        failures += 1

    if failures:
        print(f"\nFAILED: {failures} assertion(s)")
        return 1

    print("\n[PASS] HITL state-boundary probe completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
