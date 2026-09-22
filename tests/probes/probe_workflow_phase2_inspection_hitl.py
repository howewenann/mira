"""Probe the remaining Phase 2 Workflow Inspector/HITL boundary.

Run from repository root:

    python tests/probes/probe_workflow_phase2_inspection_hitl_v2.py

Why v2 exists
-------------
The first probe consumed only the child handle's ``messages`` / ``tool_calls``
projections. That is not the complete inspection path MIRA uses today.

MIRA's real turn runner also forwards the raw v3 protocol stream into
``SubagentInspectionCoordinator.handle_protocol_event()``. That raw ``values``
projection is what recovers exact child tool-call / tool-result lifecycle across
interrupts and resumes.

This probe mirrors that existing MIRA mechanism.

Questions answered
------------------
1. Can a DeepAgents runnable used as a Workflow node be correlated to its root
   Workflow task through ``handle.trigger_call_id``?
2. Can MIRA's EXISTING ``SubagentInspectionCoordinator`` recover the child's
   tool call from raw namespaced ``values`` snapshots before an interrupt?
3. On ``Command(resume=...)``, do the same root task ID and child trigger ID
   survive?
4. Does the same ``LiveInspection`` retain the tool call, receive the resumed
   tool result, and finish with the final assistant response without duplicate
   tool calls?

Deterministic only:
- fake chat model
- one local interrupting tool
- no configured MIRA model
- no network
- no MCP
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Annotated, Any, TypedDict

from deepagents.middleware.subagents import create_sub_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.stream.transformers import TasksTransformer
from langgraph.types import Command, interrupt
from pydantic import PrivateAttr

from core.execution.inspection.live import LiveInspectionStore
from core.execution.inspection.subagents import (
    SubagentInspectionCapture,
    SubagentInspectionCoordinator,
    capture_child_streams,
)
from core.execution.streams.subagents import subagent_result


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "<unknown>"


def compact(value: Any, limit: int = 1800) -> str:
    try:
        text = json.dumps(value, default=str, ensure_ascii=False)
    except Exception:
        text = repr(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "..."


def report(label: str, passed: bool, detail: Any = None) -> bool:
    print(f"\n[{'PASS' if passed else 'FAIL'}] {label}")
    if detail is not None:
        print(compact(detail))
    return passed


@tool
async def approval_probe(value: str) -> str:
    """Pause execution and return the user's resumed value."""
    resumed = interrupt(
        {
            "type": "workflow_phase2_probe",
            "question": "Approve the deterministic probe?",
            "value": value,
        }
    )
    return f"APPROVED:{resumed!r}"


class ProbeChatModel(BaseChatModel):
    """Call approval_probe once, then return its ToolMessage as final text."""

    _bound_tool_names: list[str] = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "workflow-phase2-probe"

    def bind_tools(
        self,
        tools: Any,
        *,
        tool_choice: Any | None = None,
        **kwargs: Any,
    ) -> "ProbeChatModel":
        del tool_choice, kwargs
        names: list[str] = []
        for item in tools:
            if isinstance(item, dict):
                function = item.get("function")
                if isinstance(function, dict):
                    names.append(str(function.get("name") or ""))
                else:
                    names.append(str(item.get("name") or ""))
            else:
                names.append(
                    str(
                        getattr(item, "name", None)
                        or getattr(item, "__name__", None)
                        or type(item).__name__
                    )
                )
        self._bound_tool_names = names
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs

        tool_result = next(
            (
                message
                for message in reversed(messages)
                if isinstance(message, ToolMessage)
            ),
            None,
        )

        if tool_result is not None:
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(content=f"FINAL:{tool_result.content}")
                    )
                ]
            )

        if "approval_probe" not in self._bound_tool_names:
            raise RuntimeError("approval_probe was not bound to the probe model")

        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "approval_probe",
                                "args": {"value": "phase2"},
                                "id": "approval-call",
                                "type": "tool_call",
                            }
                        ],
                    )
                )
            ]
        )


class WorkflowState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def build_workflow():
    agent = create_sub_agent(
        {
            "name": "probe-agent",
            "description": "Deterministic Workflow Phase 2 probe agent",
            "system_prompt": "Run the probe.",
            "model": ProbeChatModel(),
            "tools": [approval_probe],
        }
    )

    graph = StateGraph(WorkflowState)
    graph.add_node("agent", agent)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    return graph.compile(checkpointer=InMemorySaver())


@dataclass
class RootTask:
    phase: str
    task_id: str
    name: str
    interrupted: bool


@dataclass
class ChildObservation:
    trigger_call_id: str
    graph_name: str
    path: tuple[str, ...]
    inspection_id: str
    row_id: str
    first_start: bool
    status: str


@dataclass
class PassObservation:
    interrupted: bool
    tasks: list[RootTask]
    children: list[ChildObservation]
    output: Any
    protocol_events: int


async def consume_tasks(
    channel: Any,
    observations: list[RootTask],
    inspection: SubagentInspectionCoordinator,
) -> None:
    async for payload in channel:
        if not isinstance(payload, dict):
            raise TypeError(f"Unexpected task payload: {payload!r}")

        phase = "START" if "input" in payload else "RESULT"
        task_id = str(payload.get("id") or "")
        name = str(payload.get("name") or "")
        interrupts = list(payload.get("interrupts") or [])

        item = RootTask(
            phase=phase,
            task_id=task_id,
            name=name,
            interrupted=bool(interrupts),
        )
        observations.append(item)

        # A Workflow root task is the exact row identity we want the child
        # namespace to bind to, so make it known before values snapshots arrive.
        if phase == "START" and task_id:
            inspection.register_standalone_task(
                task_id,
                "deterministic Phase 2 probe",
            )

        print(
            "ROOT "
            f"{phase:<6} "
            f"name={name:<12} "
            f"id={task_id} "
            f"interrupt={bool(interrupts)}"
        )


async def consume_protocol(
    run: Any,
    inspection: SubagentInspectionCoordinator,
    counter: list[int],
) -> None:
    """Mirror MIRA runner's raw protocol observation path.

    ``consume_live_tool_errors`` in MIRA forwards every raw protocol event into
    ``inspection.handle_protocol_event(event)``. We do only that part here.
    """
    async for event in run:
        counter[0] += 1
        inspection.handle_protocol_event(event)


async def consume_child(
    handle: Any,
    inspection: SubagentInspectionCoordinator,
    store: LiveInspectionStore,
) -> ChildObservation:
    trigger_call_id = str(getattr(handle, "trigger_call_id", "") or "")
    path = tuple(str(part) for part in (getattr(handle, "path", ()) or ()))
    graph_name = str(getattr(handle, "graph_name", "") or "")

    if not trigger_call_id:
        raise RuntimeError(f"Child handle has no trigger_call_id: path={path!r}")

    inspection_id, row_id, first_start = inspection.standalone_started(
        handle,
        graph_name or "agent",
        "deterministic Phase 2 probe",
    )

    if not inspection_id:
        raise RuntimeError(
            f"No LiveInspection allocated for child trigger={trigger_call_id}"
        )

    capture = SubagentInspectionCapture(store, inspection_id)

    values = await asyncio.gather(
        capture_child_streams(
            handle,
            capture,
            tool_renderer=capture,
        ),
        subagent_result(handle),
    )

    result = values[-1]
    status = str(getattr(handle, "status", "") or "")

    if status != "interrupted":
        capture.ensure_final_response(str(result))

    print(
        "CHILD "
        f"graph={graph_name:<12} "
        f"trigger={trigger_call_id} "
        f"row_id={row_id} "
        f"inspection={inspection_id} "
        f"first={first_start} "
        f"status={status:<12} "
        f"path={path!r}"
    )

    return ChildObservation(
        trigger_call_id=trigger_call_id,
        graph_name=graph_name,
        path=path,
        inspection_id=inspection_id,
        row_id=row_id,
        first_start=first_start,
        status=status,
    )


async def consume_children(
    channel: Any,
    inspection: SubagentInspectionCoordinator,
    store: LiveInspectionStore,
    observations: list[ChildObservation],
) -> None:
    async for handle in channel:
        observations.append(await consume_child(handle, inspection, store))


async def run_pass(
    graph: Any,
    payload: Any,
    *,
    config: dict[str, Any],
    store: LiveInspectionStore,
    inspection: SubagentInspectionCoordinator,
    label: str,
) -> PassObservation:
    print()
    print("=" * 100)
    print(label)
    print("=" * 100)

    inspection.begin_pass()

    run = await graph.astream_events(
        payload,
        config=config,
        version="v3",
        transformers=[TasksTransformer],
    )

    tasks_channel = run.extensions.get("tasks")
    if tasks_channel is None:
        raise RuntimeError("TasksTransformer did not expose a tasks extension.")

    tasks: list[RootTask] = []
    children: list[ChildObservation] = []
    protocol_counter = [0]
    output_box: dict[str, Any] = {}

    async def collect_output() -> None:
        output_box["value"] = await run.output()

    async with run:
        await asyncio.gather(
            consume_protocol(run, inspection, protocol_counter),
            consume_tasks(tasks_channel, tasks, inspection),
            consume_children(run.subgraphs, inspection, store, children),
            collect_output(),
        )
        interrupted = await run.interrupted()

    output = output_box.get("value")

    print(f"PASS interrupted={interrupted}")
    print(f"PASS raw_protocol_events={protocol_counter[0]}")
    print(f"PASS output={compact(output)}")

    return PassObservation(
        interrupted=bool(interrupted),
        tasks=tasks,
        children=children,
        output=output,
        protocol_events=protocol_counter[0],
    )


def task_start(observation: PassObservation) -> RootTask:
    starts = [
        item
        for item in observation.tasks
        if item.phase == "START" and item.name == "agent"
    ]
    if len(starts) != 1:
        raise RuntimeError(f"Expected exactly one root agent START, got {starts!r}")
    return starts[0]


def print_inspection(store: LiveInspectionStore, inspection_id: str) -> None:
    item = store.get(inspection_id)

    print()
    print("INSPECTION")
    print("-" * 100)

    if item is None:
        print("<missing>")
        return

    print(
        f"id={item.id!r} "
        f"type={item.inspection_type!r} "
        f"status={item.status!r}"
    )

    for index, event in enumerate(item.events, 1):
        print(
            f"{index:02d}. "
            f"kind={event.kind:<11} "
            f"name={event.name!r} "
            f"call_id={event.call_id!r} "
            f"text={event.text!r} "
            f"args={event.args!r}"
        )


async def main() -> int:
    print("=" * 100)
    print("MIRA Workflow Phase 2 · Inspector + HITL boundary probe v2")
    print("=" * 100)
    print(f"langgraph:  {package_version('langgraph')}")
    print(f"langchain:  {package_version('langchain')}")
    print(f"deepagents: {package_version('deepagents')}")

    graph = build_workflow()
    store = LiveInspectionStore()
    inspection = SubagentInspectionCoordinator(store)

    config = {
        "configurable": {
            "thread_id": "workflow-phase2-probe-v2",
        }
    }

    first = await run_pass(
        graph,
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Run the deterministic probe.",
                }
            ]
        },
        config=config,
        store=store,
        inspection=inspection,
        label="PASS 1 · EXPECT INTERRUPT",
    )

    first_start = task_start(first)
    if len(first.children) != 1:
        raise RuntimeError(
            f"Expected one child handle in pass 1, got {first.children!r}"
        )

    first_child = first.children[0]
    first_inspection = store.get(first_child.inspection_id)

    print_inspection(store, first_child.inspection_id)

    failures = 0

    failures += not report("Pass 1 interrupts", first.interrupted)

    failures += not report(
        "Raw protocol stream was actually observed",
        first.protocol_events > 0,
        {"events": first.protocol_events},
    )

    failures += not report(
        "Root Workflow task correlates to child trigger_call_id",
        first_start.task_id == first_child.trigger_call_id == first_child.row_id,
        {
            "root_task_id": first_start.task_id,
            "child_trigger_call_id": first_child.trigger_call_id,
            "coordinator_row_id": first_child.row_id,
        },
    )

    first_tool_calls = (
        [
            event
            for event in first_inspection.events
            if event.kind == "tool_call" and event.call_id == "approval-call"
        ]
        if first_inspection is not None
        else []
    )

    failures += not report(
        "Existing MIRA coordinator recovers interrupted child tool call",
        len(first_tool_calls) == 1,
        [
            {
                "kind": event.kind,
                "name": event.name,
                "call_id": event.call_id,
                "args": event.args,
            }
            for event in (first_inspection.events if first_inspection else [])
        ],
    )

    failures += not report(
        "Interrupted inspection remains live for resume",
        first_inspection is not None and first_inspection.status == "RUNNING",
        {
            "inspection_id": first_child.inspection_id,
            "status": first_inspection.status if first_inspection else None,
        },
    )

    second = await run_pass(
        graph,
        Command(
            resume={
                "approved": True,
                "source": "phase2-probe-v2",
            }
        ),
        config=config,
        store=store,
        inspection=inspection,
        label="PASS 2 · RESUME",
    )

    second_start = task_start(second)
    if len(second.children) != 1:
        raise RuntimeError(
            f"Expected one child handle in pass 2, got {second.children!r}"
        )

    second_child = second.children[0]
    final_inspection = store.get(second_child.inspection_id)

    print_inspection(store, second_child.inspection_id)

    failures += not report(
        "Resume completes without another interrupt",
        not second.interrupted,
    )

    failures += not report(
        "LangGraph reuses the same root Workflow task ID across resume",
        first_start.task_id == second_start.task_id,
        {
            "pass_1": first_start.task_id,
            "pass_2": second_start.task_id,
        },
    )

    failures += not report(
        "Resumed child still correlates to that same Workflow task",
        first_child.trigger_call_id
        == second_child.trigger_call_id
        == first_start.task_id,
        {
            "pass_1_trigger": first_child.trigger_call_id,
            "pass_2_trigger": second_child.trigger_call_id,
            "root_task_id": first_start.task_id,
        },
    )

    failures += not report(
        "Coordinator reuses the same LiveInspection identity",
        first_child.inspection_id == second_child.inspection_id,
        {
            "pass_1": first_child.inspection_id,
            "pass_2": second_child.inspection_id,
            "pass_2_first_start": second_child.first_start,
        },
    )

    failures += not report(
        "Resume is recognized as the same child rather than a new child",
        first_child.first_start and not second_child.first_start,
        {
            "pass_1_first_start": first_child.first_start,
            "pass_2_first_start": second_child.first_start,
        },
    )

    if final_inspection is None:
        failures += 1
        report("Final LiveInspection exists", False)
    else:
        tool_calls = [
            event
            for event in final_inspection.events
            if event.kind == "tool_call" and event.call_id == "approval-call"
        ]
        tool_results = [
            event
            for event in final_inspection.events
            if event.kind == "tool_result" and event.call_id == "approval-call"
        ]
        assistant = [
            event
            for event in final_inspection.events
            if event.kind == "assistant"
        ]

        failures += not report(
            "Interrupted tool call is not duplicated after resume",
            len(tool_calls) == 1,
            {"count": len(tool_calls)},
        )

        failures += not report(
            "Resumed tool completion lands in the same transcript",
            len(tool_results) == 1 and "APPROVED" in tool_results[0].text,
            [
                {
                    "text": item.text,
                    "call_id": item.call_id,
                }
                for item in tool_results
            ],
        )

        failures += not report(
            "Final assistant response lands in the same transcript",
            bool(assistant) and "FINAL:" in assistant[-1].text,
            [item.text for item in assistant],
        )

        failures += not report(
            "Inspection finishes DONE",
            final_inspection.status == "DONE",
            {
                "inspection_id": final_inspection.id,
                "status": final_inspection.status,
            },
        )

    print()
    print("=" * 100)
    print("INTERPRETATION")
    print("=" * 100)

    if failures:
        print(
            f"FAILED: {failures} assertion(s).\n"
            "Do not implement Phase 2 yet. The remaining failing assertion is a "
            "real boundary to resolve rather than a missing raw-protocol feed."
        )
        return 1

    print(
        "PASS.\n\n"
        "The existing MIRA inspection machinery already has the primitives "
        "needed for Workflow-agent inspection across HITL:\n\n"
        "  root Workflow task_id\n"
        "       == child.trigger_call_id\n"
        "       == coordinator standalone row_id\n"
        "            |\n"
        "            +-- one LiveInspection\n"
        "            +-- recovered tool call from namespaced values\n"
        "            +-- same identity after resume\n"
        "            +-- resumed tool result\n"
        "            +-- final assistant response\n\n"
        "Phase 2 can therefore reuse the existing inspection coordinator / store "
        "rather than inventing a second transcript pipeline."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
