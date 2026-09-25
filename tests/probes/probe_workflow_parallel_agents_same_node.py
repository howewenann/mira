"""Probe parallel nested MIRA agents inside ONE arbitrary Workflow node.

Question
--------
If one user-created LangGraph node does:

    Python
      -> await asyncio.gather(
             agent_a.ainvoke(...),
             agent_b.ainvoke(...),
         )
      -> Python

can MIRA's native V3 stream distinguish BOTH nested agent executions while
keeping them correlated to the SAME root Workflow task?

This is the final correlation probe before redesigning Workflow Inspector UI.

Run from the MIRA repo root:

    python tests/probes/probe_workflow_parallel_agents_same_node.py

The script writes the complete stdout + stderr to a sibling .log file.
"""

from __future__ import annotations

import asyncio
import sys
import time
from collections import defaultdict
from pathlib import Path
from pprint import pformat
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.stream.transformers import TasksTransformer

from mira import MiraApplication, MiraContext


T0 = 0.0


def stamp() -> str:
    return f"{time.perf_counter() - T0:8.3f}s"


def compact(value: Any, limit: int = 300) -> str:
    try:
        text = pformat(value, width=120, compact=True)
    except Exception:
        text = repr(value)
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def message_text(message: Any) -> str:
    value = field(message, "text", "") or field(message, "content", "") or ""
    if isinstance(value, str):
        return value
    return compact(value, 220)


def message_type(message: Any) -> str:
    return str(
        field(message, "type", "")
        or field(message, "role", "")
        or type(message).__name__
    )


def summarize_messages(messages: list[Any]) -> str:
    items: list[str] = []
    for message in messages:
        text = " ".join(message_text(message).split())
        if len(text) > 100:
            text = text[:97] + "..."
        items.append(f"{message_type(message)}:{text!r}")
    return "[" + ", ".join(items) + "]"


def namespace_owner(namespace: tuple[str, ...]) -> tuple[str, str]:
    """Return root (node_name, task_id) from first native `name:id` segment."""
    for part in namespace:
        name, sep, task_id = str(part).partition(":")
        if sep and task_id:
            return name, task_id
    return "", ""


def protocol_delta_summary(data: Any) -> str:
    if not isinstance(data, (list, tuple)) or not data:
        return compact(data)

    payload = data[0]
    if not isinstance(payload, dict):
        return compact(data)

    event = str(payload.get("event") or "")
    delta = payload.get("delta")
    if not isinstance(delta, dict):
        content = payload.get("content")
        if isinstance(content, dict):
            return (
                f"event={event!r} "
                f"type={str(content.get('type') or '')!r} "
                f"content={compact(content, 160)}"
            )
        return compact(data)

    kind = str(delta.get("type") or "")
    text = str(
        delta.get("text")
        or delta.get("reasoning")
        or delta.get("content")
        or ""
    )
    if text:
        text = text.replace("\n", "\\n")
        if len(text) > 140:
            text = text[:137] + "..."
        return f"event={event!r} type={kind!r} text={text!r}"

    return compact(data)


class State(TypedDict):
    topic: str
    combined: str


def last_text(result: Any) -> str:
    messages = result.get("messages", []) if isinstance(result, dict) else []
    return message_text(messages[-1]) if messages else ""


def build_workflow(mira: Any):
    agent_a = mira.agent(
        name="probe-parallel-agent-a",
        tools=[],
        system_prompt=(
            "You are parallel agent A. "
            "Reply in exactly one short sentence beginning with A."
        ),
    )
    agent_b = mira.agent(
        name="probe-parallel-agent-b",
        tools=[],
        system_prompt=(
            "You are parallel agent B. "
            "Reply in exactly one short sentence beginning with B."
        ),
    )

    async def parallel_agents(state: State) -> dict[str, str]:
        # Plain Python before the nested agents.
        topic = state["topic"].strip()
        prompt_a = (
            f"Topic: {topic}\n"
            "Give one concise BENEFIT of local LLMs for internal document Q&A."
        )
        prompt_b = (
            f"Topic: {topic}\n"
            "Give one concise RISK of local LLMs for internal document Q&A."
        )

        print(stamp(), "PY root BEFORE asyncio.gather")
        print(stamp(), "PY invoking AGENT A and AGENT B concurrently")

        result_a, result_b = await asyncio.gather(
            agent_a.ainvoke(
                {"messages": [HumanMessage(content=prompt_a)]}
            ),
            agent_b.ainvoke(
                {"messages": [HumanMessage(content=prompt_b)]}
            ),
        )

        # Plain Python after both agents.
        print(stamp(), "PY both agents returned; plain-python AFTER")
        combined = (
            f"A_RESULT={last_text(result_a)} | "
            f"B_RESULT={last_text(result_b)}"
        )
        return {"combined": combined}

    graph = StateGraph(State, context_schema=MiraContext)
    graph.add_node("parallel_agents", parallel_agents)
    graph.add_edge(START, "parallel_agents")
    graph.add_edge("parallel_agents", END)
    return graph.compile()


class Observation:
    def __init__(self) -> None:
        self.root_task_id = ""
        self.root_name = ""
        self.namespaces: dict[tuple[str, ...], dict[str, int]] = defaultdict(
            lambda: {"events": 0, "values": 0, "messages": 0}
        )
        self.human_prompts: dict[tuple[str, ...], list[str]] = defaultdict(list)
        self.child_handles: list[tuple[str, str, tuple[str, ...]]] = []


async def consume_tasks(tasks: Any, obs: Observation) -> None:
    async for event in tasks:
        if not isinstance(event, dict):
            continue

        task_id = str(event.get("id") or "")
        name = str(event.get("name") or "")

        if "input" in event:
            obs.root_task_id = task_id
            obs.root_name = name
            print(
                stamp(),
                "ROOT START ",
                f"name={name!r}",
                f"id={task_id}",
                f"input={compact(event['input'])}",
            )
        else:
            print(
                stamp(),
                "ROOT RESULT",
                f"name={name!r}",
                f"id={task_id}",
                f"result={compact(event.get('result'))}",
                f"error={compact(event.get('error'))}",
                f"interrupts={compact(event.get('interrupts'))}",
            )


async def consume_subgraphs(subgraphs: Any, obs: Observation) -> None:
    index = 0
    async for child in subgraphs:
        index += 1
        graph_name = str(getattr(child, "graph_name", "") or "")
        trigger = str(getattr(child, "trigger_call_id", "") or "")
        path = tuple(str(x) for x in (getattr(child, "path", ()) or ()))

        obs.child_handles.append((graph_name, trigger, path))

        print(
            stamp(),
            f"CHILD#{index}",
            f"graph_name={graph_name!r}",
            f"trigger_call_id={trigger!r}",
            f"path={path!r}",
            f"task_input={compact(getattr(child, 'task_input', ''))}",
        )


async def consume_protocol(run: Any, obs: Observation) -> None:
    """Observe namespaced raw V3 protocol while root node is still running."""
    async for event in run:
        if not isinstance(event, dict):
            continue

        method = str(event.get("method") or "")
        params = event.get("params")
        if not isinstance(params, dict):
            continue

        namespace = tuple(str(x) for x in (params.get("namespace") or ()))
        if not namespace:
            continue

        stats = obs.namespaces[namespace]
        stats["events"] += 1

        owner_name, owner_id = namespace_owner(namespace)
        owner = f"{owner_name}:{owner_id[-8:]}" if owner_id else "<unknown>"

        if method == "values":
            data = params.get("data")
            messages = data.get("messages") if isinstance(data, dict) else None
            if not isinstance(messages, list):
                continue

            stats["values"] += 1

            for message in messages:
                if message_type(message) in {"human", "user"}:
                    text = message_text(message)
                    if text and text not in obs.human_prompts[namespace]:
                        obs.human_prompts[namespace].append(text)

            print(
                stamp(),
                "RAW VALUES  ",
                f"owner={owner}",
                f"ns={namespace!r}",
                f"messages={summarize_messages(messages)}",
            )
            continue

        if method == "messages":
            stats["messages"] += 1
            print(
                stamp(),
                "RAW MESSAGE ",
                f"owner={owner}",
                f"ns={namespace!r}",
                protocol_delta_summary(params.get("data")),
            )


def evaluate(obs: Observation) -> int:
    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"root_name={obs.root_name!r}")
    print(f"root_task_id={obs.root_task_id}")

    print()
    print("Child handles:")
    if not obs.child_handles:
        print("  <none>")
    for graph_name, trigger, path in obs.child_handles:
        print(
            f"  graph={graph_name!r} "
            f"trigger={trigger!r} "
            f"path={path!r}"
        )

    print()
    print("Raw namespaces:")
    for namespace in sorted(obs.namespaces, key=str):
        owner_name, owner_id = namespace_owner(namespace)
        stats = obs.namespaces[namespace]
        prompts = obs.human_prompts.get(namespace, [])
        print(
            f"  ns={namespace!r}\n"
            f"     owner={owner_name!r}:{owner_id}\n"
            f"     events={stats['events']} "
            f"values={stats['values']} "
            f"messages={stats['messages']}\n"
            f"     human_prompts={compact(prompts, 800)}"
        )

    root_namespaces = []
    for namespace in obs.namespaces:
        _owner_name, owner_id = namespace_owner(namespace)
        if owner_id and owner_id == obs.root_task_id:
            root_namespaces.append(namespace)

    prompt_values = [
        prompt
        for namespace in root_namespaces
        for prompt in obs.human_prompts.get(namespace, [])
    ]
    saw_a = any("BENEFIT" in prompt for prompt in prompt_values)
    saw_b = any("RISK" in prompt for prompt in prompt_values)

    distinct_prompt_namespaces = {
        namespace
        for namespace in root_namespaces
        if obs.human_prompts.get(namespace)
    }

    message_namespaces = {
        namespace
        for namespace in root_namespaces
        if obs.namespaces[namespace]["messages"] > 0
    }

    all_children_correlate = bool(obs.child_handles) and all(
        trigger == obs.root_task_id
        for _name, trigger, _path in obs.child_handles
        if trigger
    )

    checks = [
        (
            "Both different nested agent prompts are observable",
            saw_a and saw_b,
            {"saw_A_benefit_prompt": saw_a, "saw_B_risk_prompt": saw_b},
        ),
        (
            "Both nested executions stay owned by the SAME root task id",
            bool(root_namespaces)
            and all(namespace_owner(ns)[1] == obs.root_task_id for ns in root_namespaces),
            root_namespaces,
        ),
        (
            "Parallel nested executions are distinguishable by namespace",
            len(distinct_prompt_namespaces) >= 2,
            distinct_prompt_namespaces,
        ),
        (
            "Live model message traffic exists for multiple nested namespaces",
            len(message_namespaces) >= 2,
            message_namespaces,
        ),
        (
            "Any exposed child handles correlate to the same root task",
            all_children_correlate,
            obs.child_handles,
        ),
    ]

    failures = 0
    print()
    print("CHECKS")
    for label, passed, detail in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {label}")
        print("      ", compact(detail, 1200))
        if not passed:
            failures += 1

    print()
    print("=" * 100)
    print("INTERPRETATION")
    print("=" * 100)

    if failures == 0:
        print(
            """PASS.

This supports the Workflow Inspector model:

    one native root task id
        =
    one permanent Workflow Inspector

The Inspector can be created immediately from TasksTransformer START.

Nested observable LangGraph/LangChain activity can then be appended live by
routing raw V3 protocol namespaces whose first `node:task_id` segment contains
that root task id.

Parallel agents inside the same arbitrary Python node do NOT require node-type
classification or Inspector rebinding. Their nested namespaces distinguish
their concurrent executions while preserving one root owner.

Plain Python inside the root node remains intentionally opaque.

The root TasksTransformer RESULT remains the exact node output boundary.
"""
        )
    else:
        print(
            f"""FAILED: {failures} check(s).

Do NOT implement the new Workflow Inspector architecture yet.

Inspect the namespace/prompt evidence above and determine which native public
surface is missing or ambiguous for parallel nested agents inside one root node.
"""
        )

    return failures


async def main() -> int:
    global T0
    T0 = time.perf_counter()

    print("=" * 100)
    print("MIRA Workflow probe · parallel nested agents inside ONE root node")
    print("=" * 100)
    print()
    print("Target shape:")
    print()
    print("  Workflow root: parallel_agents")
    print("      plain Python")
    print("          |")
    print("          +---- Agent A ----+")
    print("          |                 |")
    print("          +---- Agent B ----+   asyncio.gather")
    print("                            |")
    print("                       plain Python")
    print("                            |")
    print("                          return")
    print()
    print("We need BOTH agents live and distinguishable, while both remain owned")
    print("by the same native root Workflow task id.")
    print()

    application = await MiraApplication.start(workspace=".")
    try:
        mira = application.workflows
        graph = build_workflow(mira)
        obs = Observation()

        run = await graph.astream_events(
            {
                "topic": (
                    "Local LLMs for internal enterprise document question answering."
                ),
                "combined": "",
            },
            config={
                "configurable": {
                    "thread_id": "probe-workflow-parallel-agents-same-node",
                }
            },
            context=mira.context,
            version="v3",
            transformers=[TasksTransformer],
        )

        tasks = run.extensions.get("tasks")
        if tasks is None:
            raise RuntimeError("TasksTransformer projection missing.")

        output_box: dict[str, Any] = {}

        async def collect_output() -> None:
            output_box["value"] = await run.output()

        async with run:
            await asyncio.gather(
                consume_tasks(tasks, obs),
                consume_subgraphs(run.subgraphs, obs),
                consume_protocol(run, obs),
                collect_output(),
            )

        print()
        print("FINAL GRAPH OUTPUT")
        print(compact(output_box.get("value"), 1400))

        return evaluate(obs)
    finally:
        await application.shutdown()


class Tee:
    """Write to both terminal and UTF-8 log file."""

    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(
            bool(getattr(stream, "isatty", lambda: False)())
            for stream in self.streams
        )


if __name__ == "__main__":
    log_path = Path(__file__).with_suffix(".log")
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = Tee(original_stderr, log_file)
        try:
            print(f"Writing complete probe output to: {log_path}")
            exit_code = asyncio.run(main())
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    print(f"Probe log written to: {log_path}")
    raise SystemExit(exit_code)
