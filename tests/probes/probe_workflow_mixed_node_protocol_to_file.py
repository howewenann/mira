"""Probe raw V3 protocol correlation for arbitrary/mixed Workflow nodes.

This is the follow-up to probe_workflow_mixed_node_streaming.py.

The first probe proved:
- root TasksTransformer START gives exact node input immediately;
- root RESULT gives exact node output;
- a nested MIRA agent can emit live activity before root RESULT;
- parallel root tasks correlate cleanly through trigger_call_id/path.

But it also exposed two unanswered questions:
1. A root node containing agent -> agent produced only ONE run.subgraphs child handle.
2. child.task_input was empty and the child.messages projection was not decoded
   correctly by that probe.

This probe therefore watches the RAW V3 protocol (`async for event in run`) at
the same time as root TasksTransformer events and `run.subgraphs`.

The key question is:

    Can one permanent Workflow-root Inspector consume all observable nested
    agent activity live, even when a root node invokes multiple agents?

Run from repository root:

    python tests/probes/probe_workflow_mixed_node_protocol.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from collections import defaultdict
from pprint import pformat
from pathlib import Path
from typing import Any, NotRequired, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.stream.transformers import TasksTransformer

from mira import MiraApplication, MiraContext


T0 = 0.0


def stamp() -> str:
    return f"{time.perf_counter() - T0:8.3f}s"


def compact(value: Any, limit: int = 280) -> str:
    try:
        text = pformat(value, width=120, compact=True)
    except Exception:
        text = repr(value)
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def value_field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def message_text(message: Any) -> str:
    text = (
        value_field(message, "text", "")
        or value_field(message, "content", "")
        or ""
    )
    if isinstance(text, str):
        return text
    return compact(text, 220)


def message_type(message: Any) -> str:
    return str(
        value_field(message, "type", "")
        or value_field(message, "role", "")
        or type(message).__name__
    )


def summarize_messages(messages: list[Any]) -> str:
    values: list[str] = []
    for item in messages:
        text = " ".join(message_text(item).split())
        if len(text) > 90:
            text = text[:87] + "..."
        values.append(f"{message_type(item)}:{text!r}")
    return "[" + ", ".join(values) + "]"


def namespace_owner(namespace: tuple[str, ...]) -> tuple[str, str]:
    """Return (node_name, task_id) from the first native node:task segment."""
    for part in namespace:
        name, sep, task_id = str(part).partition(":")
        if sep and task_id:
            return name, task_id
    return "", ""


def delta_summary(data: Any) -> str:
    """Best-effort decode of raw V3 `messages` protocol data."""
    if not isinstance(data, (list, tuple)) or not data:
        return compact(data)

    payload = data[0]
    if not isinstance(payload, dict):
        return compact(data)

    event = str(payload.get("event") or "")
    delta = payload.get("data")
    if isinstance(delta, dict):
        inner = delta.get("delta", delta)
        if isinstance(inner, dict):
            kind = str(inner.get("type") or "")
            text = str(
                inner.get("text")
                or inner.get("reasoning")
                or inner.get("content")
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
    left: NotRequired[str]
    right: NotRequired[str]
    chain: NotRequired[str]


def python_before(topic: str, branch: str) -> str:
    return (
        f"Branch {branch}. Topic: {topic}. "
        "Reply in exactly two short sentences. "
        "Start your first sentence with the branch name."
    )


def python_after(result: Any, branch: str) -> str:
    messages = result.get("messages", []) if isinstance(result, dict) else []
    text = message_text(messages[-1]) if messages else ""
    return f"{branch.upper()}_POSTPROCESSED: {text}"


def build_workflow(mira: Any):
    left_agent = mira.agent(
        name="probe-left-agent",
        tools=[],
        system_prompt="You are LEFT. Follow the user's requested format exactly.",
    )
    right_agent = mira.agent(
        name="probe-right-agent",
        tools=[],
        system_prompt="You are RIGHT. Follow the user's requested format exactly.",
    )
    chain_agent_1 = mira.agent(
        name="probe-chain-agent-1",
        tools=[],
        system_prompt="Reply in one short sentence beginning with ONE.",
    )
    chain_agent_2 = mira.agent(
        name="probe-chain-agent-2",
        tools=[],
        system_prompt="Reply in one short sentence beginning with TWO.",
    )

    async def mixed_left(state: State) -> dict[str, str]:
        print(stamp(), "PY left plain-python BEFORE")
        prompt = python_before(state["topic"], "left")
        print(stamp(), "PY left invoking AGENT")
        result = await left_agent.ainvoke(
            {"messages": [HumanMessage(content=prompt)]}
        )
        print(stamp(), "PY left plain-python AFTER")
        return {"left": python_after(result, "left")}

    async def mixed_right(state: State) -> dict[str, str]:
        print(stamp(), "PY right plain-python BEFORE")
        prompt = python_before(state["topic"], "right")
        print(stamp(), "PY right invoking AGENT")
        result = await right_agent.ainvoke(
            {"messages": [HumanMessage(content=prompt)]}
        )
        print(stamp(), "PY right plain-python AFTER")
        return {"right": python_after(result, "right")}

    async def agent_chain(state: State) -> dict[str, str]:
        prompt1 = (
            "Summarize these branch outputs in one short sentence.\n"
            f"LEFT: {state['left']}\n"
            f"RIGHT: {state['right']}"
        )
        print(stamp(), "PY chain invoking AGENT ONE")
        first = await chain_agent_1.ainvoke(
            {"messages": [HumanMessage(content=prompt1)]}
        )
        first_text = message_text(first["messages"][-1])

        prompt2 = (
            "Rewrite the following in one short sentence for a technical reader.\n"
            f"{first_text}"
        )
        print(stamp(), "PY chain invoking AGENT TWO")
        second = await chain_agent_2.ainvoke(
            {"messages": [HumanMessage(content=prompt2)]}
        )

        print(stamp(), "PY chain plain-python AFTER")
        return {"chain": python_after(second, "chain")}

    graph = StateGraph(State, context_schema=MiraContext)
    graph.add_node("mixed_left", mixed_left)
    graph.add_node("mixed_right", mixed_right)
    graph.add_node("agent_chain", agent_chain)

    graph.add_edge(START, "mixed_left")
    graph.add_edge(START, "mixed_right")
    graph.add_edge(["mixed_left", "mixed_right"], "agent_chain")
    graph.add_edge("agent_chain", END)

    return graph.compile()


class Observation:
    def __init__(self) -> None:
        self.root_names: dict[str, str] = {}
        self.protocol_counts: dict[tuple[str, ...], int] = defaultdict(int)
        self.value_snapshots: dict[tuple[str, ...], int] = defaultdict(int)
        self.message_events: dict[tuple[str, ...], int] = defaultdict(int)
        self.child_handles: list[tuple[str, str, tuple[str, ...]]] = []


async def consume_tasks(tasks: Any, obs: Observation) -> None:
    async for event in tasks:
        if not isinstance(event, dict):
            continue

        task_id = str(event.get("id") or "")
        name = str(event.get("name") or "")
        if "input" in event:
            obs.root_names[task_id] = name
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
            )


async def consume_subgraphs(subgraphs: Any, obs: Observation) -> None:
    n = 0
    async for child in subgraphs:
        n += 1
        graph_name = str(getattr(child, "graph_name", "") or "")
        trigger = str(getattr(child, "trigger_call_id", "") or "")
        path = tuple(str(x) for x in (getattr(child, "path", ()) or ()))
        obs.child_handles.append((graph_name, trigger, path))
        print(
            stamp(),
            f"CHILD#{n}",
            f"graph_name={graph_name!r}",
            f"trigger_call_id={trigger!r}",
            f"path={path!r}",
            f"task_input={compact(getattr(child, 'task_input', ''))}",
        )

        # Important: this probe intentionally does NOT consume child.messages.
        # The first probe showed that surface was easy to misread. Raw V3 below
        # is the source we are testing now.


async def consume_protocol(run: Any, obs: Observation) -> None:
    """Print only namespaced raw V3 events relevant to nested execution."""
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

        obs.protocol_counts[namespace] += 1
        owner_name, owner_id = namespace_owner(namespace)
        owner = f"{owner_name}:{owner_id[-8:]}" if owner_id else "<unknown>"

        if method == "values":
            data = params.get("data")
            messages = data.get("messages") if isinstance(data, dict) else None
            if isinstance(messages, list):
                obs.value_snapshots[namespace] += 1
                print(
                    stamp(),
                    "RAW VALUES  ",
                    f"owner={owner}",
                    f"ns={namespace!r}",
                    f"messages={summarize_messages(messages)}",
                )
            continue

        if method == "messages":
            obs.message_events[namespace] += 1
            print(
                stamp(),
                "RAW MESSAGE ",
                f"owner={owner}",
                f"ns={namespace!r}",
                delta_summary(params.get("data")),
            )


async def main() -> None:
    global T0
    T0 = time.perf_counter()

    print("=" * 100)
    print("MIRA Workflow mixed-node RAW V3 protocol probe")
    print("=" * 100)
    print()
    print("What matters:")
    print("  - ROOT START should appear immediately with exact input.")
    print("  - RAW VALUES/MESSAGE should appear before ROOT RESULT.")
    print("  - namespace should contain the owning root node task id.")
    print("  - agent_chain should expose BOTH agent-one and agent-two activity somehow,")
    print("    even if run.subgraphs gives only one child handle.")
    print()

    application = await MiraApplication.start(workspace=".")
    try:
        mira = application.workflows
        graph = build_workflow(mira)
        obs = Observation()

        run = await graph.astream_events(
            {
                "topic": (
                    "Whether local LLMs are useful for internal document Q&A. "
                    "Keep responses short; this is a runtime probe."
                )
            },
            config={
                "configurable": {
                    "thread_id": "probe-workflow-mixed-node-protocol",
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
        print("=" * 100)
        print("SUMMARY")
        print("=" * 100)
        print("Root tasks:")
        for task_id, name in obs.root_names.items():
            print(f"  {name:<14} {task_id}")

        print()
        print("Child handles:")
        for graph_name, trigger, path in obs.child_handles:
            print(
                f"  graph={graph_name!r} trigger={trigger!r} path={path!r}"
            )

        print()
        print("Raw namespaces:")
        for namespace in sorted(obs.protocol_counts, key=str):
            owner_name, owner_id = namespace_owner(namespace)
            print(
                " ",
                namespace,
                f"owner={owner_name!r}:{owner_id}",
                f"events={obs.protocol_counts[namespace]}",
                f"values={obs.value_snapshots[namespace]}",
                f"messages={obs.message_events[namespace]}",
            )

        print()
        print("Final output:")
        print(compact(output_box.get("value"), 1400))

        print()
        print("=" * 100)
        print("MANUAL INTERPRETATION")
        print("=" * 100)
        print(
            "1. For mixed_left/mixed_right, check that RAW events arrive between "
            "ROOT START and ROOT RESULT and carry that root task id in namespace."
        )
        print(
            "2. For agent_chain, look after 'PY chain invoking AGENT TWO'. "
            "If RAW VALUES/MESSAGE continue and show the second HumanMessage / "
            "second assistant response under the same root task namespace, that is GOOD."
        )
        print(
            "3. It is NOT necessary to get a second run.subgraphs child handle if "
            "the raw protocol still exposes the second agent's live transcript. "
            "The proposed UI owns one Inspector per root Workflow task."
        )
        print(
            "4. If agent two produces no raw namespaced events at all, the architecture "
            "needs another observation surface before implementation."
        )
        print("=" * 100)

    finally:
        await application.shutdown()


class Tee:
    """Write one stream to both the terminal and a UTF-8 log file."""

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
            asyncio.run(main())
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    print(f"Probe log written to: {log_path}")
