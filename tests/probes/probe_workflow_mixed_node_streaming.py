"""Probe native LangGraph observability for arbitrary/mixed Workflow nodes.

Purpose
-------
Answer the architecture question:

    A Workflow node is an arbitrary Python execution boundary.
    MIRA always knows the node input/output.
    MIRA may additionally observe nested LangGraph/LangChain activity live.

This probe deliberately uses nodes shaped like:

    Python -> MIRA agent -> Python

and:

    MIRA agent -> MIRA agent -> Python

It does NOT import MIRA's Workflow UI/coordinator. It observes native LangGraph
V3 runtime surfaces directly so the result is not biased by current UI logic.

Run from the MIRA repo root:

    python tests/probes/probe_workflow_mixed_node_streaming.py
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from pprint import pformat
from typing import Any, NotRequired, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.stream.transformers import TasksTransformer

from mira import MiraApplication, MiraContext


T0 = 0.0


def stamp() -> str:
    return f"{time.perf_counter() - T0:8.3f}s"


def one_line(value: Any, limit: int = 240) -> str:
    """Compact diagnostics without throwing away the underlying object."""
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


def message_summary(value: Any) -> str:
    """Best-effort summary only for console readability."""
    message = field(value, "message", value)
    role = field(message, "type", "") or field(message, "role", "") or type(message).__name__
    text = field(message, "text", "") or field(message, "content", "") or ""
    if isinstance(text, list):
        text = one_line(text, 160)
    else:
        text = " ".join(str(text).split())
        if len(text) > 160:
            text = text[:157] + "..."
    return f"{role}: {text}"


class State(TypedDict):
    topic: str
    left: NotRequired[str]
    right: NotRequired[str]
    chain: NotRequired[str]


def python_before(topic: str, branch: str) -> str:
    # Intentionally plain Python. The runtime should NOT invent an event for it.
    return (
        f"Branch {branch}. Topic: {topic}. "
        "Reply in exactly two short sentences. "
        "Start your first sentence with the branch name."
    )


def python_after(result: Any, branch: str) -> str:
    # Intentionally plain Python. Again: no nested runtime event is expected.
    messages = result.get("messages", []) if isinstance(result, dict) else []
    text = ""
    if messages:
        last = messages[-1]
        text = getattr(last, "text", "") or getattr(last, "content", "") or str(last)
    return f"{branch.upper()}_POSTPROCESSED: {text}"


async def drain_tasks(tasks: AsyncIterator[Any]) -> None:
    """Print exact root TasksTransformer lifecycle as it happens."""
    async for event in tasks:
        if not isinstance(event, dict):
            print(stamp(), "ROOT ?", one_line(event))
            continue

        task_id = str(event.get("id") or "")
        name = str(event.get("name") or "")
        if "input" in event:
            print(
                stamp(),
                "ROOT START ",
                f"name={name!r}",
                f"id={task_id}",
                f"input={one_line(event['input'])}",
            )
        else:
            print(
                stamp(),
                "ROOT RESULT",
                f"name={name!r}",
                f"id={task_id}",
                f"result={one_line(event.get('result'))}",
                f"error={one_line(event.get('error'))}",
                f"interrupts={one_line(event.get('interrupts'))}",
            )


async def drain_child_messages(child: Any, label: str) -> None:
    stream = getattr(child, "messages", None)
    if stream is None:
        print(stamp(), f"{label} messages=<not exposed>")
        return
    try:
        async for event in stream:
            print(stamp(), f"{label} MESSAGE", message_summary(event))
    except Exception as exc:
        print(stamp(), f"{label} MESSAGE_STREAM_ERROR", repr(exc))


async def drain_child_tools(child: Any, label: str) -> None:
    stream = getattr(child, "tool_calls", None)
    if stream is None:
        print(stamp(), f"{label} tool_calls=<not exposed>")
        return
    try:
        async for event in stream:
            print(stamp(), f"{label} TOOL", one_line(event))
    except Exception as exc:
        print(stamp(), f"{label} TOOL_STREAM_ERROR", repr(exc))


async def child_output(child: Any, label: str) -> None:
    try:
        output = getattr(child, "output", None)
        if callable(output) and not hasattr(output, "__await__") and not hasattr(output, "__aiter__"):
            output = output()

        if hasattr(output, "__await__"):
            output = await output
        elif hasattr(output, "__aiter__"):
            chunks = []
            async for chunk in output:
                chunks.append(chunk)
            output = chunks

        print(stamp(), f"{label} OUTPUT", one_line(output))
    except Exception as exc:
        print(stamp(), f"{label} OUTPUT_ERROR", repr(exc))


async def drain_subgraphs(subgraphs: AsyncIterator[Any]) -> None:
    """Observe every native child handle and its correlation fields live."""
    child_no = 0
    child_consumers: list[asyncio.Task[Any]] = []

    try:
        async for child in subgraphs:
            child_no += 1
            label = f"CHILD#{child_no}"

            graph_name = str(getattr(child, "graph_name", "") or "")
            trigger_call_id = str(getattr(child, "trigger_call_id", "") or "")
            path = getattr(child, "path", ())
            status = str(getattr(child, "status", "") or "")
            task_input = getattr(child, "task_input", "")

            print()
            print(
                stamp(),
                label,
                "DISCOVERED",
                f"graph_name={graph_name!r}",
                f"trigger_call_id={trigger_call_id!r}",
                f"path={path!r}",
                f"status={status!r}",
            )
            print(stamp(), label, "task_input=", one_line(task_input))

            # Drain exposed child projections concurrently. If nested activity is
            # live, these lines should arrive BEFORE the owning ROOT RESULT line.
            child_consumers.extend(
                [
                    asyncio.create_task(drain_child_messages(child, label)),
                    asyncio.create_task(drain_child_tools(child, label)),
                    asyncio.create_task(child_output(child, label)),
                ]
            )
            await asyncio.sleep(0)
    finally:
        if child_consumers:
            await asyncio.gather(*child_consumers, return_exceptions=True)


def build_workflow(mira: Any):
    # No tools: this probe is about nesting/correlation, not approvals.
    left_agent = mira.agent(
        name="probe-left-agent",
        tools=[],
        system_prompt="You are the LEFT probe agent. Follow the user's requested format exactly.",
    )
    right_agent = mira.agent(
        name="probe-right-agent",
        tools=[],
        system_prompt="You are the RIGHT probe agent. Follow the user's requested format exactly.",
    )
    chain_agent_1 = mira.agent(
        name="probe-chain-agent-1",
        tools=[],
        system_prompt="You are chain agent ONE. Reply in one short sentence beginning with ONE.",
    )
    chain_agent_2 = mira.agent(
        name="probe-chain-agent-2",
        tools=[],
        system_prompt="You are chain agent TWO. Reply in one short sentence beginning with TWO.",
    )

    async def mixed_left(state: State) -> dict[str, str]:
        print(stamp(), "PYTHON left BEFORE begin")
        prompt = python_before(state["topic"], "left")
        print(stamp(), "PYTHON left BEFORE end; invoking agent")

        result = await left_agent.ainvoke({"messages": [HumanMessage(content=prompt)]})

        print(stamp(), "PYTHON left agent returned; AFTER begin")
        final = python_after(result, "left")
        print(stamp(), "PYTHON left AFTER end")
        return {"left": final}

    async def mixed_right(state: State) -> dict[str, str]:
        print(stamp(), "PYTHON right BEFORE begin")
        prompt = python_before(state["topic"], "right")
        print(stamp(), "PYTHON right BEFORE end; invoking agent")

        result = await right_agent.ainvoke({"messages": [HumanMessage(content=prompt)]})

        print(stamp(), "PYTHON right agent returned; AFTER begin")
        final = python_after(result, "right")
        print(stamp(), "PYTHON right AFTER end")
        return {"right": final}

    async def agent_chain(state: State) -> dict[str, str]:
        # This root node deliberately contains agent -> agent -> plain Python.
        # The architecture must not assume one root node == one child agent.
        prompt1 = (
            "Summarize these two branch outputs in one short sentence:\n"
            f"LEFT: {state['left']}\n"
            f"RIGHT: {state['right']}"
        )
        print(stamp(), "PYTHON chain invoking agent ONE")
        first = await chain_agent_1.ainvoke({"messages": [HumanMessage(content=prompt1)]})
        first_text = first["messages"][-1].text

        prompt2 = (
            "Rewrite this in one short sentence for a technical reader:\n"
            f"{first_text}"
        )
        print(stamp(), "PYTHON chain invoking agent TWO")
        second = await chain_agent_2.ainvoke({"messages": [HumanMessage(content=prompt2)]})

        print(stamp(), "PYTHON chain final plain-Python postprocess")
        return {"chain": python_after(second, "chain")}

    graph = StateGraph(State, context_schema=MiraContext)
    graph.add_node("mixed_left", mixed_left)
    graph.add_node("mixed_right", mixed_right)
    graph.add_node("agent_chain", agent_chain)

    # Two arbitrary mixed nodes in parallel.
    graph.add_edge(START, "mixed_left")
    graph.add_edge(START, "mixed_right")

    # Join, then a single root node containing agent -> agent -> Python.
    graph.add_edge(["mixed_left", "mixed_right"], "agent_chain")
    graph.add_edge("agent_chain", END)

    return graph.compile()


async def main() -> None:
    global T0
    T0 = time.perf_counter()

    print("=" * 88)
    print("MIRA / LangGraph mixed-node streaming probe")
    print("=" * 88)
    print()
    print("Questions:")
    print("  1. Does ROOT START expose exact node input immediately?")
    print("  2. Are nested agent child handles/events emitted before ROOT RESULT?")
    print("  3. Does child.trigger_call_id correlate to the owning root task id?")
    print("  4. Does this remain correct for two parallel mixed nodes?")
    print("  5. Can one root node expose multiple sequential child agents?")
    print("  6. Does ROOT RESULT expose the exact returned node update?")
    print("  7. Plain Python calls should have no invented runtime events.")
    print()

    application = await MiraApplication.start(workspace=".")
    try:
        mira = application.workflows
        graph = build_workflow(mira)

        kwargs: dict[str, Any] = {
            "config": {
                "configurable": {
                    "thread_id": "probe-workflow-mixed-node-streaming",
                }
            },
            "context": mira.context,
            "version": "v3",
            "transformers": [TasksTransformer],
        }

        payload = {
            "topic": (
                "Whether local LLMs are useful for internal document Q&A. "
                "Keep every response deliberately short because this is a runtime probe."
            )
        }

        print(stamp(), "OPENING graph.astream_events(...)")
        run = await graph.astream_events(payload, **kwargs)

        tasks = run.extensions.get("tasks")
        if tasks is None:
            raise RuntimeError("TasksTransformer projection was not exposed.")

        print(stamp(), "RUN OPEN")
        print()

        # Same native surfaces used by MIRA today:
        # - TasksTransformer root task lifecycle
        # - run.subgraphs child handles
        async with run:
            await asyncio.gather(
                drain_tasks(tasks),
                drain_subgraphs(run.subgraphs),
            )

        final_state = await run.output()
        print()
        print("=" * 88)
        print(stamp(), "FINAL GRAPH OUTPUT")
        print(one_line(final_state, limit=1000))
        print("=" * 88)
        print()
        print("Interpretation checklist:")
        print("  PASS A: ROOT START appears before the node's agent work and contains exact input.")
        print("  PASS B: CHILD/MESSAGE lines appear while the owning ROOT task is still running.")
        print("  PASS C: child trigger/path gives unambiguous public correlation to the root task.")
        print("  PASS D: parallel mixed_left/mixed_right children stay correctly correlated.")
        print("  PASS E: agent_chain exposes multiple sequential child agents under one root task.")
        print("  PASS F: ROOT RESULT carries the exact update returned by each root node.")
        print("  EXPECTED: plain Python is visible only through the explicit probe print() calls.")

    finally:
        await application.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
