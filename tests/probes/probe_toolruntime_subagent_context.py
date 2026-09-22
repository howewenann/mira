"""Probe MIRA-style execution context inside tools and subagents.

This probe answers four narrow questions:

1. Does an ordinary LangChain ``@tool`` receive ``ToolRuntime.context``?
2. Can that tool call another runnable exposed through ``runtime.context.tools``?
3. Can that tool call a bound subagent exposed through ``runtime.context.agents``
   while explicitly preserving the same context?
4. Does DeepAgents' native ``task`` delegation preserve that context into tools
   called inside the delegated subagent, or will MIRA need explicit propagation?

The probe uses deterministic local fake chat models. It makes no network/model
calls and does not modify MIRA production code.

Run from repository root:
    python tests/probes/probe_toolruntime_subagent_context.py

Run from tests/probes:
    python probe_toolruntime_subagent_context.py
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from deepagents import create_deep_agent
from deepagents.middleware.subagents import create_sub_agent
from langchain.tools import ToolRuntime
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool, tool


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def report(label: str, passed: bool, detail: Any = None) -> bool:
    status = "PASS" if passed else "FAIL"
    print(f"\n[{status}] {label}")
    if detail is not None:
        print(detail)
    return passed


def tool_message_ids(messages: list[BaseMessage]) -> set[str]:
    return {
        str(message.tool_call_id)
        for message in messages
        if isinstance(message, ToolMessage) and message.tool_call_id
    }


def last_tool_content(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            return str(message.content)
    return ""


# ---------------------------------------------------------------------------
# Deterministic tool-calling model
# ---------------------------------------------------------------------------


class ProbeModel(BaseChatModel):
    """Tiny deterministic model used only by this probe."""

    role: str
    bound_tool_names: list[str] = []

    @property
    def _llm_type(self) -> str:
        return "mira-toolruntime-context-probe"

    def bind_tools(
        self,
        tools: Any,
        *,
        tool_choice: Any | None = None,
        **kwargs: Any,
    ) -> "ProbeModel":
        del tool_choice, kwargs
        clone = self.model_copy(deep=True)
        clone.bound_tool_names = [_tool_name(item) for item in tools]
        return clone

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        ids = tool_message_ids(messages)

        if self.role == "outer":
            if "call_outer" not in ids:
                message = AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "outer_tool",
                            "args": {"value": "probe-value"},
                            "id": "call_outer",
                            "type": "tool_call",
                        }
                    ],
                )
            else:
                message = AIMessage(
                    content=f"outer-parent-final:{last_tool_content(messages)}"
                )
            return ChatResult(generations=[ChatGeneration(message=message)])

        if self.role == "task-parent":
            if "call_task" not in ids:
                message = AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "task",
                            "args": {
                                "subagent_type": "researcher",
                                "description": "Check whether you received the parent runtime context.",
                            },
                            "id": "call_task",
                            "type": "tool_call",
                        }
                    ],
                )
            else:
                message = AIMessage(
                    content=f"task-parent-final:{last_tool_content(messages)}"
                )
            return ChatResult(generations=[ChatGeneration(message=message)])

        if self.role == "child":
            if "call_child" not in ids:
                message = AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "child_context_tool",
                            "args": {},
                            "id": "call_child",
                            "type": "tool_call",
                        }
                    ],
                )
            else:
                message = AIMessage(
                    content=f"child-final:{last_tool_content(messages)}"
                )
            return ChatResult(generations=[ChatGeneration(message=message)])

        raise AssertionError(f"Unknown probe model role: {self.role}")


def _tool_name(item: Any) -> str:
    if isinstance(item, BaseTool):
        return item.name
    if isinstance(item, dict):
        function = item.get("function")
        if isinstance(function, dict) and function.get("name"):
            return str(function["name"])
        if item.get("name"):
            return str(item["name"])
    name = getattr(item, "__name__", None)
    return str(name) if name else type(item).__name__


# ---------------------------------------------------------------------------
# Context + probe tools
# ---------------------------------------------------------------------------


@dataclass
class ProbeContext:
    marker: str
    tools: dict[str, Any] = field(default_factory=dict)
    agents: dict[str, Any] = field(default_factory=dict)


captures: dict[str, Any] = {}


@tool
async def inner_echo(text: str) -> str:
    """Echo text from a nested runnable call."""
    captures["inner_echo_called"] = True
    return f"inner:{text}"


@tool
async def child_context_tool(runtime: ToolRuntime) -> str:
    """Report the runtime context visible inside a subagent tool."""
    context = runtime.context
    captures.setdefault("child_contexts", []).append(context)

    if context is None:
        return "child-context:none"

    marker = getattr(context, "marker", None)
    return f"child-context:{marker}"


class BoundProbeAgent:
    """Probe-only stand-in for the proposed MIRA bound-agent wrapper.

    It intentionally performs the one thing we need to verify: invoke the
    already-compiled subagent while explicitly forwarding the current context.
    """

    def __init__(self, runnable: Any, context: ProbeContext) -> None:
        self._runnable = runnable
        self._context = context

    async def ainvoke(self, prompt: str) -> str:
        result = await self._runnable.ainvoke(
            {"messages": [HumanMessage(content=prompt)]},
            context=self._context,
        )
        for message in reversed(result["messages"]):
            if isinstance(message, AIMessage) and message.text:
                return message.text
        return ""


@tool
async def outer_tool(value: str, runtime: ToolRuntime) -> str:
    """Use MIRA-style tools and agents from the injected runtime context."""
    context = runtime.context
    captures["outer_context"] = context
    captures["outer_context_id"] = id(context)

    if context is None:
        return "outer-context:none"

    nested_tool = context.tools["inner_echo"]
    nested_result = await nested_tool.ainvoke({"text": value})
    captures["nested_tool_result"] = nested_result

    researcher = context.agents["researcher"]
    subagent_result = await researcher.ainvoke(
        f"Inspect this value: {nested_result}"
    )
    captures["bound_agent_result"] = subagent_result

    return f"{nested_result}|{subagent_result}"


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------


async def main() -> int:
    print("=" * 92)
    print("MIRA ToolRuntime / subagent context propagation probe")
    print("=" * 92)
    print(f"Repository root: {REPOSITORY_ROOT}")
    print(f"Probe cwd:       {Path.cwd()}")

    hard_failures = 0

    # ------------------------------------------------------------------
    # Build a raw DeepAgents-style subagent runnable that we can invoke
    # explicitly through the proposed context.agents wrapper.
    # ------------------------------------------------------------------

    child_spec = {
        "name": "researcher",
        "description": "Probe researcher",
        "system_prompt": "Call child_context_tool, then return its result.",
        "model": ProbeModel(role="child"),
        "tools": [child_context_tool],
    }
    direct_child = create_sub_agent(child_spec)

    context = ProbeContext(marker="MIRA_CONTEXT_OK")
    context.tools["inner_echo"] = inner_echo
    context.agents["researcher"] = BoundProbeAgent(direct_child, context)

    # ------------------------------------------------------------------
    # Case 1 + 2 + 3:
    # normal @tool gets ToolRuntime.context, calls context.tools and then
    # context.agents; bound agent explicitly forwards the same context.
    # ------------------------------------------------------------------

    outer_agent = create_deep_agent(
        model=ProbeModel(role="outer"),
        tools=[outer_tool],
        context_schema=ProbeContext,
    )

    outer_result = await outer_agent.ainvoke(
        {"messages": [HumanMessage(content="Run the outer tool.")]},
        context=context,
    )

    hard_failures += not report(
        "ToolRuntime is injected but hidden from the model-facing @tool schema",
        "runtime" not in outer_tool.args and "value" in outer_tool.args,
        {"tool_args": outer_tool.args},
    )

    hard_failures += not report(
        "Ordinary @tool receives the exact LangGraph runtime.context object",
        captures.get("outer_context") is context,
        {
            "expected_id": id(context),
            "actual_id": captures.get("outer_context_id"),
            "marker": getattr(captures.get("outer_context"), "marker", None),
        },
    )

    hard_failures += not report(
        "@tool can invoke another runnable exposed through runtime.context.tools",
        captures.get("inner_echo_called") is True
        and captures.get("nested_tool_result") == "inner:probe-value",
        captures.get("nested_tool_result"),
    )

    explicit_child_context = (
        captures.get("child_contexts", [])[-1]
        if captures.get("child_contexts")
        else None
    )

    hard_failures += not report(
        "@tool can invoke a bound subagent through runtime.context.agents",
        isinstance(captures.get("bound_agent_result"), str)
        and "child-context:MIRA_CONTEXT_OK" in captures["bound_agent_result"],
        captures.get("bound_agent_result"),
    )

    hard_failures += not report(
        "Explicit bound-agent invocation preserves the exact same context into the subagent's @tool",
        explicit_child_context is context,
        {
            "expected_id": id(context),
            "actual_id": id(explicit_child_context) if explicit_child_context is not None else None,
            "marker": getattr(explicit_child_context, "marker", None),
        },
    )

    # Preserve the explicit-path capture, then isolate the native task result.
    captures["explicit_child_context"] = explicit_child_context
    captures["child_contexts"] = []

    # ------------------------------------------------------------------
    # Case 4:
    # Native DeepAgents task delegation. This is diagnostic, not a hard
    # expectation: either outcome tells us exactly what MIRA must implement.
    # ------------------------------------------------------------------

    task_parent = create_deep_agent(
        model=ProbeModel(role="task-parent"),
        subagents=[
            {
                "name": "researcher",
                "description": "Probe researcher",
                "system_prompt": "Call child_context_tool, then return its result.",
                "model": ProbeModel(role="child"),
                "tools": [child_context_tool],
            }
        ],
        context_schema=ProbeContext,
    )

    task_result = await task_parent.ainvoke(
        {"messages": [HumanMessage(content="Delegate to researcher.")]},
        context=context,
    )

    native_child_context = (
        captures.get("child_contexts", [])[-1]
        if captures.get("child_contexts")
        else None
    )
    native_preserved = native_child_context is context

    report(
        "Native DeepAgents task delegation preserves parent runtime.context into the subagent tool",
        native_preserved,
        {
            "expected_id": id(context),
            "actual_id": id(native_child_context) if native_child_context is not None else None,
            "marker": getattr(native_child_context, "marker", None),
            "NOTE": (
                "PASS => MIRA can rely on native task propagation."
                if native_preserved
                else "FAIL here is diagnostic, not probe failure: MIRA must explicitly forward/bind context for subagent execution."
            ),
        },
    )

    print("\n" + "=" * 92)
    print("INTERPRETATION")
    print("=" * 92)

    if hard_failures:
        print(
            "A required baseline failed. Do not implement the proposed execution API "
            "until the failing baseline is understood."
        )
    else:
        print(
            "Required baseline PASS: normal LangChain @tool functions can consume a "
            "MIRA execution surface through ToolRuntime.context."
        )
        print(
            "Required baseline PASS: a tool can call both runtime.context.tools[...] "
            "and runtime.context.agents[...] without exposing runtime to the model."
        )
        if native_preserved:
            print(
                "Native DeepAgents task propagation also preserved the same context. "
                "No extra child-context bridge is required for this installed stack."
            )
        else:
            print(
                "Native DeepAgents task propagation did NOT preserve the context. "
                "The MIRA bound-agent/task bridge must explicitly carry MiraContext "
                "into the delegated subagent invocation."
            )
            print(
                "The explicit bound-agent case already proved that forwarding context "
                "at invocation time is sufficient."
            )

    print("\nOuter result:")
    print(outer_result["messages"][-1].text)

    print("\nNative task result:")
    print(task_result["messages"][-1].text)

    print("\n" + "=" * 92)
    print("SUMMARY")
    print("=" * 92)

    if hard_failures:
        print(f"FAILED: {hard_failures} required check(s)")
        return 1

    if native_preserved:
        print("PASS: ToolRuntime + nested tools + subagents + native context propagation all work")
    else:
        print("PASS: ToolRuntime + nested tools + subagents work; explicit child-context propagation is required")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
