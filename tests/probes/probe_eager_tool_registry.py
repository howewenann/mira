"""Probe whether DeepAgents exposes its exact executable tool registry eagerly.

Goal
----
Determine whether the exact BaseTool instances registered for execution are
reachable *after create_deep_agent() returns but before the first model call*,
and whether those same objects later appear in ModelRequest.tools.

Run from repository root:
    python tests/probes/probe_eager_tool_registry.py

Run from tests/probes:
    python probe_eager_tool_registry.py
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from collections import deque
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import ToolNode


TARGETS = {
    "read_file",
    "write_file",
    "task",
    "mcp__probe__echo",
}


class ProbeChatModel(BaseChatModel):
    """Deterministic model that never calls tools."""

    @property
    def _llm_type(self) -> str:
        return "probe-chat-model"

    def _generate(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del messages, stop, run_manager, kwargs
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(content="PROBE_MODEL_RESPONSE")
                )
            ]
        )

    def bind_tools(
        self,
        tools: Any,
        *,
        tool_choice: Any | None = None,
        **kwargs: Any,
    ) -> "ProbeChatModel":
        del tools, tool_choice, kwargs
        return self


class CaptureLiveToolsMiddleware(AgentMiddleware):
    """Capture the exact tool objects offered at the model boundary."""

    def __init__(self) -> None:
        super().__init__()
        self.live_tools: list[BaseTool] | None = None

    async def awrap_model_call(self, request: ModelRequest, handler: Any) -> Any:
        self.live_tools = list(request.tools)
        return await handler(request)


@tool("mcp__probe__echo")
async def mcp_probe_echo(value: str) -> str:
    """MCP-shaped custom tool used only for object-identity checks."""
    return value


def tool_map(tools: list[Any]) -> dict[str, BaseTool]:
    return {
        tool.name: tool
        for tool in tools
        if isinstance(tool, BaseTool) and isinstance(getattr(tool, "name", None), str)
    }


def narrow_unwrap(start_path: str, start: Any) -> tuple[str, ToolNode] | None:
    """Search only common runnable-wrapper fields; do not crawl arbitrary objects."""
    queue: deque[tuple[str, Any, int]] = deque([(start_path, start, 0)])
    seen: set[int] = set()

    while queue:
        path, obj, depth = queue.popleft()
        if id(obj) in seen or depth > 6:
            continue
        seen.add(id(obj))

        if isinstance(obj, ToolNode):
            return path, obj

        for attr in ("bound", "runnable", "data", "node"):
            try:
                child = getattr(obj, attr)
            except Exception:
                continue
            if child is not None:
                queue.append((f"{path}.{attr}", child, depth + 1))

        try:
            steps = getattr(obj, "steps")
        except Exception:
            steps = None
        if isinstance(steps, (list, tuple)):
            for index, child in enumerate(steps):
                queue.append((f"{path}.steps[{index}]", child, depth + 1))

    return None


def find_eager_tool_node(agent: Any) -> tuple[str, ToolNode] | None:
    """Try construction-time graph surfaces, from least to most internal."""

    # CompiledStateGraph.nodes is the first and simplest construction-time surface.
    try:
        nodes = agent.nodes
        if isinstance(nodes, dict) and "tools" in nodes:
            found = narrow_unwrap("agent.nodes['tools']", nodes["tools"])
            if found:
                return found
    except Exception:
        pass

    # Public graph rendering/introspection surface.
    try:
        graph = agent.get_graph()
        nodes = getattr(graph, "nodes", None)
        if isinstance(nodes, dict) and "tools" in nodes:
            found = narrow_unwrap(
                "agent.get_graph().nodes['tools']",
                nodes["tools"],
            )
            if found:
                return found
    except Exception:
        pass

    # Diagnostic fallback only. If this is the sole path, production code should
    # not depend on it without an explicit compatibility decision.
    try:
        builder = agent.builder
        nodes = getattr(builder, "nodes", None)
        if isinstance(nodes, dict) and "tools" in nodes:
            found = narrow_unwrap(
                "agent.builder.nodes['tools']",
                nodes["tools"],
            )
            if found:
                return found
    except Exception:
        pass

    return None


def report(label: str, passed: bool, detail: Any = None) -> bool:
    status = "PASS" if passed else "FAIL"
    print(f"\n[{status}] {label}")
    if detail is not None:
        print(detail)
    return passed


async def main() -> int:
    print("=" * 88)
    print("DeepAgents eager tool-registry probe")
    print("=" * 88)
    print(f"Repository root: {REPOSITORY_ROOT}")
    print(f"Probe cwd:       {Path.cwd()}")

    capture = CaptureLiveToolsMiddleware()
    model = ProbeChatModel()

    with tempfile.TemporaryDirectory(prefix="mira-eager-tools-") as temp_dir:
        backend = FilesystemBackend(
            root_dir=Path(temp_dir),
            virtual_mode=True,
        )

        agent = create_deep_agent(
            model=model,
            backend=backend,
            tools=[mcp_probe_echo],
            middleware=[capture],
            subagents=[
                {
                    "name": "researcher",
                    "description": "Probe subagent.",
                    "system_prompt": "Return a concise deterministic answer.",
                    "model": model,
                    "tools": [],
                }
            ],
        )

        failures = 0

        # Critical ordering check: nothing has invoked the model yet.
        failures += not report(
            "No model request has occurred before eager-registry inspection",
            capture.live_tools is None,
        )

        found = find_eager_tool_node(agent)
        if found is None:
            report(
                "Exact execution ToolNode is reachable immediately after construction",
                False,
                "No ToolNode found on the tested construction-time graph surfaces.",
            )
            print("\nRESULT: No usable eager construction-time registry found.")
            return 1

        eager_path, tool_node = found
        eager = dict(tool_node.tools_by_name)

        failures += not report(
            "Exact execution ToolNode is reachable immediately after construction",
            True,
            f"path={eager_path}",
        )

        failures += not report(
            "Eager registry already contains built-ins, task, and custom/MCP-shaped tool",
            TARGETS.issubset(eager),
            sorted(eager),
        )

        failures += not report(
            "Custom/MCP-shaped tool keeps exact object identity at construction time",
            eager.get("mcp__probe__echo") is mcp_probe_echo,
            (
                f"passed_id={id(mcp_probe_echo)} "
                f"eager_id={id(eager.get('mcp__probe__echo'))}"
            ),
        )

        # Only now invoke the agent once so ModelRequest.tools can be captured.
        await agent.ainvoke(
            {"messages": [HumanMessage(content="Do not call tools. Reply briefly.")]},
            config={"configurable": {"thread_id": "eager-tool-registry-probe"}},
        )

        live = tool_map(capture.live_tools or [])

        failures += not report(
            "ModelRequest.tools was captured after the first model call",
            bool(live),
            sorted(live),
        )

        identity = {
            name: (
                name in eager
                and name in live
                and eager[name] is live[name]
            )
            for name in TARGETS
        }
        failures += not report(
            "Eager execution registry and live ModelRequest.tools share exact BaseTool instances",
            all(identity.values()),
            identity,
        )

        eager_names = set(eager)
        live_names = set(live)
        report(
            "Construction registry vs model-visible registry",
            True,
            {
                "eager_only": sorted(eager_names - live_names),
                "live_only": sorted(live_names - eager_names),
                "shared": sorted(eager_names & live_names),
            },
        )

        print("\n" + "=" * 88)
        print("INTERPRETATION")
        print("=" * 88)
        if failures == 0:
            print(
                "PASS: exact executable BaseTool objects are obtainable before the first "
                "model call, and the target objects are identical to those later exposed "
                "through ModelRequest.tools."
            )
            print(f"Construction-time access path: {eager_path}")
            if eager_path.startswith("agent.nodes"):
                print(
                    "CAVEAT: this proves eager availability through the compiled LangGraph "
                    "surface. It does not by itself prove that DeepAgents promises this path "
                    "as a stable public API."
                )
            elif ".builder." in eager_path:
                print(
                    "CAVEAT: access required builder introspection; treat this as diagnostic "
                    "evidence, not a production API."
                )
        else:
            print(f"FAILED: {failures} check(s)")

        return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
