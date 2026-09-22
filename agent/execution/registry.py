"""Compatibility boundary for compiled DeepAgents executable tools."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any

from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode


class ExecutableRegistryError(RuntimeError):
    """Raised when the supported DeepAgents ToolNode shape is unavailable."""


def executable_tool_registry(
    agent: Any,
    *,
    excluded: Iterable[str] = (),
) -> Mapping[str, BaseTool]:
    """Return the agent's exact executable tools after policy filtering."""
    tool_node = _tool_node(agent)
    blocked = {str(name) for name in excluded if str(name)}
    tools = {
        name: tool
        for name, tool in tool_node.tools_by_name.items()
        if name not in blocked
    }
    if not tools:
        raise ExecutableRegistryError(
            "MIRA could not expose execution capabilities because the compiled "
            "DeepAgents ToolNode contains no policy-available tools."
        )
    if not all(isinstance(tool, BaseTool) for tool in tools.values()):
        raise ExecutableRegistryError(
            "MIRA expected the compiled DeepAgents ToolNode registry to contain "
            "only LangChain BaseTool instances."
        )
    return MappingProxyType(tools)


def _tool_node(agent: Any) -> ToolNode:
    try:
        nodes = agent.nodes
        start = nodes["tools"]
    except (AttributeError, KeyError, TypeError) as exc:
        raise ExecutableRegistryError(
            "MIRA could not locate the compiled DeepAgents 'tools' node; the "
            "installed DeepAgents/LangGraph graph shape is unsupported."
        ) from exc

    queue: deque[tuple[Any, int]] = deque([(start, 0)])
    seen: set[int] = set()
    while queue:
        value, depth = queue.popleft()
        if id(value) in seen or depth > 6:
            continue
        seen.add(id(value))
        if isinstance(value, ToolNode):
            return value
        for attribute in ("bound", "runnable", "data", "node"):
            try:
                child = getattr(value, attribute)
            except Exception:
                continue
            if child is not None:
                queue.append((child, depth + 1))
        steps = getattr(value, "steps", None)
        if isinstance(steps, (list, tuple)):
            queue.extend((child, depth + 1) for child in steps)

    raise ExecutableRegistryError(
        "MIRA found the compiled DeepAgents 'tools' node but it did not contain "
        "the expected LangGraph ToolNode."
    )


__all__ = ["ExecutableRegistryError", "executable_tool_registry"]
