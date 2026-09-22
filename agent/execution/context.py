"""LangGraph runtime context for trusted in-process MIRA capabilities."""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, ToolCallRequest
from langgraph.prebuilt import ToolRuntime

from agent.execution.agents import BoundAgentMap, bind_agents
from agent.execution.tools import (
    BoundToolMap,
    _ignore_unparameterized_runtime_warning,
    bind_tools,
)


@dataclass(frozen=True, slots=True)
class MiraContext:
    """Capability-bearing context supplied to trusted LangGraph/Python code."""

    tools: BoundToolMap
    agents: BoundAgentMap


_ACTIVE_TOOL_RUNTIME: ContextVar[ToolRuntime[Any, Any] | None] = ContextVar(
    "mira_active_tool_runtime",
    default=None,
)


def active_tool_runtime() -> ToolRuntime[Any, Any] | None:
    """Return the ToolRuntime for the currently executing outer tool."""
    return _ACTIVE_TOOL_RUNTIME.get()


class ExecutionContextMiddleware(AgentMiddleware[Any, MiraContext, Any]):
    """Make native ToolRuntime data available to deterministic nested calls."""

    def wrap_tool_call(self, request: ToolCallRequest, handler: Any) -> Any:
        token = _set_runtime(request.runtime)
        try:
            with _ignore_unparameterized_runtime_warning():
                return handler(request)
        finally:
            _ACTIVE_TOOL_RUNTIME.reset(token)

    async def awrap_tool_call(self, request: ToolCallRequest, handler: Any) -> Any:
        token = _set_runtime(request.runtime)
        try:
            with _ignore_unparameterized_runtime_warning():
                return await handler(request)
        finally:
            _ACTIVE_TOOL_RUNTIME.reset(token)


def _set_runtime(runtime: ToolRuntime[Any, Any]) -> Token[ToolRuntime[Any, Any] | None]:
    return _ACTIVE_TOOL_RUNTIME.set(runtime)


def create_mira_context(tool_registry: Any, subagent_names: Any) -> MiraContext:
    """Bind one immutable context to an already-resolved executable surface."""
    bound_tools = bind_tools(tool_registry)
    return MiraContext(
        tools=bound_tools,
        agents=bind_agents(
            task=bound_tools.get("task"),
            names=subagent_names,
        ),
    )


__all__ = [
    "ExecutionContextMiddleware",
    "MiraContext",
    "active_tool_runtime",
    "create_mira_context",
]
