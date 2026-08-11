"""Convert workspace tool exceptions into model-visible tool errors."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.errors import GraphBubbleUp


class ProjectToolErrorMiddleware(AgentMiddleware):
    """Keep ordinary exceptions from enabled workspace tools inside the agent loop."""

    def __init__(self, tool_names: Iterable[str]) -> None:
        self.tool_names = frozenset(str(name) for name in tool_names if str(name))

    def wrap_tool_call(self, request: Any, handler: Any) -> Any:
        if not self._handles(request):
            return handler(request)
        try:
            return handler(request)
        except GraphBubbleUp:
            raise
        except Exception as error:
            return project_tool_error_message(request, error)

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        if not self._handles(request):
            return await handler(request)
        try:
            return await handler(request)
        except GraphBubbleUp:
            raise
        except Exception as error:
            return project_tool_error_message(request, error)

    def _handles(self, request: Any) -> bool:
        call = getattr(request, "tool_call", None)
        return isinstance(call, dict) and str(call.get("name") or "") in self.tool_names


def project_tool_error_message(request: Any, error: Exception) -> ToolMessage:
    """Return a concise native error result for one workspace tool failure."""
    call = request.tool_call
    name = str(call.get("name") or "tool")
    detail = str(error).strip() or type(error).__name__
    return ToolMessage(
        content=(
            f"{name} failed in the MIRA runtime.\n"
            f"{type(error).__name__}: {detail}"
        ),
        name=name,
        tool_call_id=str(call.get("id") or ""),
        status="error",
    )


__all__ = ["ProjectToolErrorMiddleware", "project_tool_error_message"]
