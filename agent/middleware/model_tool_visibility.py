"""Model tool visibility filtering and execution enforcement."""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import ToolMessage

from agent.tools.specs import tool_name as resource_tool_name


class ModelToolVisibilityMiddleware(AgentMiddleware[Any, Any, Any]):
    """Hide and reject tools outside the configured visible surface."""

    def __init__(
        self,
        excluded_tools: tuple[str, ...] = (),
        *,
        allowed_tools: tuple[str, ...] | None = None,
    ) -> None:
        self.excluded_tools = set(excluded_tools)
        self.allowed_tools = set(allowed_tools) if allowed_tools is not None else None

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        return handler(self._filter_request(request))

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        return await handler(self._filter_request(request))

    def wrap_tool_call(self, request: Any, handler: Any) -> Any:
        error = self._tool_error(request)
        return error if error is not None else handler(request)

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        error = self._tool_error(request)
        return error if error is not None else await handler(request)

    def _filter_request(self, request: Any) -> Any:
        tools = [tool for tool in request.tools if self._is_available(resource_tool_name(tool))]
        return request.override(tools=tools)

    def _tool_error(self, request: Any) -> ToolMessage | None:
        call = getattr(request, "tool_call", None)
        if not isinstance(call, dict):
            return None
        name = str(call.get("name") or "")
        if self._is_available(name):
            return None
        return ToolMessage(
            content=f"Error: {name or 'tool'} is not available.",
            name=name or "tool",
            tool_call_id=str(call.get("id") or ""),
            status="error",
        )

    def _is_available(self, name: str | None) -> bool:
        if not name:
            return self.allowed_tools is None
        if name in self.excluded_tools:
            return False
        return self.allowed_tools is None or name in self.allowed_tools


__all__ = ["ModelToolVisibilityMiddleware"]
