"""Request-time model tool visibility filtering."""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware.types import AgentMiddleware

from agent.tools.specs import tool_name as resource_tool_name


class ModelToolVisibilityMiddleware(AgentMiddleware[Any, Any, Any]):
    """Hide selected tools from model calls."""

    def __init__(self, excluded_tools: tuple[str, ...]) -> None:
        self.excluded_tools = set(excluded_tools)

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        return handler(self._filter_request(request))

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        return await handler(self._filter_request(request))

    def _filter_request(self, request: Any) -> Any:
        tools = [tool for tool in request.tools if resource_tool_name(tool) not in self.excluded_tools]
        return request.override(tools=tools)


__all__ = ["ModelToolVisibilityMiddleware"]
