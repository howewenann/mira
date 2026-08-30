"""Passive request observation for Context Reports."""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware.types import AgentMiddleware

from runtime.context_usage import record_context_report_inputs


class ContextReportMiddleware(AgentMiddleware[Any, Any, Any]):
    """Capture the effective MIRA request without changing model behavior."""

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        record_context_report_inputs(request.messages, request.system_message, request.tools)
        return handler(request)

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        record_context_report_inputs(request.messages, request.system_message, request.tools)
        return await handler(request)


__all__ = ["ContextReportMiddleware"]
