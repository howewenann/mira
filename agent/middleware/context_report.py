"""Passive request observation for Context Reports."""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware.types import AgentMiddleware

from core.context.observation import record_context_report_inputs, record_deepagents_context_tokens


class ContextReportMiddleware(AgentMiddleware[Any, Any, Any]):
    """Capture the effective MIRA request without changing model behavior."""

    def __init__(self, summarization: Any | None = None) -> None:
        self._summarization = summarization

    def _observe(self, request: Any) -> None:
        record_context_report_inputs(request.messages, request.system_message, request.tools)
        counter = getattr(self._summarization, "_count_tokens", None)
        if callable(counter):
            record_deepagents_context_tokens(counter(request.messages, request.system_message, request.tools))

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        self._observe(request)
        return handler(request)

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        self._observe(request)
        return await handler(request)


__all__ = ["ContextReportMiddleware"]
