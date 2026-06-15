"""Tests for context overflow normalization."""

from __future__ import annotations

import unittest
from typing import Any

from langchain_core.exceptions import ContextOverflowError

from agent.context_overflow import ContextPressureMiddleware, is_context_overflow_error


class Stats:
    """LM Studio-style stats object."""

    prompt_tokens_count = 1020
    predicted_tokens_count = 0
    total_tokens_count = 1020


class Message:
    """Message with provider response metadata."""

    id = "ai-1"
    response_metadata = {"stats": Stats()}
    usage_metadata: dict[str, Any] = {}


class Runtime:
    """Small runtime double exposing LangGraph config."""

    config = {"configurable": {"thread_id": "thread-1"}}


class Request:
    """Small model request double."""

    def __init__(self, messages: list[Any] | None = None) -> None:
        self.messages = messages if messages is not None else [Message()]
        self.system_message = None
        self.tools: list[Any] = []
        self.runtime = Runtime()


class ContextOverflowTests(unittest.IsolatedAsyncioTestCase):
    """Context overflow middleware behavior."""

    async def test_reported_context_pressure_raises_once_per_signature(self) -> None:
        middleware = ContextPressureMiddleware(context_limit_tokens=1000, threshold_fraction=0.98)
        calls = 0

        async def handler(request: Any) -> str:
            nonlocal calls
            calls += 1
            return "ok"

        with self.assertRaises(ContextOverflowError):
            await middleware.awrap_model_call(Request(), handler)

        self.assertEqual(await middleware.awrap_model_call(Request(), handler), "ok")
        self.assertEqual(calls, 1)

    async def test_estimated_context_pressure_raises(self) -> None:
        middleware = ContextPressureMiddleware(
            context_limit_tokens=100,
            threshold_fraction=0.98,
            token_counter=lambda messages, tools=None: 99,
        )

        async def handler(request: Any) -> str:
            return "ok"

        with self.assertRaises(ContextOverflowError):
            await middleware.awrap_model_call(Request(messages=[]), handler)

    async def test_provider_context_errors_are_rethrown_as_context_overflow(self) -> None:
        middleware = ContextPressureMiddleware(context_limit_tokens=1000, enabled=False)

        async def handler(request: Any) -> str:
            raise RuntimeError("Input tokens exceed the configured limit of this model.")

        with self.assertRaises(ContextOverflowError):
            await middleware.awrap_model_call(Request(messages=[]), handler)

    def test_context_error_heuristic_avoids_rate_limits(self) -> None:
        self.assertTrue(is_context_overflow_error(RuntimeError("prompt is too long for the context window")))
        self.assertFalse(is_context_overflow_error(RuntimeError("rate limit exceeded; retry later")))


if __name__ == "__main__":
    unittest.main()
