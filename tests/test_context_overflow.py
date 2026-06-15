"""Tests for context overflow normalization."""

from __future__ import annotations

import unittest
from typing import Any

from langchain_core.exceptions import ContextOverflowError

from agent.context_overflow import (
    PROVIDER_CONTEXT_NOTICE,
    ContextPressureMiddleware,
    is_context_overflow_error,
    pop_context_overflow_notice,
)


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

    def tearDown(self) -> None:
        pop_context_overflow_notice()

    async def test_reported_context_pressure_raises_once_per_signature(self) -> None:
        middleware = ContextPressureMiddleware(context_limit_tokens=1000, threshold_fraction=0.98)
        calls = 0

        async def handler(request: Any) -> str:
            nonlocal calls
            calls += 1
            return "ok"

        with self.assertRaises(ContextOverflowError) as caught:
            await middleware.awrap_model_call(Request(), handler)

        self.assertNotIn("MIRA simulated a context overflow", str(caught.exception))
        notice = pop_context_overflow_notice(caught.exception)
        self.assertIn("Configured context threshold reached", notice)
        self.assertIn("1.0k tokens reported", notice)
        self.assertIn("threshold 980", notice)
        self.assertIn("limit 1.0k", notice)

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

        with self.assertRaises(ContextOverflowError) as caught:
            await middleware.awrap_model_call(Request(messages=[]), handler)

        notice = pop_context_overflow_notice(caught.exception)
        self.assertIn("99 tokens estimated", notice)
        self.assertIn("threshold 98", notice)

    async def test_provider_context_errors_are_rethrown_as_context_overflow(self) -> None:
        middleware = ContextPressureMiddleware(context_limit_tokens=1000, enabled=False)

        async def handler(request: Any) -> str:
            raise RuntimeError("Input tokens exceed the configured limit of this model.")

        with self.assertRaises(ContextOverflowError) as caught:
            await middleware.awrap_model_call(Request(messages=[]), handler)

        self.assertEqual(str(caught.exception), "provider context limit reached")
        self.assertEqual(pop_context_overflow_notice(caught.exception), PROVIDER_CONTEXT_NOTICE)
        self.assertNotIn("Input tokens exceed", str(caught.exception))

    def test_context_error_heuristic_avoids_rate_limits(self) -> None:
        self.assertTrue(is_context_overflow_error(RuntimeError("prompt is too long for the context window")))
        self.assertFalse(is_context_overflow_error(RuntimeError("rate limit exceeded; retry later")))


if __name__ == "__main__":
    unittest.main()
