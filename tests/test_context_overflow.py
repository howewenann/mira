"""Tests for provider context overflow normalization."""

from __future__ import annotations

import unittest
from typing import Any

from langchain_core.exceptions import ContextOverflowError

from agent.context_overflow import (
    PROVIDER_CONTEXT_NOTICE,
    ProviderContextOverflowMiddleware,
    is_context_overflow_error,
    pop_context_overflow_notice,
)


class Runtime:
    """Small runtime double exposing LangGraph config."""

    config = {"configurable": {"thread_id": "thread-1"}}


class Request:
    """Small model request double."""

    runtime = Runtime()


class ContextOverflowTests(unittest.IsolatedAsyncioTestCase):
    """Context overflow middleware behavior."""

    def tearDown(self) -> None:
        pop_context_overflow_notice()

    async def test_provider_context_errors_are_rethrown_as_context_overflow(self) -> None:
        middleware = ProviderContextOverflowMiddleware()

        async def handler(request: Any) -> str:
            raise RuntimeError("Input tokens exceed the configured limit of this model.")

        with self.assertRaises(ContextOverflowError) as caught:
            await middleware.awrap_model_call(Request(), handler)

        self.assertEqual(str(caught.exception), "provider context limit reached")
        self.assertEqual(pop_context_overflow_notice(caught.exception), PROVIDER_CONTEXT_NOTICE)
        self.assertNotIn("Input tokens exceed", str(caught.exception))

    async def test_existing_context_overflow_gets_notice(self) -> None:
        middleware = ProviderContextOverflowMiddleware()

        async def handler(request: Any) -> str:
            raise ContextOverflowError("context overflow")

        with self.assertRaises(ContextOverflowError) as caught:
            await middleware.awrap_model_call(Request(), handler)

        self.assertEqual(pop_context_overflow_notice(caught.exception), PROVIDER_CONTEXT_NOTICE)

    async def test_non_context_errors_pass_through(self) -> None:
        middleware = ProviderContextOverflowMiddleware()

        async def handler(request: Any) -> str:
            raise RuntimeError("temporary provider outage")

        with self.assertRaisesRegex(RuntimeError, "temporary provider outage"):
            await middleware.awrap_model_call(Request(), handler)

        self.assertEqual(pop_context_overflow_notice(), "")

    def test_context_error_heuristic_avoids_rate_limits(self) -> None:
        self.assertTrue(is_context_overflow_error(RuntimeError("prompt is too long for the context window")))
        self.assertFalse(is_context_overflow_error(RuntimeError("rate limit exceeded; retry later")))


if __name__ == "__main__":
    unittest.main()
