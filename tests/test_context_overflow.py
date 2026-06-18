"""Tests for context overflow normalization."""

from __future__ import annotations

import unittest
from typing import Any

from langchain_core.exceptions import ContextOverflowError

from agent.context_overflow import (
    PROVIDER_CONTEXT_NOTICE,
    ContextPressureMiddleware,
    is_context_overflow_error,
    pop_context_floor_tokens,
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

    def __init__(self, thread_id: str = "thread-1") -> None:
        self.config = {"configurable": {"thread_id": thread_id}}


class Request:
    """Small model request double."""

    def __init__(self, messages: list[Any] | None = None, thread_id: str = "thread-1") -> None:
        self.messages = messages if messages is not None else [Message()]
        self.system_message = None
        self.tools: list[Any] = []
        self.runtime = Runtime(thread_id)


class UsageMessage:
    """Message with direct LangChain usage metadata."""

    def __init__(self, message_id: str, tokens: int) -> None:
        self.id = message_id
        self.usage_metadata = {"input_tokens": tokens, "output_tokens": 0, "total_tokens": tokens}


class NoIdUsageMessage:
    """Usage-bearing message without a provider id."""

    def __init__(self, text: str, tokens: int) -> None:
        self.content = text
        self.usage_metadata = {"input_tokens": tokens, "output_tokens": 0, "total_tokens": tokens}


class SummaryMessage:
    """DeepAgents-style summary marker."""

    content = """## SESSION INTENT
Continue a story.

## SUMMARY
Earlier context was summarized.

## ARTIFACTS
None.

## NEXT STEPS
Continue from the summary.
"""


class ContextOverflowTests(unittest.IsolatedAsyncioTestCase):
    """Context overflow middleware behavior."""

    def tearDown(self) -> None:
        pop_context_overflow_notice()
        pop_context_floor_tokens("thread-1")

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

    async def test_reported_context_pressure_allows_one_immediate_retry(self) -> None:
        middleware = ContextPressureMiddleware(context_limit_tokens=1000, threshold_fraction=0.98)
        calls = 0

        async def handler(request: Any) -> str:
            nonlocal calls
            calls += 1
            return "ok"

        with self.assertRaises(ContextOverflowError):
            await middleware.awrap_model_call(Request([UsageMessage("ai-1", 1020)]), handler)

        self.assertEqual(await middleware.awrap_model_call(Request([UsageMessage("ai-2", 1030)]), handler), "ok")
        with self.assertRaises(ContextOverflowError):
            await middleware.awrap_model_call(Request([UsageMessage("ai-2", 1030)]), handler)
        self.assertEqual(calls, 1)

    async def test_reported_context_pressure_ignores_usage_before_summary_marker(self) -> None:
        middleware = ContextPressureMiddleware(
            context_limit_tokens=1000,
            threshold_fraction=0.98,
            token_counter=lambda messages, tools=None: 0,
        )
        calls = 0

        async def handler(request: Any) -> str:
            nonlocal calls
            calls += 1
            return "ok"

        messages = [
            UsageMessage("old-ai", 1030),
            SummaryMessage(),
            {"role": "user", "content": "please finish it"},
        ]

        self.assertEqual(await middleware.awrap_model_call(Request(messages), handler), "ok")
        self.assertEqual(calls, 1)

    async def test_reported_context_pressure_uses_stable_signature_without_message_id(self) -> None:
        middleware = ContextPressureMiddleware(
            context_limit_tokens=1000,
            threshold_fraction=0.98,
            token_counter=lambda messages, tools=None: 0,
        )
        usage = NoIdUsageMessage("same assistant response", 1030)

        async def handler(request: Any) -> str:
            return "ok"

        with self.assertRaises(ContextOverflowError):
            await middleware.awrap_model_call(Request([usage]), handler)

        self.assertEqual(await middleware.awrap_model_call(Request([{"role": "user", "content": "retry"}, usage]), handler), "ok")
        self.assertEqual(await middleware.awrap_model_call(Request([{"role": "user", "content": "later"}, usage]), handler), "ok")

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

    async def test_request_token_count_is_remembered_as_context_floor(self) -> None:
        middleware = ContextPressureMiddleware(
            context_limit_tokens=1000,
            threshold_fraction=0.98,
            token_counter=lambda messages, tools=None: 513,
        )

        async def handler(request: Any) -> str:
            return "ok"

        self.assertEqual(await middleware.awrap_model_call(Request(messages=[]), handler), "ok")

        self.assertEqual(pop_context_floor_tokens("thread-1"), 513)

    async def test_compaction_retry_replaces_pre_compaction_context_floor(self) -> None:
        counts = iter([99, 42])
        middleware = ContextPressureMiddleware(
            context_limit_tokens=100,
            threshold_fraction=0.98,
            token_counter=lambda messages, tools=None: next(counts),
        )

        async def handler(request: Any) -> str:
            return "ok"

        with self.assertRaises(ContextOverflowError):
            await middleware.awrap_model_call(Request(messages=[]), handler)

        self.assertEqual(await middleware.awrap_model_call(Request(messages=[]), handler), "ok")

        self.assertEqual(pop_context_floor_tokens("thread-1"), 42)

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
