"""Focused trace-payload policy tests for execution-context calls."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from agent.execution.tools import RUNTIME_PAYLOAD_TAG
from tracing.semantic_processor import (
    LangSmithOpenInferenceProcessor,
    _redact_execution_context_payload,
)


class ExecutionTracingTests(unittest.TestCase):
    def test_only_marked_execution_context_payloads_are_redacted(self) -> None:
        payload = "large-value" * 10_000
        marked = {
            "langsmith.span.tags": f"ordinary, {RUNTIME_PAYLOAD_TAG}",
            "langsmith.span.kind": "tool",
            "gen_ai.prompt": payload,
            "gen_ai.completion": payload,
            "input.value": payload,
            "output.value": payload,
            "tool.name": "write_file",
        }
        ordinary = {
            "langsmith.span.tags": "ordinary",
            "langsmith.span.kind": "tool",
            "input.value": payload,
        }

        _redact_execution_context_payload(marked)
        _redact_execution_context_payload(ordinary)

        for key in ("gen_ai.prompt", "gen_ai.completion", "input.value", "output.value"):
            self.assertNotIn(payload, str(marked[key]))
            self.assertIn("execution-context payload redacted", str(marked[key]))
        self.assertEqual(marked["tool.name"], "write_file")
        self.assertEqual(ordinary["input.value"], payload)

    def test_processor_redacts_a_marked_span_without_removing_it(self) -> None:
        payload = "private-intermediate-value" * 1_000
        span = SimpleNamespace(
            attributes={
                "langsmith.span.tags": ["nested", RUNTIME_PAYLOAD_TAG],
                "openinference.span.kind": "TOOL",
                "input.value": payload,
                "output.value": payload,
                "tool.name": "write_file",
            },
            _attributes=None,
        )

        LangSmithOpenInferenceProcessor().on_end(span)

        self.assertEqual(span._attributes["tool.name"], "write_file")
        self.assertNotIn(payload, span._attributes["input.value"])
        self.assertNotIn(payload, span._attributes["output.value"])

    def test_marker_does_not_redact_non_tool_spans(self) -> None:
        payload = "ordinary-model-content"
        attributes = {
            "langsmith.span.tags": RUNTIME_PAYLOAD_TAG,
            "langsmith.span.kind": "llm",
            "gen_ai.prompt": payload,
            "input.value": payload,
        }

        _redact_execution_context_payload(attributes)

        self.assertEqual(attributes["gen_ai.prompt"], payload)
        self.assertEqual(attributes["input.value"], payload)


if __name__ == "__main__":
    unittest.main()
