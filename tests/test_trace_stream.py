"""Tests for readable trace stream formatting."""

from __future__ import annotations

import logging
import unittest

from runtime.trace_stream import TraceStream


class ListHandler(logging.Handler):
    """Capture formatted log messages for trace stream tests."""

    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


class TraceStreamTests(unittest.TestCase):
    """Tests for coalesced sidecar transcript formatting."""

    def make_stream(self, *, output_chars: int = 240) -> tuple[TraceStream, ListHandler]:
        logger = logging.getLogger(f"trace-stream-test-{id(self)}")
        logger.handlers = []
        logger.propagate = False
        logger.setLevel(logging.INFO)
        handler = ListHandler()
        logger.addHandler(handler)
        return TraceStream(logger, output_chars=output_chars), handler

    def test_assistant_deltas_flush_once_as_one_block(self) -> None:
        trace, handler = self.make_stream()

        trace.assistant_delta("he")
        trace.assistant_delta("llo")
        trace.flush_all()
        trace.flush_all()

        self.assertEqual(handler.messages, ["\nmira:\nhello"])
        self.assertNotIn("\033[", handler.messages[0])

    def test_reasoning_deltas_flush_once_as_one_block(self) -> None:
        trace, handler = self.make_stream()

        trace.reasoning_delta("think")
        trace.reasoning_delta("ing")
        trace.flush_all()
        trace.flush_all()

        self.assertEqual(handler.messages, ["\nthinking:\nthinking"])

    def test_assistant_delta_flushes_reasoning_first(self) -> None:
        trace, handler = self.make_stream()

        trace.reasoning_delta("considering")
        trace.assistant_delta("answer")
        trace.flush_all()

        self.assertEqual(handler.messages, ["\nthinking:\nconsidering\n\nmira:\nanswer"])

    def test_reasoning_delta_flushes_assistant_first(self) -> None:
        trace, handler = self.make_stream()

        trace.assistant_delta("answer")
        trace.reasoning_delta("afterthought")
        trace.flush_all()

        self.assertEqual(handler.messages, ["\nmira:\nanswer\n\nthinking:\nafterthought"])

    def test_discard_reasoning_drops_pending_reasoning(self) -> None:
        trace, handler = self.make_stream()

        trace.reasoning_delta("hidden compaction")
        trace.discard_reasoning()
        trace.flush_all()

        self.assertEqual(handler.messages, [])

    def test_tool_call_flushes_assistant_before_tool(self) -> None:
        trace, handler = self.make_stream()

        trace.reasoning_delta("before answer")
        trace.assistant_delta("before tool")
        trace.tool_call("read_file", {"path": "README.md"})

        rendered = "\n".join(handler.messages)
        self.assertIn("thinking:\nbefore answer", rendered)
        self.assertIn("mira:\nbefore tool", rendered)
        self.assertIn("read_file:\nargs:", rendered)

    def test_values_are_truncated(self) -> None:
        trace, handler = self.make_stream(output_chars=8)

        trace.tool_result("read_file", "x" * 20)

        self.assertEqual(handler.messages, ["read_file output: xxxxxxxx ... truncated ..."])

    def test_recovered_tool_result_matches_terminal_callback_order(self) -> None:
        trace, handler = self.make_stream()

        trace.assistant_delta("final answer")
        trace.recovered_tool_result("write_todos", "updated")

        self.assertEqual(handler.messages, ["\nmira:\nfinal answer\nwrite_todos output: updated"])

    def test_normal_tool_result_flushes_pending_assistant_first(self) -> None:
        trace, handler = self.make_stream()

        trace.assistant_delta("before output")
        trace.tool_result("write_todos", "updated")

        self.assertEqual(handler.messages, ["\nmira:\nbefore output\nwrite_todos output: updated"])

    def test_user_system_subagent_and_delegation_blocks(self) -> None:
        trace, handler = self.make_stream()

        trace.user_message("hello")
        trace.system_message("bad", kind="error")
        trace.subagent_started("worker", "do work")
        trace.subagent_finished("worker", "done")
        trace.delegation_started([{"args": {"description": "check files"}}])

        joined = "\n".join(handler.messages)
        self.assertIn("user:\nhello", joined)
        self.assertIn("error:\nbad", joined)
        self.assertIn("subagent - worker:\nrequest: do work", joined)
        self.assertIn("subagent - worker:\ndone\noutput: done", joined)
        self.assertIn("task:\ndelegating to 1 subagent(s)", joined)
        self.assertIn("request: check files", joined)

    def test_disabled_stream_is_no_op(self) -> None:
        trace = TraceStream.disabled()

        trace.user_message("hello")
        trace.assistant_delta("hello")
        trace.flush_all()

        self.assertFalse(trace.enabled)


if __name__ == "__main__":
    unittest.main()
