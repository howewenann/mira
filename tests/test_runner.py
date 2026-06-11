"""Tests for runtime event handling."""

from __future__ import annotations

import unittest
from typing import Any

from runtime import runner
from runtime.message_events import consume_messages
from runtime.output_events import final_text
from runtime.subagent_events import consume_subagent, consume_subagents
from runtime.usage import usage_from_message, usage_from_output
from ui.interrupts import ASK_USER_OPEN_OPTION, ask_user_options


class AsyncItems:
    """Async iterable test double for DeepAgents event streams."""

    def __init__(self, items: list[Any]) -> None:
        self.items = items

    async def __aiter__(self) -> Any:
        for item in self.items:
            yield item


COMPACTION_SUMMARY = """## SESSION INTENT
User requested a story.

## SUMMARY
The conversation was summarized.

## ARTIFACTS
None.

## NEXT STEPS
Await further instructions.
"""

SUMMARY_THEN_ANSWER = f"{COMPACTION_SUMMARY}\nThe rain tapped against the window."


class Message:
    """Fake streamed message containing optional tool calls."""

    def __init__(
        self,
        tool_calls: list[Any] | None = None,
        reasoning: Any | None = None,
        text: Any | None = None,
        additional_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.tool_calls = tool_calls or []
        self.reasoning = reasoning
        self.text = text
        self.additional_kwargs = additional_kwargs or {}


class RawMessageStream:
    """Fake ChatModelStream exposing ordered raw protocol events."""

    def __init__(self, events: list[dict[str, Any]], tool_calls: Any | None = None) -> None:
        self.events = events
        self.text = AsyncItems([])
        self.reasoning = AsyncItems([])
        self.tool_calls = tool_calls or []

    async def __aiter__(self) -> Any:
        for event in self.events:
            yield event


class OutputMessage:
    """Fake final message object that exposes a text attribute."""

    def __init__(self, text: str, usage_metadata: dict[str, int] | None = None) -> None:
        self.text = text
        self.usage_metadata = usage_metadata or {}


class ToolCall:
    """Fake tool call event with name, args, and output."""

    def __init__(self, name: str, args: dict[str, Any], output: Any, call_id: str = "") -> None:
        self.name = name
        self.args = args
        self.output = output
        self.id = call_id


class Subagent:
    """Fake subagent with a final message shaped like DeepAgents output."""

    def __init__(self, name: str, tool_calls: list[ToolCall]) -> None:
        self.name = name
        self.task_input = "look around"
        self.tool_calls = AsyncItems(tool_calls)
        self.output = {
            "messages": [
                type("Message", (), {"text": tool_calls[-1].output if tool_calls else ""})()
            ]
        }


class RecordingRenderer:
    """Renderer double that records high-level events."""

    def __init__(self) -> None:
        self.events: list[tuple[Any, ...]] = []

    def reasoning_delta(self, value: str) -> None:
        self.events.append(("reasoning", value))

    def text_delta(self, value: str) -> None:
        self.events.append(("text", value))

    def tool_call(self, name: str, args: Any, call_id: str = "") -> None:
        self.events.append(("tool_call", name, args, call_id))

    def tool_result(self, name: str, result: str, call_id: str = "") -> None:
        self.events.append(("tool_result", name, result, call_id))

    def delegation_started(self, calls: list[dict[str, Any]]) -> None:
        self.events.append(("delegation_started", calls))

    def compaction_started(self) -> None:
        self.events.append(("compaction_started",))

    def compaction_finished(self) -> None:
        self.events.append(("compaction_finished",))

    def subagent_label(self, subagent: Any) -> str:
        return subagent.name

    def subagent_started(self, name: str, task_input: str = "") -> None:
        self.events.append(("subagent_started", name, task_input))

    def subagent_finished(self, name: str, result: str = "") -> None:
        self.events.append(("subagent_finished", name, result))


class RunTurnRenderer(RecordingRenderer):
    """Renderer double with approval support for full run-turn tests."""

    def __init__(self, decisions: list[dict[str, Any]] | None = None, ask_user_answer: str = "Use B") -> None:
        super().__init__()
        self.decisions = decisions or [{"type": "approve"}]
        self.approvals: list[list[Any]] = []
        self.ask_user_answer = ask_user_answer
        self.ask_user_prompts: list[Any] = []

    def finish_main(self) -> None:
        self.events.append(("finish_main",))

    async def ask_approvals(self, interrupts: list[Any]) -> list[dict[str, Any]]:
        self.approvals.append(interrupts)
        self.events.append(("ask_approvals", interrupts))
        return self.decisions

    async def ask_user(self, interrupt: Any) -> str:
        self.ask_user_prompts.append(interrupt)
        self.events.append(("ask_user", interrupt))
        return self.ask_user_answer


class FakeStream:
    """Fake DeepAgents stream with the channels the runner consumes."""

    def __init__(self, output: Any = None, interrupts: list[Any] | None = None) -> None:
        self.messages = AsyncItems([])
        self.tool_calls = AsyncItems([])
        self.subagents = AsyncItems([])
        self.output_value = output or {}
        self.interrupt_values = interrupts or []

    async def output(self) -> Any:
        return self.output_value

    def interrupts(self) -> list[Any]:
        return self.interrupt_values


class FakeAgent:
    """Fake agent that returns prebuilt streams for each invocation."""

    def __init__(self, streams: list[FakeStream]) -> None:
        self.streams = list(streams)
        self.payloads: list[Any] = []

    async def astream_events(self, payload: Any, config: dict[str, Any], version: str) -> FakeStream:
        self.payloads.append(payload)
        return self.streams.pop(0)


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    """Tests for runner event handling."""

    async def test_run_turn_asks_approval_for_stream_interrupts(self) -> None:
        interrupt = {
            "action_requests": [
                {
                    "name": "write_file",
                    "args": {"file_path": "/test.txt", "content": "hello world"},
                }
            ]
        }
        agent = FakeAgent(
            [
                FakeStream(output={"messages": []}, interrupts=[interrupt]),
                FakeStream(output={"messages": []}),
            ]
        )
        renderer = RunTurnRenderer(decisions=[{"type": "approve"}])

        result = await runner.run_turn(agent, "write file", renderer, "thread-1")

        self.assertEqual(renderer.approvals, [[interrupt]])
        self.assertEqual(len(agent.payloads), 2)
        self.assertEqual(agent.payloads[0], {"messages": [{"role": "user", "content": "write file"}]})
        self.assertEqual(agent.payloads[1].resume, {"decisions": [{"type": "approve"}]})
        self.assertEqual(result.final_text, "")

    async def test_run_turn_resumes_ask_user_interrupt_with_answer(self) -> None:
        interrupt = {
            "type": "ask_user",
            "question": "Which path should MIRA take?",
            "options": ["Use A", "Use B", ASK_USER_OPEN_OPTION],
        }
        agent = FakeAgent(
            [
                FakeStream(output={"messages": []}, interrupts=[interrupt]),
                FakeStream(output={"messages": []}),
            ]
        )
        renderer = RunTurnRenderer(ask_user_answer="Use B")

        await runner.run_turn(agent, "choose", renderer, "thread-1")

        self.assertEqual(renderer.ask_user_prompts, [interrupt])
        self.assertEqual(renderer.approvals, [])
        self.assertEqual(agent.payloads[1].resume, "Use B")

    async def test_run_turn_exits_when_stream_has_no_interrupts(self) -> None:
        agent = FakeAgent([FakeStream(output={"messages": []})])
        renderer = RunTurnRenderer()

        await runner.run_turn(agent, "no write", renderer, "thread-1")

        self.assertEqual(renderer.approvals, [])
        self.assertEqual(len(agent.payloads), 1)

    async def test_run_turn_supports_output_interrupt_fallback(self) -> None:
        interrupt = {
            "action_requests": [
                {
                    "name": "edit_file",
                    "args": {"file_path": "/test.txt", "old_string": "a", "new_string": "b"},
                }
            ]
        }
        agent = FakeAgent(
            [
                FakeStream(output={"__interrupt__": [interrupt]}),
                FakeStream(output={"messages": []}),
            ]
        )
        renderer = RunTurnRenderer(decisions=[{"type": "reject"}])

        await runner.run_turn(agent, "edit file", renderer, "thread-1")

        self.assertEqual(renderer.approvals, [[interrupt]])
        self.assertEqual(agent.payloads[1].resume, {"decisions": [{"type": "reject"}]})

    async def test_run_turn_returns_final_text_from_output_messages(self) -> None:
        agent = FakeAgent([FakeStream(output={"messages": [OutputMessage("final plan")]})])
        renderer = RunTurnRenderer()

        result = await runner.run_turn(agent, "plan", renderer, "thread-1")

        self.assertEqual(result.final_text, "final plan")

    def test_final_text_skips_trailing_compaction_summary(self) -> None:
        """Metadata-marked compaction summaries should not become the visible final reply."""
        self.assertEqual(
            final_text(
                {
                    "messages": [
                        OutputMessage("real reply"),
                        Message(text=COMPACTION_SUMMARY, additional_kwargs={"lc_source": "summarization"}),
                    ]
                }
            ),
            "real reply",
        )

    def test_final_text_returns_empty_for_only_compaction_summary(self) -> None:
        """A metadata-marked compaction-only output should not create an assistant reply."""
        self.assertEqual(
            final_text({"messages": [Message(text=COMPACTION_SUMMARY, additional_kwargs={"lc_source": "summarization"})]}),
            "",
        )

    def test_final_text_strips_unmarked_compaction_summary_prefix(self) -> None:
        """Structured compaction summaries should be hidden even without metadata."""
        self.assertEqual(final_text({"messages": [OutputMessage(SUMMARY_THEN_ANSWER)]}), "The rain tapped against the window.")

    def test_final_text_returns_empty_for_unmarked_compaction_summary(self) -> None:
        """A compaction-only output should not become an assistant reply without metadata."""
        self.assertEqual(final_text({"messages": [OutputMessage(COMPACTION_SUMMARY)]}), "")

    def test_final_text_skips_langchain_summarization_message(self) -> None:
        """DeepAgents summary metadata should hide a summary regardless of text shape."""
        summary = Message(text="internal summary", additional_kwargs={"lc_source": "summarization"})

        self.assertEqual(final_text({"messages": [OutputMessage("real reply"), summary]}), "real reply")

    def test_final_text_keeps_normal_markdown_headings(self) -> None:
        """Ordinary assistant markdown should render unless it matches compaction shape."""
        text = "## SUMMARY\nThis is a normal project summary, not a compacted session."

        self.assertEqual(final_text({"messages": [OutputMessage(text)]}), text)

    async def test_run_turn_records_final_message_usage(self) -> None:
        agent = FakeAgent(
            [
                FakeStream(
                    output={
                        "messages": [
                            OutputMessage(
                                "done",
                                {"input_tokens": 5512, "output_tokens": 91, "total_tokens": 5603},
                            )
                        ]
                    }
                )
            ]
        )
        renderer = RunTurnRenderer()

        result = await runner.run_turn(agent, "use tokens", renderer, "thread-1")

        self.assertEqual(result.usage["input_tokens"], 5512)
        self.assertEqual(result.usage["output_tokens"], 91)
        self.assertEqual(result.usage["context_tokens"], 5603)

    def test_commit_loop_usage_returns_per_loop_delta(self) -> None:
        result = runner.TurnResult()

        first = result.commit_loop_usage(
            {"messages": [OutputMessage("first", {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110})]}
        )
        second = result.commit_loop_usage(
            {"messages": [OutputMessage("second", {"input_tokens": 200, "output_tokens": 20, "total_tokens": 220})]}
        )

        self.assertEqual(first["input_tokens"], 100)
        self.assertEqual(first["output_tokens"], 10)
        self.assertEqual(second["input_tokens"], 200)
        self.assertEqual(second["output_tokens"], 20)
        self.assertEqual(result.usage["input_tokens"], 300)
        self.assertEqual(result.usage["output_tokens"], 30)

    def test_final_output_uses_latest_usage_message_only(self) -> None:
        """DeepAgents final state may contain older messages with stale usage."""
        usage = usage_from_output(
            {
                "messages": [
                    OutputMessage("first", {"input_tokens": 8000, "output_tokens": 100}),
                    {"role": "user", "content": "next request"},
                    OutputMessage("latest", {"input_tokens": 9200, "output_tokens": 200}),
                ]
            }
        )

        self.assertEqual(usage["input_tokens"], 9200)
        self.assertEqual(usage["output_tokens"], 200)
        self.assertEqual(usage["context_tokens"], 9400)

    async def test_run_turn_does_not_sum_historical_final_message_usage(self) -> None:
        """Cumulative usage should add one current call per turn, not all state messages."""
        agent = FakeAgent(
            [
                FakeStream(
                    output={
                        "messages": [
                            OutputMessage("first", {"input_tokens": 8000, "output_tokens": 100}),
                        ]
                    }
                ),
                FakeStream(
                    output={
                        "messages": [
                            OutputMessage("first", {"input_tokens": 8000, "output_tokens": 100}),
                            {"role": "user", "content": "second request"},
                            OutputMessage("second", {"input_tokens": 9000, "output_tokens": 200}),
                        ]
                    }
                ),
                FakeStream(
                    output={
                        "messages": [
                            OutputMessage("first", {"input_tokens": 8000, "output_tokens": 100}),
                            OutputMessage("second", {"input_tokens": 9000, "output_tokens": 200}),
                            {"role": "user", "content": "third request"},
                            OutputMessage("third", {"input_tokens": 10000, "output_tokens": 300}),
                        ]
                    }
                ),
            ]
        )
        result = runner.TurnResult()

        for _ in range(3):
            stream = await agent.astream_events({"messages": []}, config={}, version="v3")
            result.commit_loop_usage(await stream.output())

        self.assertEqual(result.usage["input_tokens"], 27000)
        self.assertEqual(result.usage["output_tokens"], 600)
        self.assertEqual(result.usage["context_tokens"], 10300)

    async def test_run_turn_uses_counter_only_for_context_when_metadata_is_missing(self) -> None:
        agent = FakeAgent(
            [
                FakeStream(
                    output={
                        "messages": [
                            {"role": "user", "content": "hello world"},
                            {"role": "assistant", "content": "OK"},
                        ]
                    }
                )
            ]
        )
        renderer = RunTurnRenderer()

        result = await runner.run_turn(
            agent,
            "use tokens",
            renderer,
            "thread-1",
            token_counter=lambda text: len(text.split()),
        )

        self.assertEqual(result.usage["input_tokens"], 0)
        self.assertEqual(result.usage["output_tokens"], 0)
        self.assertEqual(result.usage["context_tokens"], 5)
        self.assertEqual(result.usage["source"], "langchain_approx.count_tokens")

    async def test_run_turn_does_not_lower_provider_context_with_visible_text_estimate(self) -> None:
        agent = FakeAgent(
            [
                FakeStream(
                    output={
                        "messages": [
                            {"role": "user", "content": "hello world"},
                            OutputMessage(
                                "OK",
                                {"input_tokens": 5512, "output_tokens": 91, "total_tokens": 5603},
                            ),
                        ]
                    }
                )
            ]
        )
        renderer = RunTurnRenderer()

        result = await runner.run_turn(
            agent,
            "use tokens",
            renderer,
            "thread-1",
            token_counter=lambda text: len(text.split()),
        )

        self.assertEqual(result.usage["input_tokens"], 5512)
        self.assertEqual(result.usage["output_tokens"], 91)
        self.assertEqual(result.usage["total_tokens"], 5603)
        self.assertEqual(result.usage["context_tokens"], 5603)

    async def test_run_turn_records_tool_calls_and_results(self) -> None:
        agent = FakeAgent([FakeStream(output={"messages": []}, interrupts=[])])
        agent.streams[0].messages = AsyncItems([Message([{"id": "call-1", "name": "write_file", "args": {}}])])
        agent.streams[0].tool_calls = AsyncItems([ToolCall("write_file", {}, "permission denied for write on /x", "call-1")])
        renderer = RunTurnRenderer()

        result = await runner.run_turn(agent, "write", renderer, "thread-1")

        self.assertEqual(result.tool_calls, ["write_file"])
        self.assertIn("permission denied for write on /x", result.tool_results)

    async def test_run_turn_strips_blank_leading_reply_gap(self) -> None:
        agent = FakeAgent([FakeStream(output={"messages": [OutputMessage("\n\nHello")]} )])
        renderer = RunTurnRenderer()

        result = await runner.run_turn(agent, "hello", renderer, "thread-1")

        self.assertEqual(result.final_text, "Hello")

    async def test_task_tool_calls_are_hidden(self) -> None:
        renderer = RecordingRenderer()
        messages = AsyncItems(
            [
                Message(
                    [
                        {"name": "task", "args": {"description": "delegate"}},
                        {"name": "read_file", "args": {"path": "README.md"}},
                    ]
                )
            ]
        )

        await consume_messages(messages, renderer)

        self.assertEqual(
            renderer.events,
            [
                (
                    "delegation_started",
                    [{"name": "task", "args": {"description": "delegate"}}],
                ),
                ("tool_call", "read_file", {"path": "README.md"}, ""),
            ],
        )

    async def test_compaction_reasoning_is_hidden_behind_status(self) -> None:
        renderer = RecordingRenderer()
        messages = AsyncItems(
            [
                Message(
                    reasoning=AsyncItems(
                        [
                            "Thinking Process:\n\n1. **Analyze the Request:**\n",
                            "* **Role:** Context Extraction Assistant\n",
                            "* **Primary Objective:** Extract the highest quality/most relevant context ",
                            "from the conversation history to replace it due to token limits.",
                        ]
                    )
                )
            ]
        )

        await consume_messages(messages, renderer)

        self.assertEqual(renderer.events, [("compaction_started",), ("compaction_finished",)])

    async def test_leaked_compaction_reasoning_shape_is_hidden_behind_status(self) -> None:
        renderer = RecordingRenderer()
        messages = AsyncItems(
            [
                Message(
                    reasoning=AsyncItems(
                        [
                            "Thinking Process:\n\n",
                            "1. **Analyze the Request:**\n",
                            "   * **Role:** Context Extraction Assistant.\n",
                            "   * **Objective:** Extract the highest quality/most relevant context ",
                            "from the conversation history to replace it due to nearing token limits.\n",
                        ]
                    )
                )
            ]
        )

        await consume_messages(messages, renderer)

        self.assertEqual(renderer.events, [("compaction_started",), ("compaction_finished",)])

    async def test_streamed_structured_summary_text_is_hidden_without_compaction_signal(self) -> None:
        renderer = RecordingRenderer()
        messages = AsyncItems([Message(text=COMPACTION_SUMMARY)])

        await consume_messages(messages, renderer)

        self.assertEqual(renderer.events, [("compaction_started",), ("compaction_finished",)])

    async def test_streamed_compaction_summary_text_is_hidden_after_compaction_reasoning(self) -> None:
        renderer = RecordingRenderer()
        messages = AsyncItems(
            [
                Message(
                    reasoning=AsyncItems(
                        [
                            "Thinking Process:\n\n",
                            "* **Role:** Context Extraction Assistant\n",
                            "* **Primary Objective:** Extract the highest quality/most relevant context ",
                            "from the conversation history to replace it.",
                        ]
                    ),
                    text=COMPACTION_SUMMARY,
                )
            ]
        )

        await consume_messages(messages, renderer)

        self.assertEqual(renderer.events, [("compaction_started",), ("compaction_finished",)])

    async def test_streamed_compaction_summary_prefix_keeps_following_answer_text_after_signal(self) -> None:
        renderer = RecordingRenderer()
        messages = AsyncItems(
            [
                Message(
                    reasoning=AsyncItems(
                        [
                            "Thinking Process:\n\n",
                            "* **Role:** Context Extraction Assistant\n",
                            "* **Primary Objective:** Extract the highest quality/most relevant context ",
                            "from the conversation history to replace it.",
                        ]
                    ),
                    text=AsyncItems(
                        [
                            COMPACTION_SUMMARY[:40],
                            COMPACTION_SUMMARY[40:],
                            "\nThe rain tapped against the window.",
                            " More story followed.",
                        ]
                    )
                )
            ]
        )

        await consume_messages(messages, renderer)

        self.assertEqual(
            renderer.events,
            [
                ("compaction_started",),
                ("compaction_finished",),
                ("text", "The rain tapped against the window."),
                ("text", " More story followed."),
            ],
        )

    async def test_raw_message_stream_preserves_provider_order(self) -> None:
        renderer = RecordingRenderer()
        messages = AsyncItems(
            [
                RawMessageStream(
                    [
                        {"delta": {"type": "reasoning-delta", "reasoning": "User sent a greeting. "}},
                        {"delta": {"type": "text-delta", "text": "Hello"}},
                        {"delta": {"type": "reasoning-delta", "reasoning": "Respond briefly."}},
                        {"delta": {"type": "text-delta", "text": " there"}},
                    ]
                )
            ]
        )

        await consume_messages(messages, renderer)

        self.assertEqual(
            renderer.events,
            [
                ("reasoning", "User sent a greeting. "),
                ("text", "Hello"),
                ("reasoning", "Respond briefly."),
                ("text", " there"),
            ],
        )

    async def test_raw_compaction_reasoning_is_hidden_behind_status(self) -> None:
        renderer = RecordingRenderer()
        messages = AsyncItems(
            [
                RawMessageStream(
                    [
                        {"delta": {"type": "reasoning-delta", "reasoning": "Thinking Process:\n\n"}},
                        {
                            "delta": {
                                "type": "reasoning-delta",
                                "reasoning": "1. **Analyze the Request:**\n* **Role:** Context Extraction Assistant\n",
                            }
                        },
                        {
                            "delta": {
                                "type": "reasoning-delta",
                                "reasoning": "* **Primary Objective:** Extract the highest quality/most relevant context "
                                "from the conversation history to replace it.",
                            }
                        },
                    ]
                )
            ]
        )

        await consume_messages(messages, renderer)

        self.assertEqual(renderer.events, [("compaction_started",), ("compaction_finished",)])

    async def test_raw_structured_summary_text_is_hidden_without_compaction_signal(self) -> None:
        renderer = RecordingRenderer()
        messages = AsyncItems(
            [
                RawMessageStream(
                    [
                        {"delta": {"type": "text-delta", "text": COMPACTION_SUMMARY[:120]}},
                        {"delta": {"type": "text-delta", "text": COMPACTION_SUMMARY[120:]}},
                    ]
                )
            ]
        )

        await consume_messages(messages, renderer)

        self.assertEqual(renderer.events, [("compaction_started",), ("compaction_finished",)])

    async def test_raw_compaction_summary_text_is_hidden_after_compaction_reasoning(self) -> None:
        renderer = RecordingRenderer()
        messages = AsyncItems(
            [
                RawMessageStream(
                    [
                        {"delta": {"type": "reasoning-delta", "reasoning": "Thinking Process:\n\n"}},
                        {
                            "delta": {
                                "type": "reasoning-delta",
                                "reasoning": "* **Role:** Context Extraction Assistant\n"
                                "* **Primary Objective:** Extract the highest quality/most relevant context "
                                "from the conversation history to replace it.",
                            }
                        },
                        {"delta": {"type": "text-delta", "text": COMPACTION_SUMMARY[:120]}},
                        {"delta": {"type": "text-delta", "text": COMPACTION_SUMMARY[120:]}},
                    ]
                )
            ]
        )

        await consume_messages(messages, renderer)

        self.assertEqual(renderer.events, [("compaction_started",), ("compaction_finished",)])

    async def test_streamed_normal_text_renders_first_delta_without_probe_delay(self) -> None:
        renderer = RecordingRenderer()
        messages = AsyncItems([Message(text=AsyncItems(["The rain ", "tapped against the window."]))])

        await consume_messages(messages, renderer)

        self.assertEqual(
            renderer.events,
            [
                ("text", "The rain "),
                ("text", "tapped against the window."),
            ],
        )

    async def test_streamed_normal_markdown_heading_renders_first_delta(self) -> None:
        renderer = RecordingRenderer()
        messages = AsyncItems([Message(text=AsyncItems(["## SUMMARY\n", "This is a normal answer."]))])

        await consume_messages(messages, renderer)

        self.assertEqual(renderer.events, [("text", "## SUMMARY\n"), ("text", "This is a normal answer.")])

    async def test_streamed_markdown_keeps_blank_line_after_first_delta(self) -> None:
        renderer = RecordingRenderer()
        messages = AsyncItems([Message(text=AsyncItems(["## Summary", "\n\nBody text."]))])

        await consume_messages(messages, renderer)

        self.assertEqual(renderer.events, [("text", "## Summary"), ("text", "\n\nBody text.")])

    async def test_long_streamed_summary_prefix_hides_summary_and_renders_tail(self) -> None:
        renderer = RecordingRenderer()
        messages = AsyncItems(
            [
                Message(
                    text=AsyncItems(
                        [
                            COMPACTION_SUMMARY[:120],
                            COMPACTION_SUMMARY[120:],
                            "\nVisible answer.",
                        ]
                    )
                )
            ]
        )

        await consume_messages(messages, renderer)

        self.assertEqual(
            renderer.events,
            [
                ("compaction_started",),
                ("compaction_finished",),
                ("text", "Visible answer."),
            ],
        )

    async def test_streamed_summarization_metadata_text_is_hidden(self) -> None:
        renderer = RecordingRenderer()
        messages = AsyncItems([Message(text="internal summary", additional_kwargs={"lc_source": "summarization"})])

        await consume_messages(messages, renderer)

        self.assertEqual(renderer.events, [])

    async def test_streamed_normal_markdown_heading_text_still_renders(self) -> None:
        renderer = RecordingRenderer()
        text = "## SUMMARY\nThis is a normal answer."
        messages = AsyncItems([Message(text=text)])

        await consume_messages(messages, renderer)

        self.assertEqual(renderer.events, [("text", text)])

    async def test_normal_thinking_process_reasoning_still_renders(self) -> None:
        renderer = RecordingRenderer()
        messages = AsyncItems(
            [
                Message(
                    reasoning=AsyncItems(
                        [
                            "Thinking Process:\n\n",
                            "The user asked for a greeting, so I can answer briefly.",
                        ]
                    )
                )
            ]
        )

        await consume_messages(messages, renderer)

        self.assertEqual(
            renderer.events,
            [("reasoning", "Thinking Process:\n\nThe user asked for a greeting, so I can answer briefly.")],
        )

    async def test_two_task_calls_produce_one_delegation_event(self) -> None:
        renderer = RecordingRenderer()
        messages = AsyncItems(
            [
                Message(
                    [
                        {"name": "task", "args": {"description": "one"}},
                        {"name": "task", "args": {"description": "two"}},
                    ]
                )
            ]
        )

        await consume_messages(messages, renderer)

        self.assertEqual(len(renderer.events), 1)
        self.assertEqual(renderer.events[0][0], "delegation_started")
        self.assertEqual(len(renderer.events[0][1]), 2)

    async def test_subagent_prints_one_header_and_final_call(self) -> None:
        renderer = RecordingRenderer()
        subagent = Subagent(
            "general-purpose [one]",
            [
                ToolCall("grep", {"pattern": "TODO"}, "first output"),
                ToolCall("read_file", {"path": "ui/renderer.py"}, "final output"),
            ],
        )

        await consume_subagent(subagent, renderer)

        self.assertEqual(
            renderer.events,
            [
                ("subagent_started", "general-purpose [one]", "look around"),
                ("subagent_finished", "general-purpose [one]", "final output"),
            ],
        )

    async def test_two_subagents_print_two_headers(self) -> None:
        renderer = RecordingRenderer()

        await consume_subagents(
            AsyncItems(
                [
                    Subagent("general-purpose [one]", [ToolCall("grep", {}, "one")]),
                    Subagent("general-purpose [two]", [ToolCall("grep", {}, "two")]),
                ]
            ),
            renderer,
        )

        headers = [event for event in renderer.events if event[0] == "subagent_started"]
        self.assertEqual(len(headers), 2)

    def test_ask_user_options_keeps_open_ended_option_last(self) -> None:
        options = ask_user_options(
            {
                "options": [
                    "Use A",
                    ASK_USER_OPEN_OPTION,
                    "Use B",
                    "Use A",
                    "",
                ]
            }
        )

        self.assertEqual(options, ["Use A", "Use B", ASK_USER_OPEN_OPTION])

    def test_usage_parser_accepts_lmstudio_native_stats_object(self) -> None:
        stats = type(
            "Stats",
            (),
            {
                "prompt_tokens_count": 8200,
                "predicted_tokens_count": 1424,
                "total_tokens_count": 9624,
            },
        )()
        message = type("Message", (), {"response_metadata": {"stats": stats}})()

        usage = usage_from_message(message)

        self.assertEqual(usage["input_tokens"], 8200)
        self.assertEqual(usage["output_tokens"], 1424)
        self.assertEqual(usage["total_tokens"], 9624)
        self.assertEqual(usage["context_tokens"], 9624)
        self.assertEqual(usage["source"], "response_metadata.stats")

    def test_usage_parser_accepts_lmstudio_native_stats_dict(self) -> None:
        usage = usage_from_message(
            {
                "stats": {
                    "promptTokensCount": 8200,
                    "predictedTokensCount": 1424,
                    "totalTokensCount": 9624,
                }
            }
        )

        self.assertEqual(usage["input_tokens"], 8200)
        self.assertEqual(usage["output_tokens"], 1424)
        self.assertEqual(usage["total_tokens"], 9624)
        self.assertEqual(usage["context_tokens"], 9624)


if __name__ == "__main__":
    unittest.main()
