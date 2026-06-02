"""Tests for runtime event handling."""

from __future__ import annotations

import unittest
from typing import Any

from runtime import runner
from runtime.message_events import consume_messages
from runtime.subagent_events import consume_subagent, consume_subagents
from ui.interrupts import ASK_USER_OPEN_OPTION, ask_user_options


class AsyncItems:
    """Async iterable test double for DeepAgents event streams."""

    def __init__(self, items: list[Any]) -> None:
        self.items = items

    async def __aiter__(self) -> Any:
        for item in self.items:
            yield item


class Message:
    """Fake streamed message containing optional tool calls."""

    def __init__(self, tool_calls: list[Any] | None = None) -> None:
        self.tool_calls = tool_calls or []


class OutputMessage:
    """Fake final message object that exposes a text attribute."""

    def __init__(self, text: str, usage_metadata: dict[str, int] | None = None) -> None:
        self.text = text
        self.usage_metadata = usage_metadata or {}


class ToolCall:
    """Fake tool call event with name, args, and output."""

    def __init__(self, name: str, args: dict[str, Any], output: Any) -> None:
        self.name = name
        self.args = args
        self.output = output


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

    def tool_call(self, name: str, args: Any) -> None:
        self.events.append(("tool_call", name, args))

    def tool_result(self, name: str, result: str) -> None:
        self.events.append(("tool_result", name, result))

    def delegation_started(self, calls: list[dict[str, Any]]) -> None:
        self.events.append(("delegation_started", calls))

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
        self.assertEqual(result.usage["context_tokens"], 5512)

    async def test_run_turn_records_tool_calls_and_results(self) -> None:
        agent = FakeAgent([FakeStream(output={"messages": []}, interrupts=[])])
        agent.streams[0].messages = AsyncItems([Message([{"name": "write_file", "args": {}}])])
        agent.streams[0].tool_calls = AsyncItems([ToolCall("write_file", {}, "permission denied for write on /x")])
        renderer = RunTurnRenderer()

        result = await runner.run_turn(agent, "write", renderer, "thread-1")

        self.assertIn("write_file", result.tool_calls)
        self.assertIn("permission denied for write on /x", result.tool_results)

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
                ("tool_call", "read_file", {"path": "README.md"}),
            ],
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


if __name__ == "__main__":
    unittest.main()
