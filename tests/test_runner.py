"""Tests for runtime event handling and terminal rendering behavior."""

from __future__ import annotations

import unittest
from io import StringIO
from typing import Any, ClassVar
from unittest.mock import patch

from pyfiglet import Figlet
from rich.console import Console

from runtime.message_events import consume_messages
from runtime import runner
from runtime.subagent_events import consume_subagent, consume_subagents
from ui.renderer import Renderer


class AsyncItems:
    """Async iterable test double for DeepAgents event streams."""

    def __init__(self, items: list[Any]) -> None:
        """Store the items that should be yielded asynchronously."""
        self.items = items

    async def __aiter__(self) -> Any:
        """Yield each stored item as an async stream would."""
        for item in self.items:
            yield item


class Message:
    """Fake streamed message containing optional tool calls."""

    def __init__(self, tool_calls: list[Any] | None = None) -> None:
        """Create a message test double with a tool-call list."""
        self.tool_calls = tool_calls or []


class OutputMessage:
    """Fake final message object that exposes a text attribute."""

    def __init__(self, text: str) -> None:
        """Store text in the same place LangChain messages often expose it."""
        self.text = text


class ToolCall:
    """Fake tool call event with name, args, and output."""

    def __init__(self, name: str, args: dict[str, Any], output: Any) -> None:
        """Store tool-call fields used by the runner."""
        self.name = name
        self.args = args
        self.output = output


class Subagent:
    """Fake subagent with a final message shaped like DeepAgents output."""

    def __init__(self, name: str, tool_calls: list[ToolCall]) -> None:
        """Create a subagent test double with final tool output text."""
        self.name = name
        self.task_input = "look around"
        self.tool_calls = AsyncItems(tool_calls)
        self.output = {
            "messages": [
                type("Message", (), {"text": tool_calls[-1].output if tool_calls else ""})()
            ]
        }


class RecordingRenderer:
    """Renderer double that records high-level events instead of printing."""

    def __init__(self) -> None:
        """Create an empty event list."""
        self.events: list[tuple[Any, ...]] = []

    def reasoning_delta(self, value: str) -> None:
        """Record streamed reasoning text."""
        self.events.append(("reasoning", value))

    def text_delta(self, value: str) -> None:
        """Record streamed response text."""
        self.events.append(("text", value))

    def tool_call(self, name: str, args: Any) -> None:
        """Record a rendered tool call."""
        self.events.append(("tool_call", name, args))

    def tool_result(self, name: str, result: str) -> None:
        """Record a rendered tool result."""
        self.events.append(("tool_result", name, result))

    def delegation_started(self, calls: list[dict[str, Any]]) -> None:
        """Record the compact delegation event."""
        self.events.append(("delegation_started", calls))

    def subagent_label(self, subagent: Any) -> str:
        """Use the subagent test-double name directly."""
        return subagent.name

    def subagent_started(self, name: str, task_input: str = "") -> None:
        """Record a subagent start event."""
        self.events.append(("subagent_started", name, task_input))

    def subagent_finished(self, name: str, result: str = "") -> None:
        """Record a subagent finish event."""
        self.events.append(("subagent_finished", name, None, None, result))


class RunTurnRenderer(RecordingRenderer):
    """Renderer double with approval support for full run-turn tests."""

    def __init__(self, decisions: list[dict[str, Any]] | None = None) -> None:
        """Create approval storage and optional canned decisions."""
        super().__init__()
        self.decisions = decisions or [{"type": "approve"}]
        self.approvals: list[list[Any]] = []

    def finish_main(self) -> None:
        """Record the end of the main turn."""
        self.events.append(("finish_main",))

    async def ask_approvals(self, interrupts: list[Any]) -> list[dict[str, Any]]:
        """Return canned approval decisions for interrupts."""
        self.approvals.append(interrupts)
        self.events.append(("ask_approvals", interrupts))
        return self.decisions


class RecordingConsole:
    """Small console double used by renderer tests that do not need Rich."""

    def __init__(self) -> None:
        """Create an empty list of printed lines."""
        self.lines: list[str] = []

    def print(self, *values: Any, **kwargs: Any) -> None:
        """Record printed values as a single text line."""
        self.lines.append(" ".join(str(value) for value in values))


class FakeStream:
    """Fake DeepAgents stream with the four channels the runner consumes."""

    def __init__(self, output: Any = None, interrupts: list[Any] | None = None) -> None:
        """Create empty event channels and optional output/interrupt values."""
        self.messages = AsyncItems([])
        self.tool_calls = AsyncItems([])
        self.subagents = AsyncItems([])
        self.output_value = output or {}
        self.interrupt_values = interrupts or []

    async def output(self) -> Any:
        """Return the configured final-output payload."""
        return self.output_value

    def interrupts(self) -> list[Any]:
        """Return configured stream interrupts."""
        return self.interrupt_values


class FakeAgent:
    """Fake agent that returns prebuilt streams for each invocation."""

    def __init__(self, streams: list[FakeStream]) -> None:
        """Store streams and record each payload the runner sends."""
        self.streams = list(streams)
        self.payloads: list[Any] = []

    async def astream_events(self, payload: Any, config: dict[str, Any], version: str) -> FakeStream:
        """Return the next stream test double and record the payload."""
        self.payloads.append(payload)
        return self.streams.pop(0)


class FakeLive:
    """Replacement for Rich Live that records start/update/stop calls."""

    instances: ClassVar[list["FakeLive"]] = []

    def __init__(self, renderable: Any, console: Any, refresh_per_second: int, transient: bool) -> None:
        """Record the initial renderable and Live configuration."""
        self.renderable = renderable
        self.console = console
        self.refresh_per_second = refresh_per_second
        self.transient = transient
        self.updates: list[Any] = []
        self.started = False
        self.stopped = False
        self.__class__.instances.append(self)

    def start(self) -> None:
        """Record that the live display started."""
        self.started = True

    def update(self, renderable: Any) -> None:
        """Record a live display update."""
        self.updates.append(renderable)

    def stop(self) -> None:
        """Record that the live display stopped."""
        self.stopped = True


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    """Tests for runner event handling and renderer behavior."""

    async def test_run_turn_asks_approval_for_stream_interrupts(self) -> None:
        """A stream interrupt should pause, ask approval, then resume."""
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

    async def test_run_turn_exits_when_stream_has_no_interrupts(self) -> None:
        """A normal stream should finish after one agent invocation."""
        agent = FakeAgent([FakeStream(output={"messages": []})])
        renderer = RunTurnRenderer()

        await runner.run_turn(agent, "no write", renderer, "thread-1")

        self.assertEqual(renderer.approvals, [])
        self.assertEqual(len(agent.payloads), 1)

    async def test_run_turn_supports_output_interrupt_fallback(self) -> None:
        """Interrupts stored in final output should still be handled."""
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
        """Final assistant text should be copied into TurnResult."""
        agent = FakeAgent([FakeStream(output={"messages": [OutputMessage("final plan")]})])
        renderer = RunTurnRenderer()

        result = await runner.run_turn(agent, "plan", renderer, "thread-1")

        self.assertEqual(result.final_text, "final plan")

    async def test_run_turn_records_tool_calls_and_results(self) -> None:
        """Tool names and outputs should be kept for plan validation."""
        agent = FakeAgent(
            [
                FakeStream(
                    output={"messages": []},
                    interrupts=[],
                )
            ]
        )
        agent.streams[0].messages = AsyncItems([Message([{"name": "write_file", "args": {}}])])
        agent.streams[0].tool_calls = AsyncItems([ToolCall("write_file", {}, "permission denied for write on /x")])
        renderer = RunTurnRenderer()

        result = await runner.run_turn(agent, "write", renderer, "thread-1")

        self.assertIn("write_file", result.tool_calls)
        self.assertIn("permission denied for write on /x", result.tool_results)

    async def test_task_tool_calls_are_hidden(self) -> None:
        """The task tool should produce delegation UI, not a normal tool panel."""
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
        """Multiple task tool calls in one message should share one summary."""
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
        """A subagent should render one lifecycle block with final output."""
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
                (
                    "subagent_finished",
                    "general-purpose [one]",
                    None,
                    None,
                    "final output",
                ),
            ],
        )

    async def test_two_subagents_print_two_headers(self) -> None:
        """Two subagents should produce two separate status blocks."""
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

    def test_splash_includes_workspace_metadata_and_hints(self) -> None:
        """The splash should show metadata and useful commands."""
        output = self._splash_output(workspace="D:\\Projects\\mira")

        self.assertIn("Minimal Iterative Reasoning Agent - V1", output)
        self.assertIn("workspace: D:\\Projects\\mira", output)
        self.assertIn("enter to send", output)
        self.assertIn("↑/↓ history", output)
        self.assertIn("/help", output)
        self.assertIn("/tools", output)
        self.assertIn("/plan", output)
        self.assertIn("/act", output)
        self.assertIn("/plans", output)
        self.assertIn("ctrl+c to quit", output)

    def test_splash_does_not_print_workspace_above_logo(self) -> None:
        """Workspace metadata should stay below the logo."""
        output = self._splash_output()
        wordmark = Figlet(font="blocky").renderText("MIRA").rstrip()
        logo_width = max(len(line.rstrip()) for line in wordmark.splitlines())
        lines = output.splitlines()

        self.assertEqual(lines[0].strip(), "=" * logo_width)
        self.assertNotIn("workspace:", lines[0])
        self.assertNotIn("workspace:", lines[1])

    def test_splash_separators_match_logo_width_and_do_not_close(self) -> None:
        """The splash should use logo-width separators without a closing line."""
        output = self._splash_output()
        wordmark = Figlet(font="blocky").renderText("MIRA").rstrip()
        logo_width = max(len(line.rstrip()) for line in wordmark.splitlines())
        lines = [line.strip() for line in output.splitlines()]

        self.assertEqual(lines.count("=" * logo_width), 1)
        self.assertEqual(lines.count("-" * logo_width), 1)
        self.assertNotEqual(lines[-1], "=" * logo_width)

    def test_renderer_newline_prints_blank_line(self) -> None:
        """Renderer.newline should print a clean blank line."""
        renderer = Renderer()
        console = Console(record=True, force_terminal=False, width=100, file=StringIO())
        renderer.console = console

        renderer.newline()

        self.assertEqual(console.export_text(), "\n")

    def test_response_render_preserves_rich_markup_literal(self) -> None:
        """Model text that looks like Rich markup should stay literal."""
        renderer = Renderer()
        renderer._response_text = "[red]literal[/red]"
        console = Console(record=True, force_terminal=False, width=100, file=StringIO())

        console.print(renderer._render_response())
        output = console.export_text()

        self.assertIn("[red]literal[/red]", output)
        self.assertNotIn("\033", output)

    def test_response_streaming_updates_live_without_losing_tokens(self) -> None:
        """Streaming should still append tokens, update Live, and clear on stop."""
        renderer = Renderer()
        FakeLive.instances = []

        with patch("ui.renderer.Live", FakeLive):
            renderer.text_delta("hello")
            renderer.text_delta(" [red]literal[/red]")
            live = FakeLive.instances[0]

            self.assertTrue(live.started)
            self.assertEqual(renderer._response_text, "hello [red]literal[/red]")
            self.assertEqual(len(live.updates), 2)

            renderer._stop_response_live()

        self.assertTrue(live.stopped)
        self.assertIsNone(renderer._response_live)
        self.assertEqual(renderer._response_text, "")
        self.assertEqual(len(live.updates), 3)

    def test_context_compaction_live_panel_starts_and_stops(self) -> None:
        """Context compaction should render a live spinner panel."""
        renderer = Renderer()
        FakeLive.instances = []

        with patch("ui.renderer.Live", FakeLive):
            renderer.context_compaction_started()
            live = FakeLive.instances[0]

            self.assertTrue(live.started)
            self.assertIn("mira - compacting", str(getattr(live.renderable, "title", "")))

            renderer.context_compaction_finished()

        self.assertTrue(live.stopped)
        self.assertIsNone(renderer._context_compaction_live)

    def test_renderer_truncates_final_subagent_output(self) -> None:
        """Tool output should be shortened when a display limit is configured."""
        renderer = Renderer(tool_output_chars=5)

        self.assertEqual(renderer.truncate("abcdefgh"), "abcde ... truncated ...")

    def test_renderer_prints_each_subagent_header_once(self) -> None:
        """Rendering a group should include each subagent title once."""
        renderer = Renderer()
        renderer.console = RecordingConsole()

        renderer.subagent_started("general-purpose [one]")
        renderer.subagent_started("general-purpose [two]")
        renderer.subagent_finished("general-purpose [one]", "done")

        console = Console(record=True, force_terminal=False, width=100, file=StringIO())
        console.print(renderer.render_subagents())
        output = console.export_text()

        self.assertEqual(output.count("subagent - general-purpose [one]"), 1)
        self.assertEqual(output.count("subagent - general-purpose [two]"), 1)

    def test_renderer_renders_running_and_finished_blocks(self) -> None:
        """A finished subagent block should show request, DONE, and output."""
        renderer = Renderer(tool_output_chars=8)
        renderer.console = RecordingConsole()
        renderer.subagent_started("general-purpose [one]", "inspect files")
        renderer.subagent_finished("general-purpose [one]", "abcdefghijklmnopqrstuvwxyz")

        console = Console(record=True, force_terminal=False, width=100, file=StringIO())
        console.print(renderer.render_subagents())
        output = console.export_text()

        self.assertIn("subagent - general-purpose [one]", output)
        self.assertIn("request:", output)
        self.assertIn("DONE", output)
        self.assertIn("output:", output)
        self.assertIn("truncated", output)
        self.assertNotIn("\033", output)

    def test_renderer_renders_running_status(self) -> None:
        """A running subagent block should show the RUNNING status."""
        renderer = Renderer()
        renderer.console = RecordingConsole()
        renderer.subagent_started("general-purpose [one]", "inspect files")

        console = Console(record=True, force_terminal=False, width=100, file=StringIO())
        console.print(renderer.render_subagents())
        output = console.export_text()

        self.assertIn("RUNNING", output)
        self.assertIn("request:", output)

    async def test_renderer_choice_passes_options_by_keyword(self) -> None:
        """The prompt-toolkit choice helper should receive keyword options."""
        renderer = Renderer()

        with patch("ui.renderer.choice", return_value="y") as choice:
            answer = await renderer._choice()

        self.assertEqual(answer, "y")
        self.assertEqual(choice.call_args.args, ("Approve this action?",))
        self.assertEqual(
            choice.call_args.kwargs["options"],
            [("y", "approve"), ("e", "edit"), ("r", "reject")],
        )
        self.assertTrue(choice.call_args.kwargs["show_frame"])

    async def test_renderer_git_repo_prompt_uses_yes_no_menu(self) -> None:
        """The Git creation prompt should use a framed yes/no choice menu."""
        renderer = Renderer()

        with patch("ui.renderer.choice", return_value="y") as choice:
            answer = await renderer.ask_create_git_repo("Create Git?")

        self.assertTrue(answer)
        self.assertEqual(choice.call_args.args, ("Create Git?",))
        self.assertEqual(choice.call_args.kwargs["options"], [("y", "yes"), ("n", "no")])
        self.assertTrue(choice.call_args.kwargs["show_frame"])

    async def test_renderer_continue_without_git_prompt_uses_continue_exit_menu(self) -> None:
        """The Git failure prompt should let the user continue or exit."""
        renderer = Renderer()

        with patch("ui.renderer.choice", return_value="c") as choice:
            answer = await renderer.ask_continue_without_git("Continue?")

        self.assertTrue(answer)
        self.assertEqual(choice.call_args.args, ("Continue?",))
        self.assertEqual(choice.call_args.kwargs["options"], [("c", "continue"), ("e", "exit")])
        self.assertTrue(choice.call_args.kwargs["show_frame"])

    def _splash_output(self, workspace: str = "D:\\Projects\\mira") -> str:
        """Render the splash to an in-memory Rich console and return text."""
        renderer = Renderer()
        console = Console(record=True, force_terminal=False, width=200, file=StringIO())
        renderer.console = console

        renderer.splash(model_name="model", session_id="session-1", workspace=workspace)

        return console.export_text()


if __name__ == "__main__":
    unittest.main()
