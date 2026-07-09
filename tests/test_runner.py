"""Tests for runtime event handling."""

from __future__ import annotations

import asyncio
import unittest
from contextlib import redirect_stdout
from io import StringIO
from typing import Any

from langchain.agents.middleware.summarization import DEFAULT_SUMMARY_PROMPT
from langchain_core.messages import AIMessage, ToolMessage

from agent.compaction import (
    MiraSummarizationMiddleware,
    mark_summarization_engine,
    observe_summarization_counts,
    sanitize_messages_for_archive,
)
from runtime.context_usage import context_usage_scope
from runtime.compaction_state import compaction_active, compaction_scope
from runtime import runner
from runtime.message_events import consume_messages
from runtime.output_events import final_text
from runtime.subagent_events import consume_subagent, consume_subagents
from runtime.tool_events import consume_tool_calls
from runtime.usage import usage_from_message, usage_from_output
from scripts.stream_smoke import raw_event_summary, sse_chunk_summary
from ui.interrupts import ASK_USER_OPEN_OPTION, ask_user_options
from ui.renderer import Renderer
from ui.terminal_colors import strip_ansi


class AsyncItems:
    """Async iterable test double for DeepAgents event streams."""

    def __init__(self, items: list[Any]) -> None:
        self.items = items

    async def __aiter__(self) -> Any:
        for item in self.items:
            yield item


class DelayedAsyncItems(AsyncItems):
    """Async iterable that yields after a short delay."""

    def __init__(self, items: list[Any], delay: float = 0.01) -> None:
        super().__init__(items)
        self.delay = delay

    async def __aiter__(self) -> Any:
        await asyncio.sleep(self.delay)
        async for item in super().__aiter__():
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


class IncompleteToolCall(ToolCall):
    """Fake interrupted tool call whose output must not be awaited yet."""

    completed = False


class DocumentedToolCall:
    """Fake DeepAgents tool-call stream item using documented field names."""

    def __init__(
        self,
        tool_name: str,
        input: dict[str, Any],
        *,
        output_deltas: Any | None = None,
        output: Any | None = None,
        error: Any | None = None,
        call_id: str = "",
    ) -> None:
        self.tool_name = tool_name
        self.input = input
        self.output_deltas = output_deltas
        self.output = output
        self.error = error
        self.completed = True
        self.id = call_id


class SingleSubscriptionDeltas:
    """Async deltas stream that fails if subscribed more than once."""

    def __init__(self, items: list[Any]) -> None:
        self.items = items
        self.subscribers = 0

    async def __aiter__(self) -> Any:
        self.subscribers += 1
        if self.subscribers > 1:
            raise RuntimeError("StreamChannel already has a subscriber; use .atee(n) for fan-out.")
        for item in self.items:
            yield item


class DoubleSubscribeToolCall:
    """Tool call shaped like DeepAgents ToolCallStream."""

    def __init__(self) -> None:
        self.tool_name = "eval"
        self.input = {"code": "1 + 1"}
        self.output_deltas = SingleSubscriptionDeltas(["<stdout>ok</stdout>", "<result>2</result>"])
        self.output = None
        self.error = None
        self.completed = True
        self.id = "call-single"

    def __aiter__(self) -> Any:
        return self.output_deltas.__aiter__()


class BlockingOutput:
    """Awaitable test double that blocks until released."""

    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.awaited = False

    def __await__(self) -> Any:
        return self._wait().__await__()

    async def _wait(self) -> str:
        self.awaited = True
        await self.release.wait()
        return "blocked task output"


class AsyncToolCallList:
    """Tool call list that streams chunks before exposing finalized calls."""

    def __init__(self, calls: list[Any], chunks: list[Any] | None = None) -> None:
        self.calls = calls
        self.chunks = chunks or [{}]

    async def __aiter__(self) -> Any:
        for chunk in self.chunks:
            yield chunk

    def get(self) -> list[Any]:
        return self.calls


class BlockingReasoning:
    """Reasoning stream that stays open until released."""

    def __init__(self) -> None:
        self.release = asyncio.Event()

    async def __aiter__(self) -> Any:
        yield "Thinking about delegation."
        await self.release.wait()


class Subagent:
    """Fake subagent with a final message shaped like DeepAgents output."""

    def __init__(
        self,
        name: str,
        tool_calls: list[ToolCall],
        task_input: str = "look around",
        path: list[str] | None = None,
    ) -> None:
        self.name = name
        self.task_input = task_input
        self.path = path or []
        self.tool_calls = AsyncItems(tool_calls)
        self.output = {
            "messages": [
                type("Message", (), {"text": tool_calls[-1].output if tool_calls else ""})()
            ]
        }


class ContentListSubagent:
    """Fake subagent whose final message stores text in LangChain content blocks."""

    name = "general-purpose [one]"
    task_input = "find dead code"

    def __init__(self, text: str) -> None:
        self.output = {"messages": [type("Message", (), {"content": [{"type": "text", "text": text}]})()]}


class HangingSubagent:
    """Subagent whose output runs until cancelled."""

    def __init__(self) -> None:
        self.name = "general-purpose [one]"
        self.task_input = "keep working"
        self.cancelled = False

    @property
    def output(self) -> Any:
        return self._wait()

    async def _wait(self) -> str:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return ""


class RaisingSubagents:
    """Subagent stream that fails after starting one child."""

    def __init__(self, subagent: HangingSubagent) -> None:
        self.subagent = subagent

    async def __aiter__(self) -> Any:
        yield self.subagent
        await asyncio.sleep(0)
        raise RuntimeError("subagent stream failed")


class RecordingRenderer:
    """Renderer double that records high-level events."""

    def __init__(self) -> None:
        self.events: list[tuple[Any, ...]] = []

    def reasoning_delta(self, value: str) -> None:
        self.events.append(("reasoning", value))

    def discard_reasoning(self) -> None:
        self.events.append(("discard_reasoning",))

    def text_delta(self, value: str) -> None:
        self.events.append(("text", value))

    def model_activity(self) -> None:
        self.events.append(("model_activity",))

    def tool_call_delta(self, name: str, args: Any, call_id: str = "") -> None:
        self.events.append(("tool_call_delta", name, args, call_id))

    def delegation_delta(self, calls: list[dict[str, Any]]) -> None:
        self.events.append(("delegation_delta", calls))

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

    def subagent_started(self, name: str, task_input: str = "", *, origin: str = "") -> None:
        event = ("subagent_started", name, task_input)
        if origin:
            event = (*event, origin)
        self.events.append(event)

    def subagent_request_updated(self, name: str, task_input: str) -> None:
        self.events.append(("subagent_request_updated", name, task_input))

    def subagent_finished(self, name: str, result: str = "") -> None:
        self.events.append(("subagent_finished", name, result))

    def subagents_cancelled(self) -> None:
        self.events.append(("subagents_cancelled",))

    def system_message(self, text: str, *, kind: str = "system") -> None:
        self.events.append(("system_message", kind, text))

    def discard_last_assistant(self) -> None:
        self.events.append(("discard_last_assistant",))


class RunTurnRenderer(RecordingRenderer):
    """Renderer double with approval support for full run-turn tests."""

    def __init__(self, decisions: list[dict[str, Any]] | None = None, ask_user_answer: str = "Use B") -> None:
        super().__init__()
        self.decisions = decisions or [{"type": "approve"}]
        self.approvals: list[list[Any]] = []
        self.ask_user_answer = ask_user_answer
        self.ask_user_prompts: list[Any] = []
        self.plan_prompts: list[Any] = []

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

    async def present_plan(self, interrupt: Any) -> str:
        self.plan_prompts.append(interrupt)
        self.events.append(("present_plan", interrupt))
        return "Plan presented for user review."


class FakeStream:
    """Fake DeepAgents stream with the channels the runner consumes."""

    def __init__(
        self,
        output: Any = None,
        interrupts: list[Any] | None = None,
        tool_calls: list[Any] | None = None,
        custom_events: list[Any] | None = None,
    ) -> None:
        self.messages = AsyncItems([])
        self.tool_calls = AsyncItems(tool_calls or [])
        self.subagents = AsyncItems([])
        self.custom = AsyncItems(custom_events or [])
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

    async def astream_events(self, payload: Any, config: dict[str, Any], version: str, **kwargs: Any) -> FakeStream:
        self.payloads.append(payload)
        return self.streams.pop(0)


class FakeBackend:
    """Minimal backend used to prove runner tests do not execute fallbacks."""

    def __init__(self) -> None:
        self.writes: list[tuple[str, str]] = []
        self.commands: list[tuple[str, int | None]] = []

    def write(self, file_path: str, content: str) -> object:
        self.writes.append((file_path, content))
        return type("WriteResult", (), {"error": None})()

    def execute(self, command: str, *, timeout: int | None = None) -> object:
        self.commands.append((command, timeout))
        return type("ExecuteResult", (), {"output": "env list", "exit_code": 0, "truncated": False})()


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

    async def test_run_turn_presents_plan_interrupt_as_terminal_turn(self) -> None:
        interrupt = {
            "type": "present_plan",
            "title": "Plan",
            "summary": ["Do one thing."],
            "key_changes": ["Change code."],
            "test_plan": ["Run focused checks."],
            "assumptions": ["Execute is available."],
        }
        agent = FakeAgent(
            [
                FakeStream(output={"messages": []}, interrupts=[interrupt]),
            ]
        )
        renderer = RunTurnRenderer()

        result = await runner.run_turn(agent, "plan", renderer, "thread-1")

        self.assertEqual(renderer.plan_prompts, [interrupt])
        self.assertEqual(renderer.ask_user_prompts, [])
        self.assertEqual(renderer.approvals, [])
        self.assertEqual(len(agent.payloads), 1)
        self.assertEqual(result.final_text, "")

    async def test_present_plan_tool_stream_is_not_rendered(self) -> None:
        call = {
            "name": "present_plan",
            "id": "call-plan",
            "args": {"title": "Plan"},
            "completed": True,
            "output": "interrupt",
        }
        agent = FakeAgent([FakeStream(output={"messages": []}, tool_calls=[call])])
        renderer = RunTurnRenderer()

        await runner.run_turn(agent, "plan", renderer, "thread-1")

        self.assertNotIn(("tool_call", "present_plan", {"title": "Plan"}, "call-plan"), renderer.events)
        self.assertNotIn(("tool_result", "present_plan", "interrupt", "call-plan"), renderer.events)

    async def test_run_turn_exits_when_stream_has_no_interrupts(self) -> None:
        agent = FakeAgent([FakeStream(output={"messages": []})])
        renderer = RunTurnRenderer()

        await runner.run_turn(agent, "no write", renderer, "thread-1")

        self.assertEqual(renderer.approvals, [])
        self.assertEqual(len(agent.payloads), 1)

    async def test_run_turn_preserves_high_context_completion_without_retry(self) -> None:
        agent = FakeAgent(
            [
                FakeStream(
                    output={
                        "messages": [
                            OutputMessage(
                                "Since it",
                                usage_metadata={"input_tokens": 9900, "output_tokens": 12, "total_tokens": 9912},
                            )
                        ]
                    }
                )
            ]
        )
        renderer = RunTurnRenderer()

        result = await runner.run_turn(
            agent,
            "answer",
            renderer,
            "thread-1",
        )

        self.assertEqual(len(agent.payloads), 1)
        self.assertEqual(result.final_text, "Since it")
        self.assertEqual(result.context_tokens, 0)
        self.assertNotIn(("discard_last_assistant",), renderer.events)

    async def test_run_turn_fills_blank_subagent_request_from_task_description(self) -> None:
        stream = FakeStream(output={"messages": []})
        stream.tool_calls = AsyncItems(
            [
                DocumentedToolCall(
                    "task",
                    {"description": "find the README"},
                    call_id="task-1",
                )
            ]
        )
        stream.subagents = AsyncItems(
            [
                Subagent(
                    "general-purpose [one]",
                    [ToolCall("read_file", {"path": "README.md"}, "done")],
                    task_input="",
                )
            ]
        )
        renderer = RunTurnRenderer()

        await runner.run_turn(FakeAgent([stream]), "delegate", renderer, "thread-1")

        self.assertIn(("subagent_started", "general-purpose [one]", "find the README"), renderer.events)

    async def test_run_turn_hides_tool_namespace_subagent_without_task(self) -> None:
        stream = FakeStream(output={"messages": []})
        stream.subagents = AsyncItems(
            [
                Subagent(
                    "general-purpose [one]",
                    [ToolCall("noop", {}, "one")],
                    task_input="",
                    path=["tools:eval-call"],
                )
            ]
        )
        renderer = RunTurnRenderer()

        await runner.run_turn(FakeAgent([stream]), "delegate via eval", renderer, "thread-1")

        self.assertNotIn(("subagent_started", "general-purpose [one]", ""), renderer.events)
        self.assertNotIn(("subagent_finished", "general-purpose [one]", "one"), renderer.events)

    async def test_run_turn_renders_eval_custom_subagents(self) -> None:
        stream = FakeStream(
            output={"messages": []},
            custom_events=[
                {
                    "type": "subagent",
                    "phase": "start",
                    "id": "ptc_task_one",
                    "eval_id": "eval-1",
                    "subagent_type": "general-purpose",
                    "label": "satire",
                    "description": "tell a satire joke",
                },
                {
                    "type": "subagent",
                    "phase": "start",
                    "id": "ptc_task_two",
                    "eval_id": "eval-1",
                    "subagent_type": "general-purpose",
                    "label": "knock-knock",
                    "description": "tell a knock-knock joke",
                },
                {"type": "subagent", "phase": "complete", "id": "ptc_task_two"},
                {"type": "subagent", "phase": "complete", "id": "ptc_task_one"},
            ],
        )
        renderer = RunTurnRenderer()

        await runner.run_turn(FakeAgent([stream]), "delegate via eval", renderer, "thread-1")

        self.assertIn(("subagent_started", "general-purpose [satire]", "tell a satire joke", "eval_subagent"), renderer.events)
        self.assertIn(
            ("subagent_started", "general-purpose [knock-knock]", "tell a knock-knock joke", "eval_subagent"),
            renderer.events,
        )
        self.assertIn(("subagent_finished", "general-purpose [satire]", ""), renderer.events)
        self.assertIn(("subagent_finished", "general-purpose [knock-knock]", ""), renderer.events)

    async def test_run_turn_keeps_tool_namespace_subagent_with_task_request_ordinary(self) -> None:
        stream = FakeStream(output={"messages": []})
        stream.tool_calls = AsyncItems(
            [
                DocumentedToolCall(
                    "task",
                    {"description": "tell a pun"},
                    call_id="task-1",
                )
            ]
        )
        stream.subagents = AsyncItems(
            [
                Subagent(
                    "general-purpose [one]",
                    [ToolCall("noop", {}, "one")],
                    task_input="",
                    path=["tools:task-call"],
                )
            ]
        )
        renderer = RunTurnRenderer()

        await runner.run_turn(FakeAgent([stream]), "delegate", renderer, "thread-1")

        self.assertIn(("subagent_started", "general-purpose [one]", "tell a pun"), renderer.events)
        self.assertNotIn(
            ("subagent_started", "general-purpose [one]", "tell a pun", "dynamic_tool_subagent"),
            renderer.events,
        )

    async def test_run_turn_maps_multiple_task_requests_to_subagents_in_order(self) -> None:
        stream = FakeStream(output={"messages": []})
        stream.tool_calls = AsyncItems(
            [
                DocumentedToolCall("task", {"description": "write sacred story"}, call_id="task-1"),
                DocumentedToolCall("task", {"description": "write funny story"}, call_id="task-2"),
            ]
        )
        stream.subagents = AsyncItems(
            [
                Subagent("general-purpose [one]", [ToolCall("noop", {}, "one")], task_input=""),
                Subagent("general-purpose [two]", [ToolCall("noop", {}, "two")], task_input=""),
            ]
        )
        renderer = RunTurnRenderer()

        await runner.run_turn(FakeAgent([stream]), "delegate", renderer, "thread-1")

        starts = [event for event in renderer.events if event[0] == "subagent_started"]
        self.assertEqual(
            starts,
            [
                ("subagent_started", "general-purpose [one]", "write sacred story"),
                ("subagent_started", "general-purpose [two]", "write funny story"),
            ],
        )

    async def test_run_turn_preserves_explicit_subagent_request_and_consumes_queue(self) -> None:
        stream = FakeStream(output={"messages": []})
        stream.tool_calls = AsyncItems(
            [
                DocumentedToolCall("task", {"description": "queued one"}, call_id="task-1"),
                DocumentedToolCall("task", {"description": "queued two"}, call_id="task-2"),
            ]
        )
        stream.subagents = AsyncItems(
            [
                Subagent("general-purpose [one]", [ToolCall("noop", {}, "one")], task_input="provided request"),
                Subagent("general-purpose [two]", [ToolCall("noop", {}, "two")], task_input=""),
            ]
        )
        renderer = RunTurnRenderer()

        await runner.run_turn(FakeAgent([stream]), "delegate", renderer, "thread-1")

        starts = [event for event in renderer.events if event[0] == "subagent_started"]
        self.assertEqual(
            starts,
            [
                ("subagent_started", "general-purpose [one]", "provided request"),
                ("subagent_started", "general-purpose [two]", "queued two"),
            ],
        )

    async def test_run_turn_uses_tool_calls_as_only_final_task_source(self) -> None:
        stream = FakeStream(output={"messages": []})
        stream.messages = AsyncItems(
            [
                Message(
                    [
                        {"id": "task-1", "name": "task", "args": {"description": "scary"}},
                        {"id": "task-2", "name": "task", "args": {"description": "funny"}},
                    ]
                )
            ]
        )
        stream.tool_calls = AsyncItems(
            [
                DocumentedToolCall("task", {"description": "scary"}, call_id="task-1"),
                DocumentedToolCall("task", {"description": "funny"}, call_id="task-2"),
            ]
        )
        stream.subagents = AsyncItems(
            [
                Subagent("general-purpose [one]", [ToolCall("noop", {}, "one")], task_input=""),
                Subagent("general-purpose [two]", [ToolCall("noop", {}, "two")], task_input=""),
            ]
        )
        renderer = RunTurnRenderer()

        await runner.run_turn(FakeAgent([stream]), "delegate", renderer, "thread-1")

        delegations = [event for event in renderer.events if event[0] == "delegation_started"]
        self.assertEqual(len(delegations), 2)
        self.assertEqual([event[1][0]["id"] for event in delegations], ["task-1", "task-2"])

    async def test_run_turn_patches_blank_subagent_request_when_task_arrives_late(self) -> None:
        stream = FakeStream(output={"messages": []})
        stream.subagents = AsyncItems(
            [
                Subagent("general-purpose [one]", [ToolCall("noop", {}, "one")], task_input=""),
            ]
        )
        stream.tool_calls = DelayedAsyncItems(
            [
                DocumentedToolCall("task", {"description": "late request"}, call_id="task-1"),
            ]
        )
        renderer = RunTurnRenderer()

        await runner.run_turn(FakeAgent([stream]), "delegate", renderer, "thread-1")

        self.assertIn(("subagent_started", "general-purpose [one]", ""), renderer.events)
        self.assertIn(("subagent_request_updated", "general-purpose [one]", "late request"), renderer.events)

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

    def test_final_text_ignores_ai_message_tool_call_repr(self) -> None:
        """Structured tool-call messages should not be persisted as assistant prose."""
        message = AIMessage(
            content=[
                {"type": "reasoning", "reasoning": "Need to write a file."},
                {"type": "text", "text": "\n\n"},
                {
                    "type": "tool_call",
                    "id": "call-write",
                    "name": "write_file",
                    "args": {"file_path": "/story.txt", "content": "hello"},
                },
            ],
            tool_calls=[
                {
                    "name": "write_file",
                    "args": {"file_path": "/story.txt", "content": "hello"},
                    "id": "call-write",
                }
            ],
        )

        self.assertEqual(final_text({"messages": [message]}), "")

    def test_final_text_ignores_tool_messages(self) -> None:
        """Tool output should not be mistaken for assistant prose."""
        message = ToolMessage(content="raw tool output", name="execute", tool_call_id="call-execute")

        self.assertEqual(final_text({"messages": [message]}), "")

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
        self.assertEqual(result.usage["context_tokens"], 0)

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

    def test_commit_loop_usage_uses_provider_total_when_reported(self) -> None:
        result = runner.TurnResult()

        usage = result.commit_loop_usage(
            {"messages": [OutputMessage("done", {"input_tokens": 1400, "output_tokens": 67, "total_tokens": 1467})]},
        )

        self.assertEqual(usage["input_tokens"], 1400)
        self.assertEqual(usage["output_tokens"], 67)
        self.assertEqual(usage["context_tokens"], 0)
        self.assertEqual(usage["context_source"], "unknown")
        self.assertEqual(result.usage["input_tokens"], 1400)
        self.assertEqual(result.usage["output_tokens"], 67)
        self.assertEqual(result.usage["context_tokens"], 0)
        self.assertEqual(result.usage["context_source"], "unknown")

    def test_commit_loop_usage_uses_provider_pair_above_visible_estimate(self) -> None:
        result = runner.TurnResult()

        result.add_usage(
            {
                "input_tokens": 9467,
                "output_tokens": 123,
                "context_tokens": 454,
                "context_source": "langchain_approx.count_tokens",
                "source": "usage_metadata",
            }
        )

        self.assertEqual(result.usage["input_tokens"], 9467)
        self.assertEqual(result.usage["output_tokens"], 123)
        self.assertEqual(result.usage["context_tokens"], 0)
        self.assertEqual(result.usage["context_source"], "unknown")

    def test_observed_summarization_count_returns_original_value_once(self) -> None:
        calls = []

        class Summarization:
            def __init__(self) -> None:
                self.calls = 0

            def _count_tokens(self, *args: Any, **kwargs: Any) -> int:
                self.calls += 1
                return 1234

        summarization = Summarization()

        observe_summarization_counts(summarization)
        first_wrapper = summarization._count_tokens
        observe_summarization_counts(summarization)

        with context_usage_scope(calls.append):
            total = summarization._count_tokens(["message"], tools=[])

        self.assertEqual(total, 1234)
        self.assertEqual(summarization.calls, 1)
        self.assertIs(summarization._count_tokens, first_wrapper)
        self.assertEqual(calls[0]["context_tokens"], 1234)
        self.assertEqual(calls[0]["context_source"], "deepagents.summarization._count_tokens")

    def test_mira_summarization_middleware_has_distinct_exclusion_name(self) -> None:
        """DeepAgents should not remove MIRA's observed summarizer with its default one."""
        middleware = object.__new__(MiraSummarizationMiddleware)

        self.assertEqual(middleware.name, "MiraSummarizationMiddleware")

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
        self.assertEqual(usage["context_tokens"], 0)

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
        self.assertEqual(result.usage["context_tokens"], 0)

    async def test_run_turn_does_not_invent_context_when_usage_is_missing(self) -> None:
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
        )

        self.assertEqual(result.usage["input_tokens"], 0)
        self.assertEqual(result.usage["output_tokens"], 0)
        self.assertEqual(result.usage["context_tokens"], 0)
        self.assertEqual(result.usage["source"], "unknown")

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
        )

        self.assertEqual(result.usage["input_tokens"], 5512)
        self.assertEqual(result.usage["output_tokens"], 91)
        self.assertEqual(result.usage["total_tokens"], 5603)
        self.assertEqual(result.usage["context_tokens"], 0)

    async def test_run_turn_records_tool_calls_and_results(self) -> None:
        agent = FakeAgent([FakeStream(output={"messages": []}, interrupts=[])])
        agent.streams[0].messages = AsyncItems([Message([{"id": "call-1", "name": "write_file", "args": {}}])])
        agent.streams[0].tool_calls = AsyncItems([ToolCall("write_file", {}, "permission denied for write on /x", "call-1")])
        renderer = RunTurnRenderer()

        result = await runner.run_turn(agent, "write", renderer, "thread-1")

        self.assertEqual(result.tool_calls, ["write_file"])
        self.assertIn("permission denied for write on /x", result.tool_results)

    async def test_run_turn_rejects_output_level_ai_message_tool_calls_without_execution(self) -> None:
        """Final AIMessage tool calls should not be recorded as executed tools."""
        message = AIMessage(
            content=[
                {"type": "text", "text": "\n\n"},
                {
                    "type": "tool_call",
                    "id": "call-write",
                    "name": "write_file",
                    "args": {"file_path": "/story.txt", "content": "hello"},
                },
            ],
            tool_calls=[
                {
                    "name": "write_file",
                    "args": {"file_path": "/story.txt", "content": "hello"},
                    "id": "call-write",
                }
            ],
        )
        agent = FakeAgent([FakeStream(output={"messages": [message]}, interrupts=[])])
        renderer = RunTurnRenderer()

        with self.assertRaisesRegex(RuntimeError, "unexecuted tool call"):
            await runner.run_turn(agent, "write", renderer, "thread-1")

        self.assertNotIn(
            ("tool_call", "write_file", {"file_path": "/story.txt", "content": "hello"}, "call-write"),
            renderer.events,
        )

    async def test_run_turn_native_resume_records_approved_filesystem_result(self) -> None:
        """Approved write calls should resume natively into tool output and LLM reply."""
        interrupt = {
            "action_requests": [
                {
                    "name": "write_file",
                    "args": {"file_path": "/story.txt", "content": "hello"},
                }
            ],
            "review_configs": [
                {
                    "action_name": "write_file",
                    "allowed_decisions": ["approve", "edit", "reject"],
                }
            ],
        }
        agent = FakeAgent(
            [
                FakeStream(output={"messages": []}, interrupts=[interrupt]),
                FakeStream(
                    output={
                        "messages": [
                            ToolMessage(
                                content="Successfully wrote to /story.txt",
                                name="write_file",
                                tool_call_id="call-write",
                            ),
                            OutputMessage("Wrote the story."),
                        ]
                    },
                    interrupts=[],
                ),
            ]
        )
        backend = FakeBackend()
        agent.mira_backend = backend
        renderer = RunTurnRenderer(decisions=[{"type": "approve"}])

        result = await runner.run_turn(agent, "write", renderer, "thread-1")

        self.assertEqual(backend.writes, [])
        self.assertEqual(result.final_text, "Wrote the story.")
        self.assertIn("Successfully wrote to /story.txt", result.tool_results)
        self.assertIn(("tool_result", "write_file", "Successfully wrote to /story.txt", "call-write"), renderer.events)
        self.assertEqual(len(agent.payloads), 2)
        self.assertEqual(agent.payloads[1].resume, {"decisions": [{"type": "approve"}]})
        self.assertEqual(agent.payloads[1].goto, ())

    async def test_run_turn_native_resume_records_approved_execute_result(self) -> None:
        """Approved execute calls should resume natively into tool output and LLM reply."""
        interrupt = {
            "action_requests": [
                {
                    "name": "execute",
                    "args": {"command": "conda info --envs"},
                }
            ],
            "review_configs": [
                {
                    "action_name": "execute",
                    "allowed_decisions": ["approve", "edit", "reject"],
                }
            ],
        }
        agent = FakeAgent(
            [
                FakeStream(output={"messages": []}, interrupts=[interrupt]),
                FakeStream(
                    output={
                        "messages": [
                            ToolMessage(content="env list", name="execute", tool_call_id="call-execute"),
                            OutputMessage("The envs are ai_agents and base. Here is a joke."),
                        ]
                    },
                    interrupts=[],
                ),
            ]
        )
        backend = FakeBackend()
        agent.mira_backend = backend
        renderer = RunTurnRenderer(decisions=[{"type": "approve"}])

        result = await runner.run_turn(agent, "execute", renderer, "thread-1")

        self.assertEqual(backend.commands, [])
        self.assertEqual(result.final_text, "The envs are ai_agents and base. Here is a joke.")
        self.assertNotEqual(result.final_text, "env list")
        self.assertIn("env list", result.tool_results)
        self.assertIn(("tool_result", "execute", "env list", "call-execute"), renderer.events)
        self.assertEqual(len(agent.payloads), 2)
        self.assertEqual(agent.payloads[1].resume, {"decisions": [{"type": "approve"}]})
        self.assertEqual(agent.payloads[1].goto, ())

    async def test_run_turn_raises_when_native_resume_returns_unexecuted_approved_tool(self) -> None:
        """Approved tools should not be manually executed if native resume leaves them pending."""
        interrupt = {"action_requests": [{"name": "execute", "args": {"command": "conda info --envs"}}]}
        message = AIMessage(
            content=[
                {
                    "type": "tool_call",
                    "id": "call-execute",
                    "name": "execute",
                    "args": {"command": "conda info --envs"},
                },
            ],
            tool_calls=[
                {
                    "name": "execute",
                    "args": {"command": "conda info --envs"},
                    "id": "call-execute",
                }
            ],
        )
        agent = FakeAgent(
            [
                FakeStream(output={"messages": []}, interrupts=[interrupt]),
                FakeStream(output={"messages": [message]}, interrupts=[]),
            ]
        )
        backend = FakeBackend()
        agent.mira_backend = backend
        renderer = RunTurnRenderer(decisions=[{"type": "approve"}])

        with self.assertRaisesRegex(RuntimeError, "native HITL resume returned unexecuted tool call") as caught:
            await runner.run_turn(agent, "execute", renderer, "thread-1")

        error_text = str(caught.exception)
        self.assertIn("diagnostic:", error_text)
        self.assertIn("interrupted=False", error_text)
        self.assertIn("stream_tool_calls_observed=False", error_text)
        self.assertIn("AIMessage", error_text)
        self.assertIn("execute", error_text)
        self.assertEqual(len(agent.payloads), 2)
        self.assertEqual(agent.payloads[1].resume, {"decisions": [{"type": "approve"}]})
        self.assertEqual(agent.payloads[1].goto, ())
        self.assertEqual(backend.commands, [])

    async def test_run_turn_does_not_execute_rejected_execute(self) -> None:
        """Rejected execute calls should remain unexecuted."""
        interrupt = {
            "action_requests": [
                {
                    "name": "execute",
                    "args": {"command": "conda env list"},
                }
            ]
        }
        message = AIMessage(
            content=[
                {
                    "type": "tool_call",
                    "id": "call-execute",
                    "name": "execute",
                    "args": {"command": "conda env list"},
                },
            ],
            tool_calls=[
                {
                    "name": "execute",
                    "args": {"command": "conda env list"},
                    "id": "call-execute",
                }
            ],
        )
        agent = FakeAgent(
            [
                FakeStream(output={"messages": []}, interrupts=[interrupt]),
                FakeStream(output={"messages": [message]}, interrupts=[]),
            ]
        )
        backend = FakeBackend()
        agent.mira_backend = backend
        renderer = RunTurnRenderer(decisions=[{"type": "reject"}])

        with self.assertRaisesRegex(RuntimeError, "unexecuted tool call"):
            await runner.run_turn(agent, "execute", renderer, "thread-1")

        self.assertEqual(backend.commands, [])

    async def test_run_turn_uses_tool_stream_as_canonical_normal_tool_display(self) -> None:
        agent = FakeAgent([FakeStream(output={"messages": []}, interrupts=[])])
        agent.streams[0].messages = AsyncItems([Message([{"name": "read_file", "args": {"path": "README.md"}}])])
        agent.streams[0].tool_calls = AsyncItems([ToolCall("read_file", {"path": "README.md"}, "contents")])
        renderer = RunTurnRenderer()

        result = await runner.run_turn(agent, "read", renderer, "thread-1")

        tool_blocks = [event for event in renderer.events if event[0] == "tool_call" and event[1] == "read_file"]
        self.assertEqual(len(tool_blocks), 1)
        self.assertEqual(result.tool_calls, ["read_file"])

    async def test_run_turn_records_execute_result_from_final_output_tool_message(self) -> None:
        """Always-allow execute should persist the result even if the tool stream omits output."""
        agent = FakeAgent(
            [
                FakeStream(
                    output={
                        "messages": [
                            ToolMessage(content="env list", name="execute", tool_call_id="call-execute"),
                            OutputMessage("The envs are ai_agents and base."),
                        ]
                    },
                    interrupts=[],
                )
            ]
        )
        agent.streams[0].tool_calls = AsyncItems(
            [IncompleteToolCall("execute", {"command": "conda env list"}, "", "call-execute")]
        )
        renderer = RunTurnRenderer()

        result = await runner.run_turn(agent, "execute", renderer, "thread-1")

        self.assertEqual(result.tool_calls, ["execute"])
        self.assertEqual(result.tool_results, ["env list"])
        self.assertEqual(result.final_text, "The envs are ai_agents and base.")
        self.assertIn(("tool_result", "execute", "env list", "call-execute"), renderer.events)

    async def test_run_turn_deduplicates_tool_stream_and_final_output_results(self) -> None:
        """Final output ToolMessages should not duplicate already streamed tool results."""
        agent = FakeAgent(
            [
                FakeStream(
                    output={
                        "messages": [
                            ToolMessage(content="env list", name="execute", tool_call_id="call-execute"),
                            OutputMessage("Done."),
                        ]
                    },
                    interrupts=[],
                )
            ]
        )
        agent.streams[0].tool_calls = AsyncItems(
            [ToolCall("execute", {"command": "conda env list"}, "env list", "call-execute")]
        )
        renderer = RunTurnRenderer()

        result = await runner.run_turn(agent, "execute", renderer, "thread-1")

        self.assertEqual(result.tool_results, ["env list"])
        result_events = [event for event in renderer.events if event[:2] == ("tool_result", "execute")]
        self.assertEqual(result_events, [("tool_result", "execute", "env list", "call-execute")])

    async def test_tool_call_stream_accepts_documented_fields(self) -> None:
        renderer = RecordingRenderer()
        result = runner.TurnResult()
        calls = AsyncItems(
            [
                DocumentedToolCall(
                    "eval",
                    {"code": "1 + 1"},
                    output_deltas=AsyncItems(["<stdout>ok</stdout>", "<result>2</result>"]),
                    call_id="call-doc",
                )
            ]
        )

        await consume_tool_calls(calls, renderer, result)

        self.assertEqual(result.tool_calls, ["eval"])
        self.assertEqual(result.tool_results, ["<stdout>ok</stdout><result>2</result>"])
        self.assertEqual(
            renderer.events,
            [
                ("tool_call", "eval", {"code": "1 + 1"}, "call-doc"),
                ("tool_result", "eval", "<stdout>ok</stdout><result>2</result>", "call-doc"),
            ],
        )

    async def test_tool_call_stream_does_not_double_subscribe_output_deltas(self) -> None:
        renderer = RecordingRenderer()
        result = runner.TurnResult()
        call = DoubleSubscribeToolCall()

        await consume_tool_calls(AsyncItems([call]), renderer, result)

        self.assertEqual(call.output_deltas.subscribers, 1)
        self.assertEqual(result.tool_results, ["<stdout>ok</stdout><result>2</result>"])
        self.assertEqual(
            renderer.events,
            [
                ("tool_call", "eval", {"code": "1 + 1"}, "call-single"),
                ("tool_result", "eval", "<stdout>ok</stdout><result>2</result>", "call-single"),
            ],
        )

    async def test_incomplete_tool_call_stream_item_does_not_await_output(self) -> None:
        renderer = RecordingRenderer()
        result = runner.TurnResult()
        blocked = BlockingOutput()

        await consume_tool_calls(AsyncItems([IncompleteToolCall("write_file", {"file_path": "/x"}, blocked, "call-1")]), renderer, result)

        self.assertFalse(blocked.awaited)
        self.assertEqual(result.tool_calls, ["write_file"])
        self.assertEqual(result.tool_results, [])
        self.assertEqual(renderer.events, [("tool_call", "write_file", {"file_path": "/x"}, "call-1")])

    async def test_task_tool_calls_emit_immediately_without_waiting_for_output(self) -> None:
        renderer = RecordingRenderer()
        result = runner.TurnResult()
        blocked = BlockingOutput()
        calls = AsyncItems(
            [
                DocumentedToolCall("task", {"description": "one"}, output=blocked, call_id="task-1"),
                DocumentedToolCall("task", {"description": "two"}, call_id="task-2"),
            ]
        )

        await consume_tool_calls(calls, renderer, result)

        self.assertFalse(blocked.awaited)
        self.assertEqual(result.tool_calls, ["task", "task"])
        self.assertEqual(result.tool_results, [])
        self.assertEqual(
            [event[1][0]["id"] for event in renderer.events if event[0] == "delegation_started"],
            ["task-1", "task-2"],
        )

    async def test_tool_call_stream_prefers_documented_error_field(self) -> None:
        renderer = RecordingRenderer()
        calls = AsyncItems(
            [
                {
                    "id": "call-error",
                    "tool_name": "eval",
                    "input": {"code": "bad()"},
                    "completed": True,
                    "output_deltas": ["partial output"],
                    "error": "boom",
                }
            ]
        )

        await consume_tool_calls(calls, renderer)

        self.assertEqual(
            renderer.events,
            [
                ("tool_call", "eval", {"code": "bad()"}, "call-error"),
                ("tool_result", "eval", "boom", "call-error"),
            ],
        )

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

    async def test_message_finalized_task_calls_are_hidden_in_runner_mode(self) -> None:
        renderer = RecordingRenderer()
        messages = AsyncItems([Message([{"name": "task", "args": {"description": "delegate"}}])])

        await consume_messages(messages, renderer, render_normal_tools=False)

        self.assertEqual(renderer.events, [])

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

    async def test_long_compaction_reasoning_preamble_is_buffered_until_classified(self) -> None:
        renderer = RecordingRenderer()
        messages = AsyncItems(
            [
                Message(
                    reasoning=AsyncItems(
                        [
                            "Thinking Process:\n\n" + ("Review the existing conversation. " * 60),
                            "\n* **Role:** Context Extraction Assistant\n",
                            "* **Primary Objective:** Extract context from the conversation history ",
                            "to replace it due to token limits.",
                        ]
                    )
                )
            ]
        )

        await consume_messages(messages, renderer)

        self.assertEqual(renderer.events, [("compaction_started",), ("compaction_finished",)])

    async def test_langchain_default_summary_prompt_reasoning_is_hidden(self) -> None:
        renderer = RecordingRenderer()
        messages = AsyncItems(
            [
                Message(
                    reasoning=AsyncItems(
                        [
                            "I need to follow the internal instructions.\n\n",
                            DEFAULT_SUMMARY_PROMPT.split("{messages}", 1)[0],
                        ]
                    )
                )
            ]
        )

        await consume_messages(messages, renderer)

        self.assertEqual(
            renderer.events,
            [
                ("reasoning", "I need to follow the internal instructions.\n\n"),
                ("discard_reasoning",),
                ("compaction_started",),
                ("compaction_finished",),
            ],
        )

    async def test_explicit_compaction_signal_hides_reasoning_and_text(self) -> None:
        renderer = RecordingRenderer()
        messages = AsyncItems(
            [
                Message(
                    reasoning=AsyncItems(["I am extracting context from the transcript."]),
                    text=AsyncItems(["SESSION INTENT\n", "This should not be recorded."]),
                )
            ]
        )

        with compaction_scope():
            await consume_messages(messages, renderer)

        self.assertEqual(
            renderer.events,
            [("compaction_started",), ("compaction_finished",)],
        )

    async def test_normal_turn_after_explicit_compaction_signal_still_renders(self) -> None:
        renderer = RecordingRenderer()

        with compaction_scope():
            await consume_messages(
                AsyncItems([Message(reasoning=AsyncItems(["internal summary"]), text=AsyncItems(["hidden"]))]),
                renderer,
            )
        await consume_messages(
            AsyncItems([Message(reasoning=AsyncItems(["thinking"]), text=AsyncItems(["visible answer"]))]),
            renderer,
        )

        self.assertEqual(
            renderer.events,
            [
                ("compaction_started",),
                ("compaction_finished",),
                ("reasoning", "thinking"),
                ("text", "visible answer"),
            ],
        )

    async def test_explicit_compaction_signal_hides_raw_message_stream(self) -> None:
        renderer = RecordingRenderer()
        messages = AsyncItems(
            [
                RawMessageStream(
                    [
                        {"type": "reasoning-delta", "reasoning": "internal"},
                        {"type": "text-delta", "text": "hidden"},
                    ]
                )
            ]
        )

        with compaction_scope():
            await consume_messages(messages, renderer)

        self.assertEqual(
            renderer.events,
            [("compaction_started",), ("compaction_finished",)],
        )

    async def test_compaction_after_tool_result_is_hidden_before_next_answer(self) -> None:
        renderer = RecordingRenderer()
        result = runner.TurnResult()

        await consume_tool_calls(AsyncItems([ToolCall("read_file", {"path": "x"}, "contents", "call-1")]), renderer, result)
        with compaction_scope():
            await consume_messages(AsyncItems([Message(text=AsyncItems(["internal summary"]))]), renderer)
        await consume_messages(AsyncItems([Message(text=AsyncItems(["now answering"]))]), renderer)

        self.assertEqual(
            renderer.events,
            [
                ("tool_call", "read_file", {"path": "x"}, "call-1"),
                ("tool_result", "read_file", "contents", "call-1"),
                ("compaction_started",),
                ("compaction_finished",),
                ("text", "now answering"),
            ],
        )

    async def test_summarization_engine_wrapper_marks_sync_and_async_summary_calls(self) -> None:
        class Engine:
            def _create_summary(self) -> bool:
                return compaction_active()

            async def _acreate_summary(self) -> bool:
                return compaction_active()

        engine = Engine()
        mark_summarization_engine(engine)

        self.assertTrue(engine._create_summary())
        self.assertTrue(await engine._acreate_summary())
        self.assertFalse(compaction_active())

    def test_compaction_archive_sanitizer_strips_reasoning_internals(self) -> None:
        message = AIMessage(
            content=[
                {"type": "reasoning", "reasoning": "internal chain of thought"},
                {"type": "text", "text": "Visible reply."},
                {
                    "type": "tool_call",
                    "id": "call-write",
                    "name": "write_file",
                    "args": {"file_path": "/story.txt", "content": "hello"},
                },
            ],
            additional_kwargs={
                "reasoning_content": "internal chain of thought",
                "tool_calls": ["ChoiceDeltaToolCall(...)"],
            },
            tool_calls=[
                {
                    "name": "write_file",
                    "args": {"file_path": "/story.txt", "content": "hello"},
                    "id": "call-write",
                }
            ],
        )

        sanitized = sanitize_messages_for_archive([message])
        rendered = repr(sanitized[0])

        self.assertIn("Visible reply.", rendered)
        self.assertIn("Tool call: write_file", rendered)
        self.assertNotIn("reasoning_content", rendered)
        self.assertNotIn("internal chain", rendered)
        self.assertNotIn("ChoiceDeltaToolCall", rendered)

    async def test_summary_extraction_reasoning_after_compaction_is_hidden(self) -> None:
        renderer = RecordingRenderer()
        messages = AsyncItems(
            [
                Message(
                    reasoning=AsyncItems(
                        [
                            "The user wants me to extract context from a conversation history that has already been summarized. ",
                            "The conversation history has been saved to a file and a condensed summary is provided. ",
                            "My task is to extract the most relevant context to replace this conversation history. ",
                            "I should structure this according to the required format (SESSION INTENT, SUMMARY, ARTIFACTS, NEXT STEPS).",
                        ]
                    )
                )
            ]
        )

        await consume_messages(messages, renderer)

        self.assertEqual(renderer.events, [("compaction_started",), ("compaction_finished",)])

    async def test_partial_summary_extraction_reasoning_is_hidden_on_finish(self) -> None:
        renderer = RecordingRenderer()
        messages = AsyncItems(
            [
                Message(
                    reasoning=AsyncItems(
                        [
                            "The user is asking me to extract context from a conversation history that has already ",
                            "been summarized. This appears to be a meta-task where I need to create a",
                        ]
                    )
                )
            ]
        )

        await consume_messages(messages, renderer)

        self.assertEqual(renderer.events, [("compaction_started",), ("compaction_finished",)])

    async def test_leaked_session_compaction_reasoning_shape_is_hidden(self) -> None:
        renderer = RecordingRenderer()
        messages = AsyncItems(
            [
                Message(
                    reasoning=AsyncItems(
                        [
                            "The user wants me to extract context from the conversation history. Looking at the messages provided:\n\n",
                            "1. Human asked to write a 10-word short story to a file\n",
                            "2. The AI responded with reasoning and made a tool call to write_file.\n\n",
                            "## SESSION INTENT\nWrite a 10-word short story to a file.\n\n",
                            "## SUMMARY\nThe task has been completed.\n\n",
                            "## ARTIFACTS\nFile created: `/mira-short-story.txt`.\n\n",
                            "## NEXT STEPS\nNone.\n\n",
                            "Let me structure this properly following the instructions.\n",
                        ]
                    )
                )
            ]
        )

        await consume_messages(messages, renderer)

        self.assertEqual(renderer.events, [("compaction_started",), ("compaction_finished",)])

    async def test_screenshot_style_compaction_reasoning_is_hidden(self) -> None:
        renderer = RecordingRenderer()
        messages = AsyncItems(
            [
                Message(
                    reasoning=AsyncItems(
                        [
                            'The user wants me to extract context from the conversation history provided. This is a "context compaction test" ',
                            "where I need to summarize the most important information so that the conversation history can be replaced with this summary, freeing up token space.\n\n",
                            "Looking at the conversation:\n1. Human asked for an 800-word plain text field report in chat.\n",
                            "For context extraction purposes:\n- SESSION INTENT: Write a report.\n- SUMMARY: The task is done.\n",
                            "- ARTIFACTS: None.\n- NEXT STEPS: None - the task is complete.\n\n",
                            "Let me structure this properly according to the instructions:",
                        ]
                    )
                )
            ]
        )

        await consume_messages(messages, renderer)

        self.assertEqual(renderer.events, [("compaction_started",), ("compaction_finished",)])

    async def test_late_compaction_detection_discards_visible_reasoning(self) -> None:
        renderer = RecordingRenderer()
        messages = AsyncItems(
            [
                Message(
                    reasoning=AsyncItems(
                        [
                            "I am reviewing a normal request. " * 50,
                            "Context Extraction Assistant: your sole objective is to extract the highest quality/most relevant context.",
                        ]
                    )
                )
            ]
        )

        await consume_messages(messages, renderer)

        self.assertEqual(
            renderer.events,
            [
                ("reasoning", "I am reviewing a normal request. " * 50),
                ("discard_reasoning",),
                ("compaction_started",),
                ("compaction_finished",),
            ],
        )

    async def test_most_important_context_compaction_reasoning_is_hidden(self) -> None:
        renderer = RecordingRenderer()
        messages = AsyncItems(
            [
                Message(
                    reasoning=AsyncItems(
                        [
                            "The user wants me to extract the most important context from this conversation history. ",
                            "Let me analyze what's happened:\n\n",
                            "1. User requested to write a 10-word short story to a file\n",
                            "2. A tool call was made to write_file\n\n",
                            "Key information to extract:\n",
                            "- Session intent: User wants a short story written to a file\n",
                            "- Summary: Story content was created\n",
                            "- Artifacts: File /mira-short-story.txt\n",
                            "- Next Steps: Verify the file was written successfully\n\n",
                            "Let me structure this properly according to the instructions.",
                        ]
                    )
                )
            ]
        )

        await consume_messages(messages, renderer)

        self.assertEqual(renderer.events, [("compaction_started",), ("compaction_finished",)])

    async def test_long_summary_extraction_reasoning_with_visible_headings_is_hidden(self) -> None:
        renderer = RecordingRenderer()
        messages = AsyncItems(
            [
                Message(
                    reasoning=AsyncItems(
                        [
                            "The user wants me to extract context from the conversation history. ",
                            "This is a simple task where I need to summarize what happened and what remains to be done. ",
                            "Looking at the conversation:\n",
                            "1. User asked for a 10-word short story written to a file\n",
                            "2. A tool call was made to write_file but it did not complete\n",
                            "This preamble is intentionally long. " * 45,
                            "\n\n## SESSION INTENT\nCreate a 10-word short story.\n\n",
                            "## SUMMARY\nThe write did not complete.\n\n",
                            "## ARTIFACTS\nNone.\n\n## NEXT STEPS\nRead the file only after it exists.",
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

    async def test_streamed_summary_heading_variants_are_hidden(self) -> None:
        renderer = RecordingRenderer()
        summary = """### Session Intent
User requested a story.

### Summary:
The conversation was summarized.

### Artifacts
None.

### Next Steps
Await further instructions.
"""
        messages = AsyncItems([Message(text=summary)])

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

    async def test_raw_tool_call_chunks_render_draft_without_final_tool_call(self) -> None:
        renderer = RecordingRenderer()
        messages = AsyncItems(
            [
                RawMessageStream(
                    [
                        {"delta": {"type": "tool_call_chunk", "name": "ls", "args": "{\"path\""}},
                        {"content_block": {"type": "function_call", "name": "ls", "arguments": "{\"path\":\"/\"}"}},
                    ]
                )
            ]
        )

        await consume_messages(messages, renderer, render_normal_tools=False)

        self.assertEqual(
            renderer.events,
            [
                ("tool_call_delta", "ls", '{"path"', "index:0"),
                ("tool_call_delta", "ls", {"path": "/"}, "index:0"),
            ],
        )

    async def test_langchain_block_delta_tool_call_chunk_renders_draft(self) -> None:
        renderer = RecordingRenderer()
        messages = AsyncItems(
            [
                RawMessageStream(
                    [
                        {
                            "event": "content-block-delta",
                            "index": 0,
                            "delta": {
                                "type": "block-delta",
                                "fields": {
                                    "type": "tool_call_chunk",
                                    "id": "call-task",
                                    "name": "task",
                                    "args": "",
                                    "index": 0,
                                },
                            },
                        },
                        {
                            "event": "content-block-delta",
                            "index": 0,
                            "delta": {
                                "type": "block-delta",
                                "fields": {
                                    "type": "tool_call_chunk",
                                    "args": '{"description":"write scary","subagent_type":"general-purpose"}',
                                    "index": 0,
                                },
                            },
                        },
                    ]
                )
            ]
        )

        await consume_messages(messages, renderer, render_normal_tools=False)

        self.assertEqual(
            renderer.events,
            [
                (
                    "delegation_delta",
                    [
                        {
                            "type": "tool_call",
                            "id": "call-task",
                            "name": "task",
                            "args": {},
                        }
                    ],
                ),
                (
                    "delegation_delta",
                    [
                        {
                            "type": "tool_call",
                            "id": "call-task",
                            "name": "task",
                            "args": {"description": "write scary", "subagent_type": "general-purpose"},
                        }
                    ],
                ),
            ],
        )

    def test_raw_stream_smoke_summary_detects_tool_call_chunk(self) -> None:
        summary = raw_event_summary(
            {
                "method": "messages",
                "params": {
                    "namespace": [],
                    "data": [
                        {
                            "event": "content-block-delta",
                            "delta": {
                                "type": "tool_call_chunk",
                                "id": "call-task",
                                "name": "task",
                                "args": '{"description":"write scary',
                            },
                        }
                    ],
                },
            }
        )

        self.assertEqual(summary["method"], "messages")
        self.assertEqual(summary["payload_kind"], "list:content-block-delta")
        self.assertEqual(summary["protocol_event"], "content-block-delta")
        self.assertEqual(summary["delta_type"], "tool_call_chunk")
        self.assertEqual(
            summary["tool_like"],
            {
                "type": "tool_call_chunk",
                "id": "call-task",
                "name": "task",
                "args_sample": '{"description":"write scary',
            },
        )

    def test_raw_stream_smoke_summary_detects_provider_function_call_block(self) -> None:
        summary = raw_event_summary(
            {
                "method": "messages",
                "params": {
                    "namespace": ["agent:abc"],
                    "data": [
                        {
                            "event": "content-block-delta",
                            "content_block": {
                                "type": "function_call",
                                "id": "call-read",
                                "name": "read_file",
                                "arguments": '{"path":"README.md"}',
                            },
                        }
                    ],
                },
            }
        )

        self.assertEqual(summary["namespace"], "agent")
        self.assertEqual(summary["content_block_type"], "function_call")
        self.assertEqual(summary["tool_like"]["type"], "function_call")
        self.assertEqual(summary["tool_like"]["name"], "read_file")

    def test_raw_stream_smoke_summary_scans_tuple_payloads(self) -> None:
        summary = raw_event_summary(
            {
                "method": "messages",
                "params": {
                    "namespace": [],
                    "data": (
                        {
                            "content_block": {
                                "type": "function_call",
                                "id": "call-task",
                                "name": "task",
                                "arguments": '{"description":"draft"}',
                            }
                        },
                        {"metadata": "ignored"},
                    ),
                },
            }
        )

        self.assertEqual(summary["payload_kind"], "tuple:dict")
        self.assertEqual(summary["tool_like"]["name"], "task")
        self.assertEqual(summary["tool_like"]["args_sample"], '{"description":"draft"}')

    def test_raw_stream_smoke_summary_samples_opaque_block_delta(self) -> None:
        summary = raw_event_summary(
            {
                "method": "messages",
                "params": {
                    "namespace": [],
                    "data": (
                        {
                            "event": "content-block-delta",
                            "delta": {
                                "type": "block-delta",
                                "index": 1,
                                "args": '{"description":"partial task',
                            },
                        },
                    ),
                },
            }
        )

        self.assertEqual(summary["delta_type"], "block-delta")
        self.assertEqual(
            summary["protocol_sample"],
            {"delta": {"type": "block-delta", "index": "1", "args": '{"description":"partial task'}},
        )

    def test_sse_chunk_summary_detects_openai_tool_call_arguments(self) -> None:
        summary = sse_chunk_summary(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-task",
                                    "type": "function",
                                    "function": {
                                        "name": "task",
                                        "arguments": '{"description":"write story"}',
                                    },
                                }
                            ]
                        }
                    }
                ]
            }
        )

        self.assertEqual(summary["payload_kind"], "chat.completion.chunk")
        self.assertEqual(summary["delta_keys"], ["tool_calls"])
        self.assertEqual(
            summary["tool_like"],
            {
                "type": "function",
                "id": "call-task",
                "name": "task",
                "args_sample": '{"description":"write story"}',
            },
        )

    def test_sse_chunk_summary_ignores_plain_content_as_tool_call(self) -> None:
        summary = sse_chunk_summary({"choices": [{"delta": {"content": "hello"}}]})

        self.assertEqual(summary["sample"], "hello")
        self.assertNotIn("tool_like", summary)

    async def test_async_tool_call_field_renders_draft_before_final_call(self) -> None:
        renderer = RecordingRenderer()
        messages = AsyncItems(
            [
                Message(
                    tool_calls=AsyncToolCallList(
                        [{"name": "read_file", "args": {"path": "README.md"}}],
                        chunks=[
                            {
                                "type": "tool_call_chunk",
                                "id": "call-read",
                                "name": "read_file",
                                "args": "",
                                "index": 0,
                            },
                            {
                                "type": "tool_call_chunk",
                                "id": "call-read",
                                "args": '{"path":"README.md"}',
                                "index": 0,
                            },
                        ],
                    )
                )
            ]
        )

        await consume_messages(messages, renderer)

        self.assertEqual(
            renderer.events,
            [
                ("tool_call_delta", "read_file", {}, "call-read"),
                ("tool_call_delta", "read_file", {"path": "README.md"}, "call-read"),
                ("tool_call", "read_file", {"path": "README.md"}, ""),
            ],
        )

    async def test_task_tool_call_chunks_render_delegation_draft_before_final_call(self) -> None:
        renderer = RecordingRenderer()
        messages = AsyncItems(
            [
                Message(
                    tool_calls=AsyncToolCallList(
                        [
                            {
                                "id": "call-task",
                                "name": "task",
                                "args": {
                                    "description": "summarize README",
                                    "subagent_type": "general-purpose",
                                },
                            }
                        ],
                        chunks=[
                            {
                                "type": "tool_call_chunk",
                                "id": "call-task",
                                "name": "task",
                                "args": '{"description":"summarize',
                                "index": 0,
                            },
                            {
                                "type": "tool_call_chunk",
                                "id": "call-task",
                                "args": ' README","subagent_type":"general-purpose"}',
                                "index": 0,
                            },
                        ],
                    )
                )
            ]
        )

        await consume_messages(messages, renderer)

        self.assertEqual(
            renderer.events,
            [
                (
                    "delegation_delta",
                    [
                        {
                            "type": "tool_call",
                            "id": "call-task",
                            "name": "task",
                            "args": {"description": "summarize"},
                        }
                    ],
                ),
                (
                    "delegation_delta",
                    [
                        {
                            "type": "tool_call",
                            "id": "call-task",
                            "name": "task",
                            "args": {
                                "description": "summarize README",
                                "subagent_type": "general-purpose",
                            },
                        }
                    ],
                ),
                (
                    "delegation_started",
                    [
                        {
                            "id": "call-task",
                            "name": "task",
                            "args": {
                                "description": "summarize README",
                                "subagent_type": "general-purpose",
                            },
                        }
                    ],
                ),
            ],
        )

    async def test_tool_call_chunks_stream_while_reasoning_stream_is_open(self) -> None:
        renderer = RecordingRenderer()
        reasoning = BlockingReasoning()
        messages = AsyncItems(
            [
                Message(
                    reasoning=reasoning,
                    tool_calls=AsyncToolCallList(
                        [{"id": "call-read", "name": "read_file", "args": {"path": "README.md"}}],
                        chunks=[
                            {
                                "type": "tool_call_chunk",
                                "id": "call-read",
                                "name": "read_file",
                                "args": '{"path":"README.md"}',
                                "index": 0,
                            }
                        ],
                    ),
                )
            ]
        )

        task = asyncio.create_task(consume_messages(messages, renderer))
        for _ in range(20):
            if any(event[0] == "tool_call_delta" for event in renderer.events):
                break
            await asyncio.sleep(0.01)

        self.assertIn(("tool_call_delta", "read_file", {"path": "README.md"}, "call-read"), renderer.events)
        reasoning.release.set()
        await task

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

    async def test_terminal_renderer_preserves_hitl_style_reasoning_newline_chunks(self) -> None:
        renderer = Renderer()
        messages = AsyncItems(
            [
                Message(
                    reasoning=AsyncItems(
                        [
                            "The user wants:",
                            "\n",
                            "1. Check envs",
                            "\n",
                            "2. Tell joke",
                        ]
                    )
                )
            ]
        )
        output = StringIO()

        with redirect_stdout(output):
            await consume_messages(messages, renderer)
            renderer.finish_main()

        rendered = output.getvalue()
        self.assertIn("\033[", rendered)
        self.assertEqual(strip_ansi(rendered), "\nthinking:\nThe user wants:\n1. Check envs\n2. Tell joke\n")

    def test_terminal_renderer_keeps_dynamic_subagent_origin_quiet(self) -> None:
        renderer = Renderer()
        output = StringIO()

        with redirect_stdout(output):
            renderer.subagent_started("general-purpose [one]", "", origin="dynamic_tool_subagent")

        text = strip_ansi(output.getvalue())
        self.assertIn("subagent - general-purpose [one]:", text)
        self.assertNotIn("from eval/tool", text)
        self.assertNotIn("eval/tool-created subagent", text)
        self.assertNotIn("\ntask:\n", text)

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

    async def test_subagent_result_reads_langchain_content_blocks(self) -> None:
        renderer = RecordingRenderer()

        await consume_subagent(ContentListSubagent("subagent report"), renderer)

        self.assertIn(("subagent_finished", "general-purpose [one]", "subagent report"), renderer.events)

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

    async def test_subagent_stream_error_cancels_running_children(self) -> None:
        renderer = RecordingRenderer()
        subagent = HangingSubagent()

        with self.assertRaisesRegex(RuntimeError, "subagent stream failed"):
            await consume_subagents(RaisingSubagents(subagent), renderer)

        self.assertTrue(subagent.cancelled)
        self.assertIn(("subagents_cancelled",), renderer.events)
        self.assertFalse(any(event[0] == "subagent_finished" for event in renderer.events))

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

    def test_usage_parser_accepts_lmstudio_n_tokens_context_count(self) -> None:
        usage = usage_from_message(
            {
                "stats": {
                    "promptTokensCount": 1400,
                    "predictedTokensCount": 67,
                    "n_tokens": 10013,
                }
            }
        )

        self.assertEqual(usage["input_tokens"], 1400)
        self.assertEqual(usage["output_tokens"], 67)
        self.assertEqual(usage["total_tokens"], 10013)
        self.assertEqual(usage["context_tokens"], 10013)


if __name__ == "__main__":
    unittest.main()
