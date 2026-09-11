"""Focused tests for process-local live subagent inspection."""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from langchain_core.messages import ToolMessage

from core.execution.inspection.live import InspectionEvent, LiveInspectionStore
from core.execution.inspection.subagents import (
    SubagentInspectionCapture,
    SubagentInspectionCoordinator,
    capture_child_streams,
    live_inspection_store,
)
from core.execution.streams.subagents import consume_subagent
from core.interface import FrontendEmitter
from ui.shared.adapter import RendererAdapter
from agent.middleware.code_interpreter import runtime_with_task_callbacks


class AsyncItems:
    def __init__(self, items: list[Any]) -> None:
        self.items = items

    async def __aiter__(self) -> Any:
        for item in self.items:
            yield item


class FailingAsyncItems:
    async def __aiter__(self) -> Any:
        raise RuntimeError("projection failed")
        yield  # pragma: no cover - keeps this an async generator


class StreamedMessage:
    def __init__(self, reasoning: list[str], text: list[str]) -> None:
        self.reasoning = AsyncItems(reasoning)
        self.text = AsyncItems(text)
        self.tool_calls: list[Any] = []
        self.additional_kwargs: dict[str, Any] = {}
        self.message_id = "message-one"


class ToolCall:
    completed = True
    output_deltas = None
    error = None

    def __init__(self, name: str, args: dict[str, Any], output: Any, call_id: str) -> None:
        self.tool_name = name
        self.input = args
        self.output = output
        self.id = call_id


class LiveInspectionStoreTests(unittest.TestCase):
    def test_request_is_first_and_parent_facing_response_is_last(self) -> None:
        store = LiveInspectionStore()
        inspection_id = store.allocate_id("task-call")
        store.start(inspection_id, "researcher [fox]", "Complete unshortened request")
        store.append_delta(inspection_id, "reasoning", "first ")
        store.append_delta(inspection_id, "reasoning", "second")
        store.append_delta(inspection_id, "assistant", "streamed draft")
        store.append(inspection_id, InspectionEvent("tool_result", text="later", name="read_file"))
        store.finish(inspection_id, status="DONE", final_response="exact parent response")

        inspection = store.get(inspection_id)
        assert inspection is not None
        self.assertEqual(inspection.events[0], InspectionEvent("user", text="Complete unshortened request"))
        self.assertEqual(inspection.events[1].text, "first second")
        self.assertEqual(inspection.events[-1], InspectionEvent("assistant", text="exact parent response"))

    def test_mid_run_subscription_receives_later_events_without_losing_history(self) -> None:
        store = LiveInspectionStore()
        inspection_id = store.allocate_id()
        store.start(inspection_id, "worker [owl]", "inspect")
        store.append_delta(inspection_id, "reasoning", "already captured")
        seen: list[str] = []

        def listener(_inspection_id: str, update: Any) -> None:
            if update.event is not None:
                seen.append(update.event.text)

        store.subscribe(inspection_id, listener)
        store.append_delta(inspection_id, "reasoning", " and live")
        store.unsubscribe(inspection_id, listener)
        store.append_delta(inspection_id, "assistant", "after close")

        inspection = store.get(inspection_id)
        assert inspection is not None
        self.assertEqual(seen, [" and live"])
        self.assertEqual(
            [event.text for event in inspection.events],
            ["inspect", "already captured and live", "after close"],
        )

    def test_frontend_wrappers_expose_store_and_forward_inspection_metadata(self) -> None:
        class Sink:
            def __init__(self) -> None:
                self.live_inspections = LiveInspectionStore()
                self.received: dict[str, str] = {}

            def subagent_started(
                self,
                _name: str,
                _task: str,
                *,
                inspection_id: str = "",
            ) -> None:
                self.received = {"inspection_id": inspection_id}

            eval_subagent_started = subagent_started

        sink = Sink()
        adapter = RendererAdapter(sink)
        emitter = FrontendEmitter(adapter)
        wrapper = SimpleNamespace(renderer=emitter)

        emitter.eval_subagent_started(
            "worker [fox]",
            "inspect",
            row_id="eval-row",
            inspection_id="subagent:call",
        )

        self.assertIs(live_inspection_store(wrapper), sink.live_inspections)
        self.assertEqual(
            sink.received,
            {"inspection_id": "subagent:call"},
        )

class SubagentCaptureTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_child_streams_capture_reasoning_text_tools_and_errors(self) -> None:
        store = LiveInspectionStore()
        inspection_id = store.allocate_id("native")
        store.start(inspection_id, "researcher [fox]", "research fully")
        success = ToolMessage(content="README contents", tool_call_id="read", status="success")
        failure = ToolMessage(content="missing file", tool_call_id="bad", status="error")
        subagent = SimpleNamespace(
            messages=AsyncItems([StreamedMessage(["think ", "carefully"], ["working answer"])]),
            tool_calls=AsyncItems(
                [
                    ToolCall("read_file", {"path": "README.md"}, success, "read"),
                    ToolCall("read_file", {"path": "missing"}, failure, "bad"),
                ]
            ),
        )
        capture = SubagentInspectionCapture(store, inspection_id)

        await capture_child_streams(subagent, capture)
        capture.ensure_final_response("returned answer")

        inspection = store.get(inspection_id)
        assert inspection is not None
        kinds = [event.kind for event in inspection.events]
        self.assertEqual(kinds[0], "user")
        self.assertEqual(kinds.count("tool_call"), 2)
        self.assertIn("tool_result", kinds)
        self.assertIn("tool_error", kinds)
        self.assertEqual(next(event.text for event in inspection.events if event.kind == "reasoning"), "think carefully")
        self.assertEqual(inspection.events[-1].text, "returned answer")

    async def test_eval_coordinator_reuses_custom_row_inspection(self) -> None:
        store = LiveInspectionStore()
        coordinator = SubagentInspectionCoordinator(store)
        inspection_id = coordinator.eval_started(
            {
                "id": "ptc-task-one",
                "eval_id": "eval-call",
                "description": "full eval task",
                "subagent_type": "researcher",
            }
        )

        claimed = coordinator.claim_eval_child(
            SimpleNamespace(path=("tools:eval-call", "researcher:child"), task_input="")
        )

        self.assertEqual(claimed, inspection_id)
        inspection = store.get(inspection_id)
        assert inspection is not None
        self.assertEqual(inspection.events[0].text, "full eval task")

    async def test_eval_protocol_captures_siblings_missing_from_high_level_lane(self) -> None:
        store = LiveInspectionStore()
        coordinator = SubagentInspectionCoordinator(store)
        first_id = coordinator.eval_started(
            {
                "id": "ptc-task-a",
                "eval_id": "eval-call",
                "description": "Report EVAL-A",
                "subagent_type": "general-purpose",
            }
        )
        second_id = coordinator.eval_started(
            {
                "id": "ptc-task-b",
                "eval_id": "eval-call",
                "description": "Report EVAL-B",
                "subagent_type": "general-purpose",
            }
        )

        coordinator.handle_protocol_event(
            {
                "method": "values",
                "params": {
                    "namespace": ["general-purpose:child-a"],
                    "data": {
                        "messages": [
                            {"type": "human", "content": "Report EVAL-A"},
                            {
                                "type": "ai",
                                "content": [
                                    {"type": "reasoning", "reasoning": "check A"},
                                    {"type": "text", "text": "EVAL-A"},
                                ],
                            },
                        ]
                    },
                },
            }
        )
        coordinator.handle_protocol_event(
            {
                "method": "values",
                "params": {
                    "namespace": ["general-purpose:child-b"],
                    "data": {
                        "messages": [
                            {"type": "human", "content": "Report EVAL-B"},
                            {"type": "ai", "content": "EVAL-B"},
                        ]
                    },
                },
            }
        )

        first = store.get(first_id)
        second = store.get(second_id)
        assert first is not None and second is not None
        self.assertEqual(first.events[0].text, "Report EVAL-A")
        self.assertEqual(first.events[-1].text, "EVAL-A")
        self.assertEqual(second.events[0].text, "Report EVAL-B")
        self.assertEqual(second.events[-1].text, "EVAL-B")

    async def test_eval_cancellation_finalizes_only_live_inspection_state(self) -> None:
        store = LiveInspectionStore()
        coordinator = SubagentInspectionCoordinator(store)
        inspection_id = coordinator.eval_started(
            {
                "id": "ptc-task-cancelled",
                "eval_id": "eval-call",
                "description": "Wait until cancelled",
            }
        )

        coordinator.cancel_running()

        inspection = store.get(inspection_id)
        assert inspection is not None
        self.assertEqual(inspection.status, "CANCELLED")

    async def test_eval_success_and_error_finalize_their_matching_rows(self) -> None:
        store = LiveInspectionStore()
        coordinator = SubagentInspectionCoordinator(store)
        success_id = coordinator.eval_started(
            {"id": "ptc-success", "description": "succeed"}
        )
        error_id = coordinator.eval_started(
            {"id": "ptc-error", "description": "fail"}
        )

        coordinator.eval_finished({"id": "ptc-success", "phase": "complete"})
        coordinator.eval_finished(
            {"id": "ptc-error", "phase": "error", "error": "child failed"}
        )

        success = store.get(success_id)
        error = store.get(error_id)
        assert success is not None and error is not None
        self.assertEqual(success.status, "DONE")
        self.assertEqual(error.status, "ERROR")
        self.assertEqual(error.events[-1], InspectionEvent("error", text="child failed"))

    async def test_quickjs_bridge_forwards_callbacks_only_to_task(self) -> None:
        seen: list[Any] = []

        class TaskTool:
            name = "task"

            async def arun(self, *_args: Any, callbacks: Any = None, **_kwargs: Any) -> str:
                seen.append(callbacks)
                return "done"

        callbacks = object()
        other = SimpleNamespace(name="read_file")
        @dataclass
        class Runtime:
            config: dict[str, Any]
            tools: list[Any]

        runtime = Runtime(config={"callbacks": callbacks}, tools=[TaskTool(), other])

        wrapped = runtime_with_task_callbacks(runtime)
        result = await wrapped.tools[0].arun({})

        self.assertEqual(result, "done")
        self.assertEqual(seen, [callbacks])
        self.assertIs(wrapped.tools[1], other)
        self.assertIsNot(wrapped.tools[0], runtime.tools[0])

    async def test_capture_failure_marks_standalone_lifecycle_terminal(self) -> None:
        class Renderer:
            def __init__(self) -> None:
                self.live_inspections = LiveInspectionStore()
                self.events: list[tuple[str, str]] = []

            def subagent_label(self, _subagent: Any) -> str:
                return "worker [fox]"

            def subagent_started(self, _name: str, task: str, *, inspection_id: str = "") -> None:
                self.live_inspections.start(inspection_id, "worker [fox]", task)
                self.events.append(("started", inspection_id))

            def subagent_finished(
                self,
                name: str,
                result: str,
                **_kwargs: Any,
            ) -> None:
                self.events.append((name, result))

        async def fail() -> Any:
            raise RuntimeError("child failed")

        renderer = Renderer()
        subagent = SimpleNamespace(
            task_input="full failing task",
            trigger_call_id="failed-call",
            path=(),
            output=fail(),
        )

        await consume_subagent(subagent, renderer)

        self.assertEqual(renderer.events[-1], ("worker [fox]", "error: child failed"))
        inspection = renderer.live_inspections.get(renderer.events[0][1])
        assert inspection is not None
        self.assertEqual(inspection.status, "ERROR")
        self.assertEqual(inspection.events[-1].kind, "assistant")
        self.assertEqual(inspection.events[-1].text, "error: child failed")

    async def test_projection_failure_does_not_prevent_standalone_completion(self) -> None:
        class Renderer:
            def __init__(self) -> None:
                self.live_inspections = LiveInspectionStore()
                self.finished = ""

            def subagent_label(self, _subagent: Any) -> str:
                return "worker [owl]"

            def subagent_started(self, _name: str, task: str, *, inspection_id: str = "") -> None:
                self.live_inspections.start(inspection_id, "worker [owl]", task)

            def subagent_finished(self, _name: str, result: str, **_kwargs: Any) -> None:
                self.finished = result

        async def succeed() -> str:
            return "parent-facing result"

        renderer = Renderer()
        subagent = SimpleNamespace(
            task_input="full task",
            trigger_call_id="successful-call",
            path=(),
            messages=FailingAsyncItems(),
            output=succeed(),
        )

        await consume_subagent(subagent, renderer)

        self.assertEqual(renderer.finished, "parent-facing result")
        inspection = renderer.live_inspections.get("subagent:successful-call")
        assert inspection is not None
        self.assertEqual(inspection.status, "DONE")
        self.assertEqual(inspection.events[-1].text, "parent-facing result")


if __name__ == "__main__":
    unittest.main()
