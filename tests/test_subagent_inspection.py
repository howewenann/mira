"""Focused tests for process-local live subagent inspection."""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from langchain_core.messages import ToolMessage
from langchain_quickjs import CodeInterpreterMiddleware

from agent.middleware import builder as middleware_builder
from agent.middleware.code_interpreter import (
    EVAL_SUBAGENT_ROW_METADATA,
    InspectableCodeInterpreterMiddleware,
    runtime_with_task_callbacks,
)
from core.execution.inspection.live import InspectionEvent, LiveInspectionStore
from core.execution.inspection.subagents import (
    EvalInspectionTransformer,
    SubagentInspectionCapture,
    SubagentInspectionCoordinator,
    capture_child_streams,
    live_inspection_store,
)
from core.execution.streams.subagents import (
    consume_subagent,
    consume_subagents,
)
from core.interface import FrontendEmitter
from ui.shared.adapter import RendererAdapter


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

    def test_fully_streamed_final_response_is_not_duplicated(self) -> None:
        store = LiveInspectionStore()
        inspection_id = store.allocate_id("fully-streamed")
        store.start(inspection_id, "worker [fox]", "inspect")
        store.append_delta(inspection_id, "assistant", "The answer is Candidate A.")

        store.finish(
            inspection_id,
            status="DONE",
            final_response="The answer is Candidate A.",
        )

        inspection = store.get(inspection_id)
        assert inspection is not None
        assistant = [event for event in inspection.events if event.kind == "assistant"]
        self.assertEqual(assistant, [InspectionEvent("assistant", text="The answer is Candidate A.")])

    def test_streamed_prefix_receives_only_final_suffix(self) -> None:
        store = LiveInspectionStore()
        inspection_id = store.allocate_id("streamed-prefix")
        store.start(inspection_id, "worker [fox]", "inspect")
        store.append_delta(inspection_id, "assistant", "The answer")
        received: list[str] = []

        def listener(_inspection_id: str, update: Any) -> None:
            if update.operation == "delta" and update.event is not None:
                received.append(update.event.text)

        store.subscribe(inspection_id, listener)
        store.finish(
            inspection_id,
            status="DONE",
            final_response="The answer is Candidate A.",
        )

        inspection = store.get(inspection_id)
        assert inspection is not None
        assistant = [event for event in inspection.events if event.kind == "assistant"]
        self.assertEqual(received, [" is Candidate A."])
        self.assertEqual(assistant, [InspectionEvent("assistant", text="The answer is Candidate A.")])

    def test_no_stream_appends_exact_final_response_once(self) -> None:
        store = LiveInspectionStore()
        inspection_id = store.allocate_id("no-stream")
        store.start(inspection_id, "worker [fox]", "inspect")

        store.finish(
            inspection_id,
            status="DONE",
            final_response="The complete response.",
        )

        inspection = store.get(inspection_id)
        assert inspection is not None
        assistant = [event for event in inspection.events if event.kind == "assistant"]
        self.assertEqual(assistant, [InspectionEvent("assistant", text="The complete response.")])

    def test_inspector_header_divider_and_spacing_styles_are_explicit(self) -> None:
        styles = Path("ui/textual/styles/mira.tcss").read_text(encoding="utf-8")

        self.assertIn(
            "#inspector {\n"
            "    display: none;\n"
            "    width: 1fr;\n"
            "    height: 1fr;\n"
            "    padding: 0;",
            styles,
        )
        self.assertIn(
            "#inspector-header {\n"
            "    width: 1fr;\n"
            "    height: 2;\n"
            "    padding: 0 1;\n"
            "    margin-bottom: 1;\n"
            "    border-bottom: solid #B7A4E8;",
            styles,
        )
        self.assertIn(
            "#inspector-log {\n"
            "    width: 1fr;\n"
            "    height: 1fr;\n"
            "    padding: 0 1;",
            styles,
        )

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
    def test_eval_middleware_restores_callbacks_without_replacing_quickjs(self) -> None:
        self.assertIs(
            middleware_builder.CodeInterpreterMiddleware,
            InspectableCodeInterpreterMiddleware,
        )
        self.assertTrue(issubclass(InspectableCodeInterpreterMiddleware, CodeInterpreterMiddleware))

    async def test_quickjs_bridge_forwards_callbacks_only_to_task(self) -> None:
        seen: list[tuple[Any, Any]] = []

        class TaskTool:
            name = "task"

            async def arun(
                self,
                *_args: Any,
                callbacks: Any = None,
                config: Any = None,
                **_kwargs: Any,
            ) -> str:
                seen.append((callbacks, config))
                return "done"

        @dataclass
        class Runtime:
            config: dict[str, Any]
            tools: list[Any]

        callbacks = object()
        other = SimpleNamespace(name="read_file")
        runtime = Runtime(config={"callbacks": callbacks}, tools=[TaskTool(), other])

        wrapped = runtime_with_task_callbacks(runtime)
        result = await wrapped.tools[0].arun(
            {},
            config={"metadata": {"existing": True}},
            tool_call_id="ptc_task_exact",
        )

        self.assertEqual(result, "done")
        self.assertEqual(seen[0][0], callbacks)
        self.assertEqual(
            seen[0][1]["metadata"],
            {"existing": True, EVAL_SUBAGENT_ROW_METADATA: "ptc_task_exact"},
        )
        self.assertIs(wrapped.tools[1], other)
        self.assertIsNot(wrapped.tools[0], runtime.tools[0])

    async def test_quickjs_bridge_reuses_logical_rows_across_eval_replay(self) -> None:
        observed: list[tuple[str, str, str]] = []

        class TaskTool:
            name = "task"

            async def arun(
                self,
                payload: dict[str, Any],
                *,
                config: dict[str, Any],
                tool_call_id: str,
                **_kwargs: Any,
            ) -> str:
                observed.append(
                    (
                        tool_call_id,
                        payload["runtime"].tool_call_id,
                        config["metadata"][EVAL_SUBAGENT_ROW_METADATA],
                    )
                )
                return "done"

        @dataclass
        class ChildRuntime:
            tool_call_id: str

        @dataclass
        class Runtime:
            config: dict[str, Any]
            tools: list[Any]
            stream_writer: Any

        async def replay(raw_ids: list[str]) -> list[str]:
            events: list[dict[str, Any]] = []
            runtime = Runtime(
                config={"callbacks": object()},
                tools=[TaskTool()],
                stream_writer=events.append,
            )
            wrapped = runtime_with_task_callbacks(runtime)
            for raw_id in raw_ids:
                wrapped.stream_writer(
                    {
                        "type": "subagent",
                        "phase": "start",
                        "id": raw_id,
                        "eval_id": "eval-call",
                        "subagent_type": "general-purpose",
                        "description": "same request",
                        "label": "candidate",
                    }
                )
                await wrapped.tools[0].arun(
                    {"runtime": ChildRuntime(raw_id)},
                    config={"metadata": {}},
                    tool_call_id=raw_id,
                )
                wrapped.stream_writer(
                    {
                        "type": "subagent",
                        "phase": "complete",
                        "id": raw_id,
                        "eval_id": "eval-call",
                    }
                )
            return [str(event["id"]) for event in events]

        first = await replay(["ptc_task_raw_a", "ptc_task_raw_b"])
        second = await replay(["ptc_task_raw_c", "ptc_task_raw_d"])

        self.assertEqual(first, second)
        self.assertEqual(len(set(first[::2])), 2)
        self.assertEqual(first[0], first[1])
        self.assertEqual(first[2], first[3])
        self.assertTrue(all(len(set(values)) == 1 for values in observed))
        self.assertEqual([values[0] for values in observed[:2]], first[::2])
        self.assertEqual([values[0] for values in observed[2:]], second[::2])

    async def test_direct_child_assistant_output_streams_incrementally(self) -> None:
        store = LiveInspectionStore()
        inspection_id = store.allocate_id("direct-stream")
        store.start(inspection_id, "worker [fox]", "inspect")
        received: list[str] = []

        def listener(_inspection_id: str, update: Any) -> None:
            if update.event is not None and update.event.kind == "assistant":
                received.append(update.event.text)

        store.subscribe(inspection_id, listener)
        capture = SubagentInspectionCapture(store, inspection_id)
        await capture_child_streams(
            SimpleNamespace(messages=AsyncItems([StreamedMessage([], ["The", " answer"])])),
            capture,
        )

        self.assertEqual(received, ["The", " answer"])

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

    async def test_standalone_child_keeps_native_lifecycle(self) -> None:
        class Renderer:
            manages_subagent_animation = True

            def __init__(self) -> None:
                self.live_inspections = LiveInspectionStore()
                self.events: list[tuple[str, ...]] = []

            def start_subagent_live(self) -> None:
                self.events.append(("live_started",))

            def stop_subagent_live(self) -> None:
                self.events.append(("live_stopped",))

            def subagent_label(self, _subagent: Any) -> str:
                return "general-purpose [fox]"

            def subagent_started(
                self,
                name: str,
                task: str,
                *,
                inspection_id: str = "",
                **_kwargs: Any,
            ) -> None:
                self.live_inspections.start(inspection_id, name, task)
                self.events.append(("subagent_started", name, task, inspection_id))

            def subagent_finished(
                self,
                name: str,
                result: str,
                **_kwargs: Any,
            ) -> None:
                self.events.append(("subagent_finished", name, result))

        async def result() -> str:
            return "parent-facing result"

        renderer = Renderer()
        subagent = SimpleNamespace(
            name="general-purpose",
            task_input="shared task text",
            trigger_call_id="direct-call",
            path=("tools:direct-call", "general-purpose:child"),
            messages=AsyncItems([StreamedMessage([], ["parent-facing result"])]),
            output=result(),
        )

        await consume_subagents(AsyncItems([subagent]), renderer)

        self.assertEqual(renderer.events[0], ("live_started",))
        self.assertEqual(renderer.events[-1], ("live_stopped",))
        started = next(event for event in renderer.events if event[0] == "subagent_started")
        self.assertEqual(started[1:3], ("general-purpose [fox]", "shared task text"))
        self.assertEqual(
            next(event for event in renderer.events if event[0] == "subagent_finished")[1:],
            ("general-purpose [fox]", "parent-facing result"),
        )
        inspection = renderer.live_inspections.get(started[3])
        assert inspection is not None
        self.assertEqual(inspection.status, "DONE")
        self.assertEqual(inspection.events[-1].text, "parent-facing result")

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
        coordinator.register_eval_namespace(
            ("general-purpose:child-a",), "ptc-task-a"
        )
        coordinator.register_eval_namespace(
            ("general-purpose:child-b",), "ptc-task-b"
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

    async def test_eval_protocol_snapshot_growth_emits_only_text_deltas(self) -> None:
        store = LiveInspectionStore()
        coordinator = SubagentInspectionCoordinator(store)
        inspection_id = coordinator.eval_started(
            {"id": "ptc-growing", "description": "Report the answer"}
        )
        coordinator.register_eval_namespace(
            ("general-purpose:growing",), "ptc-growing"
        )
        received: list[str] = []

        def listener(_inspection_id: str, update: Any) -> None:
            if update.event is not None and update.event.kind == "assistant":
                received.append(update.event.text)

        store.subscribe(inspection_id, listener)
        for text in ("The", "The answer", "The answer is Candidate A."):
            coordinator.handle_protocol_event(
                {
                    "method": "values",
                    "params": {
                        "namespace": ["general-purpose:growing"],
                        "data": {
                            "messages": [
                                {"type": "human", "content": "Report the answer"},
                                {"type": "ai", "content": text},
                            ]
                        },
                    },
                }
            )

        inspection = store.get(inspection_id)
        assert inspection is not None
        assistant = [event for event in inspection.events if event.kind == "assistant"]
        self.assertEqual(received, ["The", " answer", " is Candidate A."])
        self.assertEqual(assistant, [InspectionEvent("assistant", text="The answer is Candidate A.")])

    async def test_eval_raw_protocol_streams_reasoning_and_text_without_snapshot_duplicates(
        self,
    ) -> None:
        store = LiveInspectionStore()
        coordinator = SubagentInspectionCoordinator(store)
        inspection_id = coordinator.eval_started(
            {
                "id": "ptc_task_streamed",
                "description": "Stream the answer",
                "subagent_type": "general-purpose",
            }
        )
        namespace = ("general-purpose:ptc_task_streamed", "model:run-one")
        for delta in (
            {"type": "reasoning-delta", "reasoning": "Think "},
            {"type": "reasoning-delta", "reasoning": "carefully."},
            {"type": "text-delta", "text": "Final "},
            {"type": "text-delta", "text": "answer."},
        ):
            coordinator.handle_protocol_event(
                {
                    "method": "messages",
                    "params": {
                        "namespace": namespace,
                        "data": [{"event": "content-block-delta", "delta": delta}, {}],
                    },
                }
            )
        coordinator.handle_protocol_event(
            {
                "method": "values",
                "params": {
                    "namespace": namespace[:1],
                    "data": {
                        "messages": [
                            {"type": "human", "content": "Stream the answer"},
                            {
                                "type": "ai",
                                "content": [
                                    {"type": "reasoning", "reasoning": "Think carefully."},
                                    {"type": "text", "text": "Final answer."},
                                ],
                            },
                        ]
                    },
                },
            }
        )

        inspection = store.get(inspection_id)
        assert inspection is not None
        self.assertEqual(
            [(event.kind, event.text) for event in inspection.events],
            [
                ("user", "Stream the answer"),
                ("reasoning", "Think carefully."),
                ("assistant", "Final answer."),
            ],
        )

    async def test_eval_native_handle_is_inspection_only_by_exact_trigger_id(self) -> None:
        class Renderer:
            manages_subagent_animation = True

            def __init__(self) -> None:
                self.live_inspections = LiveInspectionStore()
                self.events: list[str] = []

            def start_subagent_live(self) -> None:
                self.events.append("start")

            def stop_subagent_live(self) -> None:
                self.events.append("stop")

            def subagent_label(self, _subagent: Any) -> str:
                raise AssertionError("Eval handle must not enter normal lifecycle")

        async def result() -> str:
            return "Eval final response"

        renderer = Renderer()
        coordinator = SubagentInspectionCoordinator(renderer.live_inspections)
        inspection_id = coordinator.eval_started(
            {
                "id": "ptc_task_exact",
                "description": "same task text",
                "subagent_type": "general-purpose",
            }
        )
        transformer = EvalInspectionTransformer((), coordinator)
        self.assertTrue(
            transformer.process(
                {
                    "method": "tasks",
                    "params": {
                        "namespace": ["general-purpose:native-task-uuid"],
                        "data": {
                            "metadata": {
                                EVAL_SUBAGENT_ROW_METADATA: "ptc_task_exact"
                            }
                        },
                    },
                }
            )
        )
        subagent = SimpleNamespace(
            trigger_call_id="native-task-uuid",
            task_input="same task text",
            path=("general-purpose:native-task-uuid",),
            messages=AsyncItems([]),
            tool_calls=AsyncItems([]),
            output=result(),
        )
        ordinary = SimpleNamespace(trigger_call_id="ordinary-task-uuid")

        self.assertFalse(coordinator.is_eval_child(ordinary))

        await consume_subagents(
            AsyncItems([subagent]),
            renderer,
            inspection=coordinator,
        )

        self.assertEqual(renderer.events, [])
        inspection = renderer.live_inspections.get(inspection_id)
        assert inspection is not None
        self.assertEqual(inspection.status, "DONE")
        self.assertEqual(inspection.events[-1].text, "Eval final response")

    async def test_eval_protocol_replaces_truncated_hint_with_full_task(self) -> None:
        store = LiveInspectionStore()
        coordinator = SubagentInspectionCoordinator(store)
        full_task = "Inspect this complete Eval request: " + "long detail " * 30
        self.assertGreater(len(full_task), 200)
        inspection_id = coordinator.eval_started(
            {"id": "ptc-long", "description": full_task[:200]}
        )
        coordinator.register_eval_namespace(
            ("general-purpose:long",), "ptc-long"
        )

        coordinator.handle_protocol_event(
            {
                "method": "values",
                "params": {
                    "namespace": ["general-purpose:long"],
                    "data": {
                        "messages": [
                            {"type": "human", "content": full_task},
                            {"type": "ai", "content": "Done"},
                        ]
                    },
                },
            }
        )

        inspection = store.get(inspection_id)
        assert inspection is not None
        user_events = [event for event in inspection.events if event.kind == "user"]
        self.assertEqual(user_events, [InspectionEvent("user", text=full_task)])
        self.assertGreater(len(user_events[0].text), 200)
        self.assertNotIn("…", user_events[0].text)

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
