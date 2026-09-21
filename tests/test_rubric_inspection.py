"""Focused tests for process-local Rubric phase inspection."""

from __future__ import annotations

import unittest
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from core.execution.inspection.live import LiveInspectionStore
from core.execution.inspection.rubric import (
    inspection_event_values,
    rubric_inspection_id,
)
from core.execution.streams.rubric import RubricEventRenderer


class RecordingRenderer:
    def __init__(self) -> None:
        self.live_inspections = LiveInspectionStore()
        self.lifecycle: list[dict[str, Any]] = []
        self.created_before_lifecycle: list[bool] = []

    def rubric_lifecycle_event(self, event: dict[str, Any]) -> None:
        inspection_id = str(event.get("inspection_id") or "")
        if event["type"] in {"rubric_verification_start", "rubric_grading_start"}:
            self.created_before_lifecycle.append(
                self.live_inspections.get(inspection_id) is not None
            )
        self.lifecycle.append(dict(event))


def start_event(run_id: str, iteration: int, phase: str, text: str) -> dict[str, Any]:
    inspection_id = rubric_inspection_id(run_id, iteration, phase)
    return {
        "type": f"rubric_{'verification' if phase == 'verifier' else 'grading'}_start",
        "grading_run_id": run_id,
        "iteration": iteration,
        "inspection_id": inspection_id,
        "inspection_events": [{"kind": "user", "text": text}],
    }


class RubricInspectionTests(unittest.TestCase):
    def test_phase_is_created_before_lifecycle_and_keeps_exact_input(self) -> None:
        renderer = RecordingRenderer()
        rubric = RubricEventRenderer(renderer, 2)

        rubric.handle(start_event("run-1", 0, "verifier", "exact verifier input"))

        inspection = renderer.live_inspections.get("rubric:run-1:0:verifier")
        assert inspection is not None
        self.assertEqual(renderer.created_before_lifecycle, [True])
        self.assertEqual(inspection.inspection_type, "Rubrics")
        self.assertEqual(inspection.events[0].text, "exact verifier input")
        self.assertNotIn("inspection_events", renderer.lifecycle[0])

    def test_live_deltas_and_authoritative_tools_share_one_transcript(self) -> None:
        renderer = RecordingRenderer()
        rubric = RubricEventRenderer(renderer, 1)
        identity = {"grading_run_id": "run-tools", "iteration": 0}
        rubric.handle(start_event("run-tools", 0, "verifier", "inspect"))
        rubric.handle(
            {
                "type": "rubric_inspection_delta",
                **identity,
                "phase": "verifier",
                "kind": "reasoning",
                "text": "checking ",
            }
        )
        rubric.handle(
            {
                "type": "rubric_inspection_delta",
                **identity,
                "phase": "verifier",
                "kind": "reasoning",
                "text": "now",
            }
        )
        for chunk in (
            {"type": "tool_call_chunk", "index": 0, "name": "read_file", "args": '{"file_path":"/fo'},
            {"type": "tool_call_chunk", "index": 0, "id": "read-1", "args": 'o"}'},
        ):
            rubric.handle({"type": "rubric_tool_call_delta", **identity, "chunk": chunk})
        rubric.handle(
            {
                "type": "rubric_tool_start",
                **identity,
                "tool_call_id": "read-1",
                "tool_name": "read_file",
                "tool_args": {"file_path": "/foo"},
            }
        )
        rubric.handle(
            {
                "type": "rubric_tool_end",
                **identity,
                "tool_call_id": "read-1",
                "tool_name": "read_file",
                "output": "contents",
                "is_error": False,
            }
        )
        rubric.handle(
            {
                "type": "rubric_verification_end",
                **identity,
                "succeeded": True,
                "final_response": "VERIFICATION_COMPLETE",
            }
        )

        inspection = renderer.live_inspections.get("rubric:run-tools:0:verifier")
        assert inspection is not None
        self.assertEqual(inspection.status, "DONE")
        self.assertEqual(
            [(event.kind, event.call_id) for event in inspection.events],
            [
                ("user", ""),
                ("reasoning", ""),
                ("tool_call", "read-1"),
                ("tool_result", "read-1"),
                ("assistant", ""),
            ],
        )
        self.assertEqual(inspection.events[1].text, "checking now")
        self.assertEqual(inspection.events[2].args, {"file_path": "/foo"})

    def test_running_restart_is_hitl_continuation_and_terminal_restart_is_retry(self) -> None:
        renderer = RecordingRenderer()
        rubric = RubricEventRenderer(renderer, 1)
        start = start_event("run-retry", 0, "verifier", "request")

        rubric.handle(start)
        rubric.handle(start)
        inspection = renderer.live_inspections.get("rubric:run-retry:0:verifier")
        assert inspection is not None
        self.assertEqual([event.text for event in inspection.events], ["request"])

        rubric.handle(
            {
                "type": "rubric_verification_end",
                "grading_run_id": "run-retry",
                "iteration": 0,
                "succeeded": False,
                "error": "coverage retry",
            }
        )
        rubric.handle(start)
        self.assertEqual(inspection.status, "RUNNING")
        self.assertEqual(
            [event.text for event in inspection.events if event.kind == "user"],
            ["request", "request"],
        )

    def test_failure_and_cancellation_close_pending_tools(self) -> None:
        for cancelled, expected in ((False, "ERROR"), (True, "CANCELLED")):
            with self.subTest(cancelled=cancelled):
                renderer = RecordingRenderer()
                rubric = RubricEventRenderer(renderer, 1)
                identity = {"grading_run_id": f"run-{cancelled}", "iteration": 0}
                rubric.handle(start_event(identity["grading_run_id"], 0, "verifier", "inspect"))
                rubric.handle(
                    {
                        "type": "rubric_tool_start",
                        **identity,
                        "tool_call_id": "tool-1",
                        "tool_name": "execute",
                        "tool_args": {"command": "check"},
                    }
                )
                rubric.handle(
                    {
                        "type": "rubric_verification_end",
                        **identity,
                        "succeeded": False,
                        "cancelled": cancelled,
                        "error": "" if cancelled else "boom",
                    }
                )
                inspection = renderer.live_inspections.get(
                    f"rubric:{identity['grading_run_id']}:0:verifier"
                )
                assert inspection is not None
                self.assertEqual(inspection.status, expected)
                tool_error = next(
                    item for item in inspection.events if item.kind == "tool_error"
                )
                self.assertIn("cancelled" if cancelled else "boom", tool_error.text)

    def test_grader_input_fidelity_and_iterations_are_isolated(self) -> None:
        messages = [
            HumanMessage(content="grade this exact evidence"),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "read_file", "args": {"file_path": "/a"}, "id": "read-a"}
                ],
            ),
            ToolMessage(content="A", name="read_file", tool_call_id="read-a"),
        ]
        events = inspection_event_values(messages)
        renderer = RecordingRenderer()
        rubric = RubricEventRenderer(renderer, 2)
        for iteration in (0, 1):
            event = start_event("run-isolated", iteration, "grader", "unused")
            event["inspection_events"] = events
            rubric.handle(event)

        first = renderer.live_inspections.get("rubric:run-isolated:0:grader")
        second = renderer.live_inspections.get("rubric:run-isolated:1:grader")
        assert first is not None and second is not None
        self.assertIsNot(first, second)
        self.assertEqual(
            [(item.kind, item.text, item.call_id) for item in first.events],
            [
                ("user", "grade this exact evidence", ""),
                ("tool_call", "", "read-a"),
                ("tool_result", "A", "read-a"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
