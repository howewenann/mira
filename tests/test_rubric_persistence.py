"""Focused coverage for durable Rubric inspection transcripts."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from typing import Any

from core.execution.inspection.live import LiveInspectionStore
from core.execution.inspection.persistence import PersistentSubagentRuns
from core.execution.inspection.rubric import rubric_inspection_id
from core.execution.streams.rubric import RubricEventRenderer
from session.context import normalize_messages, with_resume_context
from session.store import SessionStore
from session.subagent_runs import (
    run_count,
    run_for_inspection_id,
    runs_for_origin,
)


class SavingStore:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.saved: list[dict[str, Any]] = []

    def save(self, record: dict[str, Any]) -> None:
        if self.fail:
            raise OSError("disk unavailable")
        self.saved.append(copy.deepcopy(record))


class InspectionRenderer:
    def __init__(self, inspections: LiveInspectionStore) -> None:
        self.live_inspections = inspections
        self.lifecycle: list[dict[str, Any]] = []

    def rubric_lifecycle_event(self, event: dict[str, Any]) -> str:
        self.lifecycle.append(dict(event))
        return "forwarded:rubric_lifecycle_event"

    def subagent_started(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def subagent_finished(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def phase_start(
    run_id: str,
    iteration: int,
    phase: str,
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "type": f"rubric_{'verification' if phase == 'verifier' else 'grading'}_start",
        "grading_run_id": run_id,
        "iteration": iteration,
        "inspection_id": rubric_inspection_id(run_id, iteration, phase),
        "inspection_events": events,
    }


def phase_end(
    run_id: str,
    iteration: int,
    phase: str,
    *,
    succeeded: bool = True,
    cancelled: bool = False,
    error: str = "",
    final_response: str | None = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": f"rubric_{'verification' if phase == 'verifier' else 'grading'}_end",
        "grading_run_id": run_id,
        "iteration": iteration,
        "inspection_id": rubric_inspection_id(run_id, iteration, phase),
        "succeeded": succeeded,
    }
    if cancelled:
        event["cancelled"] = True
    if error:
        event["error"] = error
    if final_response is not None:
        event["final_response"] = final_response
    return event


class RubricPersistenceTests(unittest.TestCase):
    def observer(self, *, fail: bool = False) -> tuple[
        PersistentSubagentRuns,
        RubricEventRenderer,
        dict[str, Any],
        LiveInspectionStore,
    ]:
        inspections = LiveInspectionStore()
        record: dict[str, Any] = {"events": [], "runs": []}
        observer = PersistentSubagentRuns(
            InspectionRenderer(inspections),
            record,
            SavingStore(fail=fail),
        )
        return observer, RubricEventRenderer(observer, 3), record, inspections

    def test_verifier_persists_exact_ordered_live_transcript(self) -> None:
        observer, rubric, record, _inspections = self.observer()
        identity = {"grading_run_id": "grade-verifier", "iteration": 0}
        rubric.handle(phase_start("grade-verifier", 0, "verifier", [{"kind": "user", "text": "full verifier input"}]))
        rubric.handle({"type": "rubric_inspection_delta", **identity, "phase": "verifier", "kind": "reasoning", "text": "inspect evidence"})
        rubric.handle({"type": "rubric_tool_start", **identity, "tool_call_id": "read-1", "tool_name": "read_file", "tool_args": {"path": "one.txt"}})
        rubric.handle({"type": "rubric_tool_end", **identity, "tool_call_id": "read-1", "tool_name": "read_file", "output": "one", "is_error": False})
        rubric.handle({"type": "rubric_tool_start", **identity, "tool_call_id": "read-2", "tool_name": "read_file", "tool_args": {"path": "two.txt"}})
        rubric.handle({"type": "rubric_tool_end", **identity, "tool_call_id": "read-2", "tool_name": "read_file", "output": "denied", "is_error": True})
        rubric.handle(phase_end("grade-verifier", 0, "verifier", final_response="verified evidence"))
        observer.close()

        run = run_for_inspection_id(record["runs"], "rubric:grade-verifier:0:verifier")
        assert run is not None
        self.assertEqual(run["status"], "DONE")
        self.assertIsNone(run["origin_event_id"])
        self.assertEqual(run["inspection_type"], "Rubrics")
        self.assertEqual(run["display_name"], "Verifier · Pass 1")
        self.assertEqual(
            run["events"],
            [
                {"kind": "user", "text": "full verifier input"},
                {"kind": "reasoning", "text": "inspect evidence"},
                {"kind": "tool_call", "name": "read_file", "args": {"path": "one.txt"}, "call_id": "read-1"},
                {"kind": "tool_result", "text": "one", "name": "read_file", "call_id": "read-1"},
                {"kind": "tool_call", "name": "read_file", "args": {"path": "two.txt"}, "call_id": "read-2"},
                {"kind": "tool_error", "text": "denied", "name": "read_file", "call_id": "read-2"},
                {"kind": "assistant", "text": "verified evidence"},
            ],
        )

    def test_grader_persists_input_evidence_reasoning_and_raw_json(self) -> None:
        observer, rubric, record, _inspections = self.observer()
        identity = {"grading_run_id": "grade-grader", "iteration": 0}
        rubric.handle(
            phase_start(
                "grade-grader",
                0,
                "grader",
                [
                    {"kind": "user", "text": "exact grader input"},
                    {"kind": "tool_result", "text": "verifier evidence", "name": "read_file", "call_id": "evidence-1"},
                ],
            )
        )
        rubric.handle({"type": "rubric_inspection_delta", **identity, "phase": "grader", "kind": "reasoning", "text": "weigh evidence"})
        rubric.handle({"type": "rubric_inspection_delta", **identity, "phase": "grader", "kind": "assistant", "text": '{"result":"satisfied"}'})
        rubric.handle(phase_end("grade-grader", 0, "grader"))
        observer.close()

        run = run_for_inspection_id(record["runs"], "rubric:grade-grader:0:grader")
        assert run is not None
        self.assertEqual(
            run["events"],
            [
                {"kind": "user", "text": "exact grader input"},
                {"kind": "tool_result", "text": "verifier evidence", "name": "read_file", "call_id": "evidence-1"},
                {"kind": "reasoning", "text": "weigh evidence"},
                {"kind": "assistant", "text": '{"result":"satisfied"}'},
            ],
        )
        self.assertEqual(run["output"], '{"result":"satisfied"}')

    def test_two_passes_create_four_origin_free_runs_without_ui_ownership(self) -> None:
        observer, rubric, record, _inspections = self.observer()
        expected: set[str] = set()
        for iteration in (0, 1):
            for phase in ("verifier", "grader"):
                inspection_id = rubric_inspection_id("grade-many", iteration, phase)
                expected.add(inspection_id)
                rubric.handle(phase_start("grade-many", iteration, phase, [{"kind": "user", "text": f"{phase} {iteration}"}]))
                rubric.handle(phase_end("grade-many", iteration, phase, final_response="done" if phase == "verifier" else None))
        observer.close()

        self.assertEqual({run["inspection_id"] for run in record["runs"]}, expected)
        self.assertTrue(all(run["origin_event_id"] is None for run in record["runs"]))
        self.assertEqual(run_count(record["runs"], 1), 0)
        self.assertEqual(runs_for_origin(record["runs"], 1), [])

    def test_subagents_still_require_an_origin_before_terminal_snapshot(self) -> None:
        inspections = LiveInspectionStore()
        inspections.start("subagent:no-origin", "worker", "task")
        record: dict[str, Any] = {"events": [], "runs": []}
        observer = PersistentSubagentRuns(
            InspectionRenderer(inspections),
            record,
            SavingStore(),
        )
        observer.subagent_started("worker", "task", inspection_id="subagent:no-origin")
        inspections.finish("subagent:no-origin", status="DONE", final_response="done")
        observer.subagent_finished("worker", "done", inspection_id="subagent:no-origin")
        observer.close()

        self.assertEqual(len(record["runs"]), 1)
        self.assertEqual(record["runs"][0]["status"], "RUNNING")

    def test_error_cancel_and_hitl_reattach_keep_authoritative_identity(self) -> None:
        observer, first, record, inspections = self.observer()
        first.handle(phase_start("grade-resume", 0, "verifier", [{"kind": "user", "text": "request"}]))
        first.handle({"type": "rubric_inspection_delta", "grading_run_id": "grade-resume", "iteration": 0, "phase": "verifier", "kind": "reasoning", "text": "before approval"})
        observer.close()
        durable_id = record["runs"][0]["id"]

        second_observer = PersistentSubagentRuns(
            InspectionRenderer(inspections),
            record,
            SavingStore(),
        )
        second = RubricEventRenderer(second_observer, 3)
        second.handle(phase_start("grade-resume", 0, "verifier", [{"kind": "user", "text": "request"}]))
        second.handle({"type": "rubric_inspection_delta", "grading_run_id": "grade-resume", "iteration": 0, "phase": "verifier", "kind": "reasoning", "text": " after approval"})
        second.handle(phase_end("grade-resume", 0, "verifier", final_response="complete"))

        for iteration, cancelled in ((1, False), (2, True)):
            second.handle(phase_start("grade-resume", iteration, "grader", [{"kind": "user", "text": "grade"}]))
            second.handle(
                phase_end(
                    "grade-resume",
                    iteration,
                    "grader",
                    succeeded=False,
                    cancelled=cancelled,
                    error="grader failed" if not cancelled else "",
                )
            )
        second_observer.close()

        resumed = run_for_inspection_id(record["runs"], "rubric:grade-resume:0:verifier")
        assert resumed is not None
        self.assertEqual(resumed["id"], durable_id)
        self.assertEqual(resumed["status"], "DONE")
        self.assertEqual([event["text"] for event in resumed["events"] if event["kind"] == "reasoning"], ["before approval after approval"])
        self.assertEqual(
            {
                run["inspection_id"]: run["status"]
                for run in record["runs"]
            },
            {
                "rubric:grade-resume:0:verifier": "DONE",
                "rubric:grade-resume:1:grader": "ERROR",
                "rubric:grade-resume:2:grader": "CANCELLED",
            },
        )

    def test_save_failure_is_observational(self) -> None:
        observer, rubric, record, inspections = self.observer(fail=True)
        rubric.handle(phase_start("grade-failure", 0, "grader", [{"kind": "user", "text": "input"}]))
        rubric.handle(phase_end("grade-failure", 0, "grader"))
        observer.close()

        inspection = inspections.get("rubric:grade-failure:0:grader")
        assert inspection is not None
        self.assertEqual(inspection.status, "DONE")
        self.assertEqual(len(record["runs"]), 1)


class RubricPersistenceCompatibilityTests(unittest.TestCase):
    def test_origin_free_stale_run_reconciles_without_touching_root_or_recency(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            record = store.new("rubric-stale", Path("workspace"))
            original_updated_at = "2026-01-01T00:00:10+00:00"
            record["updated_at"] = original_updated_at
            record["events"] = [
                {
                    "id": 1,
                    "type": "tool_call",
                    "name": "task",
                    "args": {"description": "unrelated"},
                    "call_id": "unrelated",
                    "created_at": "2026-01-01T00:00:00+00:00",
                }
            ]
            record["runs"] = [
                {
                    "id": "rubric-run",
                    "inspection_id": "rubric:grade-stale:0:verifier",
                    "origin_event_id": None,
                    "inspection_type": "Rubrics",
                    "display_name": "Verifier · Pass 1",
                    "status": "RUNNING",
                    "started_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:05+00:00",
                    "events": [
                        {"kind": "user", "text": "verify"},
                        {"kind": "tool_call", "name": "read_file", "args": {}, "call_id": "nested"},
                    ],
                }
            ]
            store.save_metadata(record)
            loaded = store.load("rubric-stale", resume=False, workspace=Path("workspace"))
            reread = store.read(store.path("rubric-stale"))

        run = loaded["runs"][0]
        self.assertEqual(loaded["updated_at"], original_updated_at)
        self.assertEqual(run["status"], "CANCELLED")
        self.assertEqual(run["duration_ms"], 5000)
        self.assertEqual(run["events"][-1]["kind"], "tool_error")
        self.assertNotIn("status", loaded["events"][0])
        self.assertEqual(reread, loaded)

    def test_rubric_runs_remain_outside_model_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            record = SessionStore(Path(directory)).new("thread", Path("workspace"))
        record["runs"] = [
            {
                "id": "rubric-run",
                "inspection_id": "rubric:grade:0:grader",
                "inspection_type": "Rubrics",
                "events": [{"kind": "assistant", "text": "private raw grader JSON"}],
            }
        ]
        record["resume_context_pending"] = True

        self.assertEqual(normalize_messages(record["events"]), [])
        self.assertNotIn("private raw grader JSON", with_resume_context(record, "next"))


if __name__ == "__main__":
    unittest.main()
