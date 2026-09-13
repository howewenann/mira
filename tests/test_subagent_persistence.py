"""Focused tests for durable retrospective subagent inspection."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
import uuid
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from core.execution.inspection.live import InspectionEvent, LiveInspectionStore
from core.execution.inspection.persistence import PersistentSubagentRuns
from core.execution.streams.subagents import consume_subagent
from session.context import SESSION_FIELDS, normalize_messages, normalize_session, with_resume_context
from session.store import SessionStore
from session.subagent_runs import normalize_runs, upsert_run


class SavingStore:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.saved: list[dict[str, Any]] = []

    def save(self, record: dict[str, Any]) -> None:
        if self.fail:
            raise OSError("disk unavailable")
        self.saved.append(deepcopy(record))


class InspectionRenderer:
    def __init__(self, inspections: LiveInspectionStore | None = None) -> None:
        if inspections is not None:
            self.live_inspections = inspections
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def __getattr__(self, name: str) -> Any:
        def callback(*args: Any, **kwargs: Any) -> str:
            self.calls.append((name, args, kwargs))
            return f"forwarded:{name}"

        return callback


def tool_call_event(event_id: int, name: str, call_id: str, description: str) -> dict[str, Any]:
    return {
        "id": event_id,
        "type": "tool_call",
        "name": name,
        "args": {"description": description},
        "call_id": call_id,
        "created_at": "2026-01-01T00:00:00+00:00",
    }


class PersistentSubagentRunsTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_standalone_consumer_projects_through_observer(self) -> None:
        async def output() -> str:
            return "native child result"

        inspections = LiveInspectionStore()
        task = "native child task"
        record = {"events": [tool_call_event(12, "task", "native-call", task)], "runs": []}
        observer = PersistentSubagentRuns(InspectionRenderer(inspections), record, SavingStore())
        subagent = SimpleNamespace(
            name="general-purpose",
            task_input=task,
            trigger_call_id="native-call",
            path=("tools:native-call", "general-purpose:child"),
            output=output(),
        )

        await consume_subagent(subagent, observer)
        observer.close()

        self.assertEqual(len(record["runs"]), 1)
        self.assertEqual(record["runs"][0]["origin_event_id"], 12)
        self.assertEqual(record["runs"][0]["status"], "DONE")
        self.assertEqual(record["runs"][0]["output"], "native child result")

    async def test_standalone_snapshots_exact_identity_transcript_and_output(self) -> None:
        task = "Inspect every relevant file without shortening this request."
        final = "x" * 600
        inspections = LiveInspectionStore()
        inspections.start("inspection-1", "researcher [violet-fox]", task)
        renderer = InspectionRenderer(inspections)
        record = {"events": [tool_call_event(7, "task", "task-call", task)], "runs": []}
        store = SavingStore()
        observer = PersistentSubagentRuns(renderer, record, store)

        returned = observer.subagent_started(
            "researcher [violet-fox]",
            task,
            row_id="task-call",
            inspection_id="inspection-1",
        )
        self.assertEqual(returned, "forwarded:subagent_started")
        created = record["runs"][0]
        durable_id = created["id"]
        self.assertEqual(str(uuid.UUID(durable_id)), durable_id)
        self.assertEqual(created["origin_event_id"], 7)
        self.assertEqual(created["origin_tool"], "task")
        self.assertEqual(created["origin_call_id"], "task-call")
        self.assertEqual(created["row_id"], "task-call")
        self.assertEqual(created["display_name"], "researcher [violet-fox]")
        self.assertEqual(created["task"], task)

        inspections.append_delta("inspection-1", "reasoning", "Checking")
        saves_before_delta_flush = len(store.saved)
        self.assertTrue(observer._dirty)
        inspections.append(
            "inspection-1",
            InspectionEvent("tool_call", name="read_file", args={"path": Path("README.md")}, call_id="read-1"),
        )
        self.assertGreater(len(store.saved), saves_before_delta_flush)
        inspections.upsert_tool_completion(
            "inspection-1",
            InspectionEvent("tool_result", text="contents", name="read_file", call_id="read-1"),
        )
        inspections.finish("inspection-1", status="DONE", final_response=final)
        observer.subagent_finished(
            "researcher [violet-fox]",
            final,
            row_id="task-call",
            inspection_id="inspection-1",
            duration_ms=1234,
        )
        observer.close()

        run = record["runs"][0]
        self.assertEqual(run["id"], durable_id)
        self.assertEqual(run["status"], "DONE")
        self.assertEqual(run["duration_ms"], 1234)
        self.assertEqual(run["output"], final)
        self.assertEqual(
            [event["kind"] for event in run["events"]],
            ["user", "reasoning", "tool_call", "tool_result", "assistant"],
        )
        self.assertEqual(run["events"][2]["args"], {"path": "README.md"})
        self.assertIn("finished_at", run)

    async def test_text_deltas_are_bounded_and_terminal_flushes(self) -> None:
        inspections = LiveInspectionStore()
        inspections.start("inspection-1", "worker [blue-heron]", "task")
        record = {"events": [tool_call_event(1, "task", "call-1", "task")], "runs": []}
        store = SavingStore()
        observer = PersistentSubagentRuns(InspectionRenderer(inspections), record, store)
        observer.subagent_started("worker", "task", row_id="call-1", inspection_id="inspection-1")
        initial_saves = len(store.saved)

        inspections.append_delta("inspection-1", "assistant", "one")
        inspections.append_delta("inspection-1", "assistant", " two")
        self.assertEqual(len(store.saved), initial_saves)
        await asyncio.sleep(0.3)
        self.assertEqual(len(store.saved), initial_saves + 1)

        inspections.finish("inspection-1", status="DONE", final_response="one two three")
        self.assertEqual(store.saved[-1]["runs"][0]["output"], "one two three")
        observer.close()

    async def test_eval_rows_share_only_their_eval_origin(self) -> None:
        inspections = LiveInspectionStore()
        record = {"events": [tool_call_event(14, "eval", "eval-14", "")], "runs": []}
        observer = PersistentSubagentRuns(InspectionRenderer(inspections), record, SavingStore())
        for index, name in enumerate(("critic [amber-owl]", "tester [silver-lynx]"), start=1):
            inspection_id = f"eval-inspection-{index}"
            row_id = f"eval-row-{index}"
            inspections.start(inspection_id, name, f"eval task {index}")
            observer.eval_subagent_started(
                name,
                f"eval task {index}",
                eval_id="eval-14",
                row_id=row_id,
                inspection_id=inspection_id,
            )
            inspections.finish(inspection_id, status="DONE", final_response=f"result {index}")
            observer.eval_subagent_finished(name, eval_id="eval-14", row_id=row_id, duration_ms=index * 10)
        observer.close()

        self.assertEqual(len(record["runs"]), 2)
        self.assertEqual({run["origin_event_id"] for run in record["runs"]}, {14})
        self.assertEqual({run["origin_tool"] for run in record["runs"]}, {"eval"})
        self.assertEqual({run["eval_id"] for run in record["runs"]}, {"eval-14"})

    async def test_idless_inverted_delivery_uses_exact_request_fifo(self) -> None:
        inspections = LiveInspectionStore()
        record: dict[str, Any] = {"events": [], "runs": []}
        observer = PersistentSubagentRuns(InspectionRenderer(inspections), record, SavingStore())
        for index in (1, 2):
            inspections.start(f"inspection-{index}", f"worker [{index}]", "same request")
            observer.subagent_started(
                f"worker [{index}]",
                "same request",
                row_id="",
                inspection_id=f"inspection-{index}",
            )
            inspections.finish(f"inspection-{index}", status="DONE", final_response=f"done {index}")
            observer.subagent_finished(
                f"worker [{index}]",
                f"done {index}",
                inspection_id=f"inspection-{index}",
            )
        self.assertEqual({run["status"] for run in record["runs"]}, {"RUNNING"})
        record["events"].extend(
            [
                tool_call_event(21, "task", "", "same request"),
                tool_call_event(22, "task", "", "same request"),
            ]
        )
        observer.tool_call("task", {"description": "same request"})
        observer.close()

        ownership = {run["inspection_id"]: run["origin_event_id"] for run in record["runs"]}
        self.assertEqual(ownership, {"inspection-1": 21, "inspection-2": 22})
        self.assertEqual({run["status"] for run in record["runs"]}, {"DONE"})

    async def test_hitl_reattach_keeps_one_durable_run(self) -> None:
        inspections = LiveInspectionStore()
        inspections.start("inspection-1", "worker [cool-name]", "original task")
        record = {"events": [tool_call_event(4, "task", "call-4", "original task")], "runs": []}
        store = SavingStore()
        first = PersistentSubagentRuns(InspectionRenderer(inspections), record, store)
        first.subagent_started("worker [cool-name]", "original task", row_id="call-4", inspection_id="inspection-1")
        inspections.append_delta("inspection-1", "reasoning", "before approval")
        first.close()
        durable_id = record["runs"][0]["id"]

        second = PersistentSubagentRuns(InspectionRenderer(inspections), record, store)
        second.subagent_started("worker [cool-name]", "original task", row_id="call-4", inspection_id="inspection-1")
        inspections.append_delta("inspection-1", "reasoning", " after approval")
        inspections.finish("inspection-1", status="DONE", final_response="complete")
        second.subagent_finished("worker", "complete", row_id="call-4", inspection_id="inspection-1")
        second.close()

        self.assertEqual(len(record["runs"]), 1)
        self.assertEqual(record["runs"][0]["id"], durable_id)
        self.assertEqual(record["runs"][0]["display_name"], "worker [cool-name]")
        self.assertEqual(record["runs"][0]["events"][1]["text"], "before approval after approval")

    async def test_runtime_cancellation_and_error_are_persisted_terminally(self) -> None:
        inspections = LiveInspectionStore()
        record = {
            "events": [
                tool_call_event(1, "task", "cancel-call", "cancel task"),
                tool_call_event(2, "eval", "eval-call", "eval task"),
            ],
            "runs": [],
        }
        observer = PersistentSubagentRuns(InspectionRenderer(inspections), record, SavingStore())
        inspections.start("cancel-inspection", "worker [red-kite]", "cancel task")
        observer.subagent_started(
            "worker [red-kite]",
            "cancel task",
            row_id="cancel-call",
            inspection_id="cancel-inspection",
        )
        inspections.finish("cancel-inspection", status="CANCELLED", error="approval rejected")
        observer.subagent_cancelled(
            "worker [red-kite]",
            "approval rejected",
            row_id="cancel-call",
            duration_ms=50,
        )

        inspections.start("error-inspection", "critic [gray-wolf]", "eval task")
        observer.eval_subagent_started(
            "critic [gray-wolf]",
            "eval task",
            eval_id="eval-call",
            row_id="eval-row",
            inspection_id="error-inspection",
        )
        inspections.finish("error-inspection", status="ERROR", error="child failed")
        observer.eval_subagent_cancelled(
            "critic [gray-wolf]",
            "child failed",
            eval_id="eval-call",
            row_id="eval-row",
            duration_ms=75,
        )
        observer.close()

        self.assertEqual(
            {run["row_id"]: run["status"] for run in record["runs"]},
            {"cancel-call": "CANCELLED", "eval-row": "ERROR"},
        )
        self.assertEqual(
            {run["row_id"]: run["duration_ms"] for run in record["runs"]},
            {"cancel-call": 50, "eval-row": 75},
        )

    async def test_save_failure_and_absent_live_store_are_transparent(self) -> None:
        inspections = LiveInspectionStore()
        inspections.start("inspection-1", "worker", "task")
        failing_renderer = InspectionRenderer(inspections)
        observer = PersistentSubagentRuns(failing_renderer, {"events": [], "runs": []}, SavingStore(fail=True))
        self.assertEqual(
            observer.subagent_started("worker", "task", inspection_id="inspection-1"),
            "forwarded:subagent_started",
        )
        inspections.append("inspection-1", InspectionEvent("tool_call", name="read_file", args={}))
        observer.close()

        plain_renderer = InspectionRenderer()
        plain = PersistentSubagentRuns(plain_renderer, {"events": []}, SavingStore())
        self.assertEqual(plain.tool_call("task", {"description": "task"}, call_id="call"), "forwarded:tool_call")
        self.assertNotIn("runs", plain.record)


class SubagentRunCompatibilityTests(unittest.TestCase):
    def test_old_and_invalid_sessions_default_to_empty_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            record = store.new("old", Path("workspace"))
            record.pop("runs")
            normalized = normalize_session(record)
            record["runs"] = {"not": "a list"}
            invalid = normalize_session(record)

        self.assertNotIn("runs", SESSION_FIELDS)
        self.assertEqual(normalized["runs"], [])
        self.assertEqual(invalid["runs"], [])

    def test_runs_never_enter_messages_or_resume_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            record = SessionStore(Path(directory)).new("thread", Path("workspace"))
        record["runs"] = [
            {
                "id": "run-1",
                "task": "secret child task",
                "events": [{"kind": "assistant", "text": "secret child response"}],
            }
        ]
        self.assertEqual(normalize_messages(record["events"]), [])
        self.assertNotIn("secret child", with_resume_context(record, "next request"))

    def test_real_load_reconciles_stale_run_without_advancing_recency(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            record = store.new("thread", Path("workspace"))
            original_updated_at = "2026-01-01T00:00:10+00:00"
            record["updated_at"] = original_updated_at
            record["events"] = [tool_call_event(8, "task", "call-8", "task")]
            record["runs"] = [
                {
                    "id": "run-8",
                    "inspection_id": "inspection-8",
                    "origin_event_id": 8,
                    "origin_tool": "task",
                    "origin_call_id": "call-8",
                    "row_id": "call-8",
                    "eval_id": "",
                    "inspection_type": "subagent",
                    "display_name": "worker [quiet-ibis]",
                    "task": "task",
                    "status": "RUNNING",
                    "started_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:05+00:00",
                    "duration_ms": None,
                    "output": "",
                    "events": [
                        {"kind": "user", "text": "task"},
                        {"kind": "tool_call", "name": "read_file", "args": {}, "call_id": "nested"},
                    ],
                }
            ]
            store.save_metadata(record)
            loaded = store.load("thread", resume=False, workspace=Path("workspace"))
            reread = store.read(store.path("thread"))

        run = loaded["runs"][0]
        self.assertEqual(loaded["updated_at"], original_updated_at)
        self.assertEqual(run["status"], "CANCELLED")
        self.assertEqual(run["duration_ms"], 5000)
        self.assertEqual(run["finished_at"], "2026-01-01T00:00:05+00:00")
        self.assertEqual(run["events"][-1]["kind"], "tool_error")
        self.assertEqual(loaded["events"][0]["status"], "interrupted")
        self.assertEqual(reread, loaded)

    def test_upsert_preserves_identity_when_resume_uses_new_inspection(self) -> None:
        record = {"runs": []}
        first = {
            "id": "durable",
            "inspection_id": "live-old",
            "origin_event_id": 3,
            "row_id": "row",
            "display_name": "worker [stable-name]",
            "started_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:01+00:00",
            "events": [],
        }
        upsert_run(record, first)
        second = {
            **first,
            "id": "replacement",
            "inspection_id": "live-new",
            "display_name": "worker [changed]",
            "updated_at": "2026-01-01T00:00:02+00:00",
        }
        run = upsert_run(record, second)
        self.assertEqual(len(normalize_runs(record["runs"])), 1)
        self.assertEqual(run["id"], "durable")
        self.assertEqual(run["inspection_id"], "live-old")
        self.assertEqual(run["display_name"], "worker [stable-name]")


if __name__ == "__main__":
    unittest.main()
