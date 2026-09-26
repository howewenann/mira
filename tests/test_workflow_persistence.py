"""Durable Workflow transcript projection tests."""

from __future__ import annotations

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from core.execution.inspection.live import LiveInspectionStore
from core.execution.inspection.persistence import PersistentSubagentRuns
from core.interface import FrontendEmitter
from session.context import normalize_messages, normalize_session
from session.store import SessionStore
from session.workflows import (
    PersistentWorkflowHistory,
    freeze_workflow_value,
    reconcile_stale_workflows,
)
from ui.shared.adapter import RendererAdapter


class SavingStore:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.saved: list[dict[str, Any]] = []

    def save(self, record: dict[str, Any]) -> None:
        if self.fail:
            raise OSError("disk unavailable")
        self.saved.append(deepcopy(record))


class Renderer:
    def __init__(self, inspections: LiveInspectionStore | None = None) -> None:
        if inspections is not None:
            self.live_inspections = inspections
        self.events: list[Any] = []

    def __getattr__(self, name: str) -> Any:
        if not name.startswith("workflow_"):
            raise AttributeError(name)

        def callback(*args: Any, **kwargs: Any) -> None:
            self.events.append((name, args, kwargs))

        return callback


class BadRepr:
    def __repr__(self) -> str:
        raise RuntimeError("repr exploded")


class WorkflowPersistenceTests(unittest.TestCase):
    def observer(
        self,
        record: dict[str, Any],
        *,
        command: str = "/workflow__demo topic=alpha",
        store: SavingStore | None = None,
    ) -> tuple[FrontendEmitter, Renderer, SavingStore]:
        renderer = Renderer()
        saving = store or SavingStore()
        observer = PersistentWorkflowHistory(
            renderer,
            record,
            saving,
            command=command,
        )
        return FrontendEmitter(RendererAdapter(observer)), renderer, saving

    def test_projection_preserves_identity_order_steps_values_and_agents(self) -> None:
        record: dict[str, Any] = {"events": []}
        observer, renderer, store = self.observer(record)

        observer.workflow_started("wf-1", "demo")
        observer.workflow_task_started(
            "task-a",
            "analyse",
            1,
            input_state={"path": Path("input.txt")},
            workflow_id="wf-1",
        )
        observer.workflow_task_started(
            "task-b",
            "analyse",
            1,
            input_state=SimpleNamespace(values=(1, 2), tags={"a", "b"}),
            workflow_id="wf-1",
        )
        observer.workflow_agent_started(
            "task-b",
            "researcher",
            "inspection-b",
            task_input="research",
            workflow_id="wf-1",
        )
        observer.workflow_agent_finished(
            "task-b",
            "researcher",
            "inspection-b",
            status="DONE",
            result="complete",
            workflow_id="wf-1",
        )
        observer.workflow_task_finished(
            "task-a",
            "analyse",
            1,
            status="DONE",
            result=None,
            result_available=True,
            workflow_id="wf-1",
        )
        observer.workflow_task_finished(
            "task-b",
            "analyse",
            1,
            status="DONE",
            result={"answer": 2},
            result_available=True,
            workflow_id="wf-1",
        )
        observer.workflow_finished(
            "wf-1",
            final_state={"done": True},
            final_state_available=True,
        )

        self.assertTrue(store.saved)
        event = record["events"][0]
        self.assertEqual(event["type"], "workflow")
        self.assertEqual(event["workflow_id"], "wf-1")
        self.assertEqual(event["workflow_name"], "demo")
        self.assertEqual(event["command"], "/workflow__demo topic=alpha")
        self.assertEqual(event["status"], "DONE")
        self.assertEqual([task["task_id"] for task in event["tasks"]], ["task-a", "task-b"])
        self.assertEqual([task["name"] for task in event["tasks"]], ["analyse", "analyse"])
        self.assertEqual([task["step"] for task in event["tasks"]], [1, 1])
        self.assertTrue(event["tasks"][0]["result_available"])
        self.assertEqual(event["tasks"][0]["result"]["copy_text"], "None")
        self.assertEqual(event["tasks"][1]["agents"][0]["inspection_id"], "inspection-b")
        self.assertNotIn("events", event["tasks"][1]["agents"][0])
        self.assertTrue(event["final_state_available"])
        self.assertIn("'done': True", event["final_state"]["copy_text"])
        self.assertEqual(renderer.events[0][0], "workflow_started")

    def test_non_json_values_survive_real_session_save_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            record = store.new("thread", Path("workspace"))
            observer = FrontendEmitter(
                RendererAdapter(
                    PersistentWorkflowHistory(
                        Renderer(), record, store, command="/workflow__demo topic=values"
                    )
                )
            )
            observer.workflow_started("wf-values", "demo")
            observer.workflow_task_started(
                "task-values",
                "values",
                1,
                input_state={
                    "path": Path("input.txt"),
                    "object": SimpleNamespace(values=(1, 2, 3), tags={"a", "b"}),
                },
                workflow_id="wf-values",
            )
            observer.workflow_task_finished(
                "task-values",
                "values",
                1,
                result=(Path("result.txt"), {1, 2}),
                result_available=True,
                workflow_id="wf-values",
            )
            observer.workflow_finished(
                "wf-values",
                final_state=SimpleNamespace(path=Path("final.txt")),
                final_state_available=True,
            )

            loaded = store.read(store.path("thread"))

        event = loaded["events"][0]
        self.assertIn("input.txt", event["tasks"][0]["input_state"]["copy_text"])
        self.assertIn("result.txt", event["tasks"][0]["result"]["copy_text"])
        self.assertIn("final.txt", event["final_state"]["copy_text"])

    def test_failed_repr_and_save_are_observational(self) -> None:
        frozen = freeze_workflow_value(BadRepr())
        self.assertEqual(frozen["display_text"], "<unrepresentable BadRepr>")
        self.assertEqual(frozen["copy_text"], "<unrepresentable BadRepr>")

        record: dict[str, Any] = {"events": []}
        observer, renderer, _store = self.observer(record, store=SavingStore(fail=True))
        observer.workflow_started("wf-fail", "demo")
        observer.workflow_task_started(
            "task-fail",
            "bad",
            1,
            input_state=BadRepr(),
            workflow_id="wf-fail",
        )
        self.assertEqual(record["events"][0]["workflow_id"], "wf-fail")
        self.assertEqual(renderer.events[-1][0], "workflow_task_started")

    def test_multiple_invocations_remain_independent_and_model_isolated(self) -> None:
        record: dict[str, Any] = {"events": [], "turns": 4, "title": "Existing title"}
        for index in (1, 2):
            observer, _renderer, _store = self.observer(record)
            workflow_id = f"wf-{index}"
            observer.workflow_started(workflow_id, "demo")
            observer.workflow_finished(
                workflow_id,
                final_state={"invocation": index},
                final_state_available=True,
            )

        self.assertEqual([event["workflow_id"] for event in record["events"]], ["wf-1", "wf-2"])
        self.assertEqual(normalize_messages(record["events"]), [])
        self.assertEqual(record["turns"], 4)
        self.assertEqual(record["title"], "Existing title")

    def test_waiting_duration_continues_and_freezes_at_terminal_boundary(self) -> None:
        current = [10.0]
        record: dict[str, Any] = {"events": []}
        observer = FrontendEmitter(
            RendererAdapter(
                PersistentWorkflowHistory(
                    Renderer(),
                    record,
                    SavingStore(),
                    command="/workflow__demo value=1",
                    clock=lambda: current[0],
                    now=lambda: f"2026-01-01T00:00:{int(current[0]):02d}+00:00",
                )
            )
        )
        observer.workflow_started("wf-timing", "demo")
        current[0] = 12.0
        observer.workflow_task_started(
            "task-timing",
            "approval",
            1,
            input_state={"value": 1},
            workflow_id="wf-timing",
        )
        current[0] = 15.0
        observer.workflow_task_waiting(
            "task-timing", "approval", 1, workflow_id="wf-timing"
        )
        current[0] = 20.0
        observer.workflow_task_resumed(
            "task-timing", "approval", 1, workflow_id="wf-timing"
        )
        current[0] = 25.0
        observer.workflow_task_finished(
            "task-timing",
            "approval",
            1,
            result=None,
            result_available=True,
            workflow_id="wf-timing",
        )
        observer.workflow_finished(
            "wf-timing", final_state=None, final_state_available=True
        )

        event = record["events"][0]
        self.assertEqual(event["duration_ms"], 15000)
        self.assertEqual(event["tasks"][0]["duration_ms"], 13000)
        self.assertEqual(event["status"], "DONE")

    def test_normalization_keeps_only_explicit_json_safe_schema(self) -> None:
        record = {
            "id": "thread",
            "title": "Title",
            "workspace": ".",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "turns": 0,
            "dashboard": {},
            "current_plan": None,
            "current_goal": None,
            "events": [
                {
                    "id": 1,
                    "type": "workflow",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:01+00:00",
                    "finished_at": "",
                    "workflow_id": "wf",
                    "workflow_name": "demo",
                    "command": "/workflow__demo x=1",
                    "status": "WAITING",
                    "error": "",
                    "duration_ms": 1000,
                    "tasks": [
                        {
                            "task_id": "a",
                            "name": "analyse",
                            "step": 2,
                            "status": "WAITING",
                            "error": "",
                            "duration_ms": 900,
                            "input_state": {"display_text": "{'x': 1}", "copy_text": "{'x': 1}"},
                            "result_available": False,
                            "result": None,
                            "agents": [],
                            "internal": object(),
                        }
                    ],
                    "final_state_available": False,
                    "final_state": None,
                    "internal": object(),
                }
            ],
        }

        normalized = normalize_session(record)
        event = normalized["events"][0]
        self.assertNotIn("internal", event)
        self.assertNotIn("internal", event["tasks"][0])
        self.assertEqual(event["tasks"][0]["step"], 2)

    def test_stale_reconciliation_interrupts_only_active_rows(self) -> None:
        record: dict[str, Any] = {
            "events": [
                {
                    "type": "workflow",
                    "workflow_id": "wf",
                    "workflow_name": "demo",
                    "command": "/workflow__demo x=1",
                    "status": "WAITING",
                    "updated_at": "2026-01-01T00:00:05+00:00",
                    "finished_at": "",
                    "duration_ms": 5000,
                    "tasks": [
                        {
                            "task_id": "a",
                            "name": "load",
                            "step": 1,
                            "status": "DONE",
                            "duration_ms": 1000,
                            "input_state": freeze_workflow_value({}),
                            "result_available": True,
                            "result": freeze_workflow_value({"loaded": True}),
                            "agents": [],
                        },
                        {
                            "task_id": "b",
                            "name": "approval",
                            "step": 2,
                            "status": "WAITING",
                            "duration_ms": 4000,
                            "input_state": freeze_workflow_value({}),
                            "result_available": False,
                            "result": None,
                            "agents": [
                                {
                                    "inspection_id": "inspection-b",
                                    "name": "reviewer",
                                    "task_input": "review",
                                    "status": "RUNNING",
                                    "duration_ms": 3000,
                                }
                            ],
                        },
                    ],
                    "final_state_available": False,
                    "final_state": None,
                }
            ]
        }

        self.assertTrue(reconcile_stale_workflows(record))
        event = record["events"][0]
        self.assertEqual(event["status"], "INTERRUPTED")
        self.assertEqual(event["finished_at"], event["updated_at"])
        self.assertEqual(event["duration_ms"], 5000)
        self.assertEqual([task["status"] for task in event["tasks"]], ["DONE", "INTERRUPTED"])
        self.assertEqual(event["tasks"][1]["agents"][0]["status"], "INTERRUPTED")

    def test_real_session_load_reconciles_without_advancing_recency(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            record = store.new("thread", Path("workspace"))
            record["updated_at"] = "2026-01-01T00:00:10+00:00"
            observer = FrontendEmitter(
                RendererAdapter(
                    PersistentWorkflowHistory(
                        Renderer(), record, store, command="/workflow__demo value=1"
                    )
                )
            )
            observer.workflow_started("wf-running", "demo")
            observer.workflow_task_started(
                "task-running",
                "approval",
                1,
                input_state={"value": 1},
                workflow_id="wf-running",
            )
            durable_updated_at = record["updated_at"]

            loaded = store.load("thread", resume=False, workspace=Path("workspace"))
            reread = store.read(store.path("thread"))

        event = loaded["events"][0]
        self.assertEqual(loaded["updated_at"], durable_updated_at)
        self.assertEqual(event["status"], "INTERRUPTED")
        self.assertEqual(event["tasks"][0]["status"], "INTERRUPTED")
        self.assertEqual(reread, loaded)

    def test_external_real_save_does_not_detach_active_workflow_event(self) -> None:
        current = [10.0]
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            record = store.new("thread-external-save", Path("workspace"))
            history = PersistentWorkflowHistory(
                Renderer(),
                record,
                store,
                command="/workflow__demo value=1",
                clock=lambda: current[0],
            )
            observer = FrontendEmitter(RendererAdapter(history))
            observer.workflow_started("wf-external-save", "demo")
            current[0] = 11.0
            observer.workflow_task_started(
                "task-external-save",
                "calculate",
                1,
                input_state={"value": 1},
                workflow_id="wf-external-save",
            )

            store.save(record)

            current[0] = 15.0
            observer.workflow_task_finished(
                "task-external-save",
                "calculate",
                1,
                status="DONE",
                result={"answer": 2},
                result_available=True,
                workflow_id="wf-external-save",
            )
            current[0] = 18.0
            observer.workflow_finished(
                "wf-external-save",
                final_state={"complete": True},
                final_state_available=True,
            )
            loaded = store.load(
                "thread-external-save", resume=True, workspace=Path("workspace")
            )

        event = loaded["events"][0]
        task = event["tasks"][0]
        self.assertEqual(event["status"], "DONE")
        self.assertEqual(task["status"], "DONE")
        self.assertEqual(event["duration_ms"], 8000)
        self.assertEqual(task["duration_ms"], 4000)
        self.assertTrue(task["result_available"])
        self.assertIn("'answer': 2", task["result"]["copy_text"])
        self.assertTrue(event["final_state_available"])
        self.assertIn("'complete': True", event["final_state"]["copy_text"])

    def test_combined_agent_persistence_keeps_canonical_terminal_rows(self) -> None:
        current = [20.0]
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            record = store.new("thread-agent-save", Path("workspace"))
            inspections = LiveInspectionStore()
            renderer = Renderer(inspections)
            history = PersistentWorkflowHistory(
                renderer,
                record,
                store,
                command="/workflow__agents topic=history",
                clock=lambda: current[0],
            )
            runs = PersistentSubagentRuns(history, record, store)
            observer = FrontendEmitter(RendererAdapter(runs))

            observer.workflow_started("wf-agent-save", "agents")
            current[0] = 21.0
            observer.workflow_task_started(
                "task-agent",
                "research",
                1,
                input_state={"topic": "history"},
                workflow_id="wf-agent-save",
            )
            inspections.start(
                "inspection-agent-save", "researcher", "research history"
            )
            current[0] = 22.0
            observer.workflow_agent_started(
                "task-agent",
                "researcher",
                "inspection-agent-save",
                task_input="research history",
                workflow_id="wf-agent-save",
            )
            inspections.finish(
                "inspection-agent-save",
                status="DONE",
                final_response="agent result",
            )
            current[0] = 25.0
            observer.workflow_agent_finished(
                "task-agent",
                "researcher",
                "inspection-agent-save",
                status="DONE",
                result="agent result",
                workflow_id="wf-agent-save",
            )
            current[0] = 27.0
            observer.workflow_task_finished(
                "task-agent",
                "research",
                1,
                status="DONE",
                result={"research": "complete"},
                result_available=True,
                workflow_id="wf-agent-save",
            )
            current[0] = 28.0
            observer.workflow_task_started(
                "task-finalize",
                "finalize",
                2,
                input_state={"research": "complete"},
                workflow_id="wf-agent-save",
            )
            current[0] = 29.0
            observer.workflow_task_finished(
                "task-finalize",
                "finalize",
                2,
                status="DONE",
                result="finished",
                result_available=True,
                workflow_id="wf-agent-save",
            )
            current[0] = 30.0
            observer.workflow_finished(
                "wf-agent-save",
                final_state={"done": True},
                final_state_available=True,
            )
            runs.close()
            history.close()
            loaded = store.load(
                "thread-agent-save", resume=True, workspace=Path("workspace")
            )

        event = loaded["events"][0]
        agent_task, final_task = event["tasks"]
        agent = agent_task["agents"][0]
        self.assertEqual(event["status"], "DONE")
        self.assertEqual(event["duration_ms"], 10000)
        self.assertEqual([task["status"] for task in event["tasks"]], ["DONE", "DONE"])
        self.assertEqual(agent["status"], "DONE")
        self.assertEqual(agent["duration_ms"], 3000)
        self.assertEqual(agent_task["duration_ms"], 6000)
        self.assertEqual(final_task["duration_ms"], 1000)
        self.assertEqual(agent_task["result_available"], True)
        self.assertIn("complete", agent_task["result"]["copy_text"])
        self.assertIn("'done': True", event["final_state"]["copy_text"])
        self.assertEqual(len(loaded["runs"]), 1)
        self.assertEqual(loaded["runs"][0]["inspection_id"], "inspection-agent-save")
        self.assertEqual(loaded["runs"][0]["status"], "DONE")


if __name__ == "__main__":
    unittest.main()
