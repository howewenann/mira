"""Focused runtime tests for the process-local Workflow UI projection."""

from __future__ import annotations

import unittest
from typing import Any

from langgraph.stream.transformers import TasksTransformer

from core.execution.workflows import WorkflowCoordinator
from core.interface import FrontendEmitter, FrontendEvent, WorkflowEvent
from ui.shared.adapter import RendererAdapter
from ui.textual.workflow_demo import build_workflow_demo


class RecordingFrontend:
    def __init__(self) -> None:
        self.events: list[FrontendEvent] = []

    def emit(self, event: FrontendEvent) -> None:
        self.events.append(event)

    async def request(self, _request: Any) -> Any:
        raise AssertionError("Workflow projection must not request frontend input")


class AsyncItems:
    def __init__(self, items: list[Any]) -> None:
        self.items = items

    async def __aiter__(self):
        for item in self.items:
            yield item


class WorkflowCoordinatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_projection_groups_same_node_fanout_and_loop_demo(self) -> None:
        frontend = RecordingFrontend()
        coordinator = WorkflowCoordinator(
            FrontendEmitter(frontend),
            workflow_id="demo",
        )
        graph = build_workflow_demo(delay_scale=0)

        coordinator.start()
        run = await graph.astream_events(
            {"events": []},
            version="v3",
            transformers=[TasksTransformer],
        )
        async with run:
            await coordinator.consume(run.extensions["tasks"])
            output = await run.output()
        coordinator.finish()

        events = [event for event in frontend.events if isinstance(event, WorkflowEvent)]
        starts = [event for event in events if event.phase == "task_start"]
        finishes = [event for event in events if event.phase == "task_finish"]
        self.assertEqual(
            [(event.name, event.step) for event in starts],
            [
                ("prepare", 1),
                ("worker", 2),
                ("worker", 2),
                ("worker", 2),
                ("review", 3),
                ("review", 4),
            ],
        )
        self.assertEqual(len({event.task_id for event in starts}), 6)
        self.assertEqual(
            len({event.task_id for event in starts if event.name == "worker"}),
            3,
        )
        self.assertEqual(
            len({event.task_id for event in starts if event.name == "review"}),
            2,
        )
        self.assertEqual([event.status for event in finishes], ["DONE"] * 6)
        self.assertEqual(events[0].phase, "run_start")
        self.assertEqual(events[-1].phase, "run_finish")
        self.assertEqual(output["review_count"], 2)
        self.assertEqual(output["events"][-2:], ["review 1 complete", "review 2 complete"])

    async def test_repeated_names_use_task_ids_and_errors_are_terminal(self) -> None:
        frontend = RecordingFrontend()
        coordinator = WorkflowCoordinator(FrontendEmitter(frontend))
        await coordinator.consume(
            AsyncItems(
                [
                    {"id": "a", "name": "worker", "input": {}},
                    {"id": "b", "name": "worker", "input": {}},
                    {"id": "a", "name": "worker", "error": None},
                    {"id": "b", "name": "worker", "error": "boom"},
                    {"id": "c", "name": "worker", "input": {}},
                    {"id": "c", "name": "worker", "error": None},
                ]
            )
        )
        coordinator.finish()

        events = [event for event in frontend.events if isinstance(event, WorkflowEvent)]
        starts = [event for event in events if event.phase == "task_start"]
        finishes = [event for event in events if event.phase == "task_finish"]
        self.assertEqual(
            [(event.task_id, event.step) for event in starts],
            [("a", 1), ("b", 1), ("c", 2)],
        )
        self.assertEqual(
            [(event.task_id, event.status) for event in finishes],
            [("a", "DONE"), ("b", "ERROR"), ("c", "DONE")],
        )

    async def test_repeated_task_id_reuses_original_row_and_step(self) -> None:
        frontend = RecordingFrontend()
        coordinator = WorkflowCoordinator(FrontendEmitter(frontend))
        await coordinator.consume(
            AsyncItems(
                [
                    {"id": "resume", "name": "review", "input": {}},
                    {"id": "resume", "name": "review", "error": None},
                    {"id": "resume", "name": "review", "input": {}},
                    {"id": "resume", "name": "review", "error": None},
                ]
            )
        )

        starts = [
            event
            for event in frontend.events
            if isinstance(event, WorkflowEvent) and event.phase == "task_start"
        ]
        self.assertEqual(
            [(event.task_id, event.step) for event in starts],
            [("resume", 1), ("resume", 1)],
        )
        self.assertEqual(len(coordinator.tasks), 1)

    async def test_cancel_emits_one_terminal_run_event(self) -> None:
        frontend = RecordingFrontend()
        coordinator = WorkflowCoordinator(FrontendEmitter(frontend), workflow_id="cancel-me")
        await coordinator.consume(AsyncItems([{"id": "a", "name": "slow", "input": {}}]))

        coordinator.cancel()
        coordinator.cancel()

        cancelled = [
            event
            for event in frontend.events
            if isinstance(event, WorkflowEvent) and event.phase == "run_cancel"
        ]
        self.assertEqual(len(cancelled), 1)
        self.assertEqual(cancelled[0].status, "CANCELLED")


class WorkflowAdapterTests(unittest.TestCase):
    def test_workflow_event_projects_to_renderer_callbacks(self) -> None:
        class Renderer:
            def __init__(self) -> None:
                self.calls: list[tuple[Any, ...]] = []

            def workflow_task_finished(
                self,
                task_id: str,
                name: str,
                step: int,
                *,
                status: str,
                error: str,
            ) -> None:
                self.calls.append((task_id, name, step, status, error))

        renderer = Renderer()
        RendererAdapter(renderer).emit(
            WorkflowEvent(
                phase="task_finish",
                task_id="task-1",
                name="review",
                step=2,
                status="ERROR",
                error="failed",
            )
        )

        self.assertEqual(renderer.calls, [("task-1", "review", 2, "ERROR", "failed")])


if __name__ == "__main__":
    unittest.main()
