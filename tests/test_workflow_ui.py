"""Focused native Workflow runtime projection tests."""

from __future__ import annotations

import asyncio
import operator
import unittest
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt

from core.execution.inspection.live import LiveInspectionStore
from core.execution.inspection.subagents import SubagentInspectionCoordinator
from core.execution.workflows import WorkflowCoordinator, execute_workflow
from core.interface import FrontendEmitter, FrontendEvent, WorkflowEvent
from core.interface.requests import AskUserRequest
from ui.shared.adapter import RendererAdapter


class RecordingFrontend:
    def __init__(self) -> None:
        self.events: list[FrontendEvent] = []

    def emit(self, event: FrontendEvent) -> None:
        self.events.append(event)

    async def request(self, request: Any) -> Any:
        if isinstance(request, AskUserRequest):
            return "approved"
        raise AssertionError(f"unexpected request: {request!r}")


async def run_workflow(
    graph: Any,
    payload: Any,
) -> tuple[Any, list[WorkflowEvent]]:
    frontend = RecordingFrontend()
    output = await execute_workflow(
        graph,
        payload,
        emitter=FrontendEmitter(frontend),
        inspection=SubagentInspectionCoordinator(LiveInspectionStore()),
        workflow_id="workflow:test",
        workflow_name="test",
    )
    return output, [
        event for event in frontend.events if isinstance(event, WorkflowEvent)
    ]


class SequentialState(TypedDict, total=False):
    seed: str
    first: str
    second: str


def sequential_graph() -> Any:
    graph = StateGraph(SequentialState)
    graph.add_node("first", lambda state: {"first": f"first({state['seed']})"})
    graph.add_node("second", lambda state: {"second": f"second({state['first']})"})
    graph.add_edge(START, "first")
    graph.add_edge("first", "second")
    graph.add_edge("second", END)
    return graph.compile()


class ParallelState(TypedDict, total=False):
    seed: str
    left: str
    right: str


def parallel_graph() -> Any:
    graph = StateGraph(ParallelState)
    graph.add_node("left", lambda state: {"left": state["seed"]})
    graph.add_node("right", lambda state: {"right": state["seed"]})
    graph.add_edge(START, "left")
    graph.add_edge(START, "right")
    graph.add_edge("left", END)
    graph.add_edge("right", END)
    return graph.compile()


class FanoutState(TypedDict):
    items: list[str]
    results: Annotated[list[str], operator.add]


def fanout_graph() -> Any:
    graph = StateGraph(FanoutState)
    graph.add_node("worker", lambda state: {"results": [state["item"]]})
    graph.add_conditional_edges(
        START,
        lambda state: [Send("worker", {"item": item}) for item in state["items"]],
        ["worker"],
    )
    graph.add_edge("worker", END)
    return graph.compile()


class HitlState(TypedDict):
    request: str
    answer: str


def hitl_graph() -> Any:
    def approval(state: HitlState) -> dict[str, str]:
        answer = interrupt(
            {
                "type": "ask_user",
                "question": f"Approve {state['request']}?",
                "options": ["approved"],
            }
        )
        return {"answer": str(answer)}

    graph = StateGraph(HitlState)
    graph.add_node("approval", approval)
    graph.add_edge(START, "approval")
    graph.add_edge("approval", END)
    return graph.compile(checkpointer=InMemorySaver())


class WorkflowCoordinatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_sequential_tasks_capture_exact_input_and_result(self) -> None:
        output, events = await run_workflow(sequential_graph(), {"seed": "alpha"})
        starts = [event for event in events if event.phase == "task_start"]
        finishes = [event for event in events if event.phase == "task_finish"]

        self.assertEqual([event.step for event in starts], [1, 2])
        self.assertEqual(starts[0].input_state, {"seed": "alpha"})
        self.assertEqual(
            starts[1].input_state,
            {"seed": "alpha", "first": "first(alpha)"},
        )
        self.assertEqual(finishes[0].result, {"first": "first(alpha)"})
        self.assertEqual(finishes[1].result, {"second": "second(first(alpha))"})
        self.assertEqual(output["second"], "second(first(alpha))")
        self.assertTrue(all(event.result_available for event in finishes))

    async def test_parallel_tasks_are_siblings_with_independent_results(self) -> None:
        _output, events = await run_workflow(parallel_graph(), {"seed": "alpha"})
        starts = [event for event in events if event.phase == "task_start"]
        finishes = [event for event in events if event.phase == "task_finish"]

        self.assertEqual({event.step for event in starts}, {1})
        self.assertEqual(len({event.task_id for event in starts}), 2)
        self.assertEqual(
            {event.name: event.result for event in finishes},
            {"left": {"left": "alpha"}, "right": {"right": "alpha"}},
        )

    async def test_send_fanout_keeps_duplicate_names_as_distinct_tasks(self) -> None:
        _output, events = await run_workflow(
            fanout_graph(),
            {"items": ["a", "b", "c"], "results": []},
        )
        starts = [event for event in events if event.phase == "task_start"]
        finishes = [event for event in events if event.phase == "task_finish"]

        self.assertEqual([event.name for event in starts], ["worker"] * 3)
        self.assertEqual(len({event.task_id for event in starts}), 3)
        self.assertEqual(
            [event.input_state for event in starts],
            [{"item": "a"}, {"item": "b"}, {"item": "c"}],
        )
        self.assertEqual(
            [event.result for event in finishes],
            [
                {"results": ["a"]},
                {"results": ["b"]},
                {"results": ["c"]},
            ],
        )

    async def test_hitl_reuses_task_and_drops_interrupted_empty_result(self) -> None:
        output, events = await run_workflow(
            hitl_graph(),
            {"request": "deploy", "answer": ""},
        )
        task_events = [event for event in events if event.task_id]

        self.assertEqual(
            [event.phase for event in task_events],
            ["task_start", "task_waiting", "task_resume", "task_finish"],
        )
        self.assertEqual(len({event.task_id for event in task_events}), 1)
        self.assertFalse(task_events[1].result_available)
        self.assertEqual(task_events[-1].result, {"answer": "approved"})
        self.assertEqual(output["answer"], "approved")

    async def test_error_and_cancellation_do_not_invent_results(self) -> None:
        graph = StateGraph(dict)

        def fail(_state: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("boom")

        graph.add_node("fail", fail)
        graph.add_edge(START, "fail")
        graph.add_edge("fail", END)
        frontend = RecordingFrontend()
        with self.assertRaisesRegex(RuntimeError, "boom"):
            await execute_workflow(
                graph.compile(),
                {"seed": "alpha"},
                emitter=FrontendEmitter(frontend),
                inspection=SubagentInspectionCoordinator(LiveInspectionStore()),
                workflow_id="failure",
                workflow_name="failure",
            )
        finish = next(
            event
            for event in frontend.events
            if isinstance(event, WorkflowEvent) and event.phase == "task_finish"
        )
        self.assertEqual(finish.status, "ERROR")
        self.assertFalse(finish.result_available)

        entered = asyncio.Event()

        async def slow(_state: dict[str, Any]) -> dict[str, Any]:
            entered.set()
            await asyncio.Event().wait()
            return {"done": True}

        cancel_graph = StateGraph(dict)
        cancel_graph.add_node("slow", slow)
        cancel_graph.add_edge(START, "slow")
        cancel_graph.add_edge("slow", END)
        cancelled_frontend = RecordingFrontend()
        running = asyncio.create_task(
            execute_workflow(
                cancel_graph.compile(),
                {"seed": "alpha"},
                emitter=FrontendEmitter(cancelled_frontend),
                inspection=SubagentInspectionCoordinator(LiveInspectionStore()),
                workflow_id="cancel",
                workflow_name="cancel",
            )
        )
        await entered.wait()
        running.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await running
        cancel = next(
            event
            for event in cancelled_frontend.events
            if isinstance(event, WorkflowEvent) and event.phase == "run_cancel"
        )
        self.assertEqual(cancel.status, "CANCELLED")

    async def test_input_snapshot_survives_in_place_python_mutation(self) -> None:
        graph = StateGraph(dict)

        def mutate(state: dict[str, Any]) -> dict[str, Any]:
            state["mutated_only"] = True
            return {"returned": True}

        graph.add_node("mutate", mutate)
        graph.add_edge(START, "mutate")
        graph.add_edge("mutate", END)
        _output, events = await run_workflow(graph.compile(), {"original": True})
        start = next(event for event in events if event.phase == "task_start")
        self.assertEqual(start.input_state, {"original": True})

    async def test_legitimate_none_result_remains_available(self) -> None:
        async def native_events():
            yield {"id": "task-none", "name": "nothing", "input": None}
            yield {
                "id": "task-none",
                "name": "nothing",
                "result": {"__root__": None},
                "error": None,
                "interrupts": (),
            }

        frontend = RecordingFrontend()
        coordinator = WorkflowCoordinator(
            FrontendEmitter(frontend),
            workflow_id="none",
            workflow_name="none",
        )
        await coordinator.consume(native_events())
        finish = next(
            event
            for event in frontend.events
            if isinstance(event, WorkflowEvent) and event.phase == "task_finish"
        )
        self.assertTrue(finish.result_available)
        self.assertIsNone(finish.result)


class WorkflowAdapterTests(unittest.TestCase):
    def test_all_tree_payloads_project_to_renderer_callbacks(self) -> None:
        class Renderer:
            def __init__(self) -> None:
                self.calls: list[tuple[Any, ...]] = []

            def workflow_started(self, workflow_id: str, name: str) -> None:
                self.calls.append(("run", workflow_id, name))

            def workflow_task_started(
                self,
                task_id: str,
                name: str,
                step: int,
                input_state: Any,
                **kwargs: Any,
            ) -> None:
                self.calls.append(("task", task_id, name, step, input_state, kwargs))

            def workflow_agent_started(
                self,
                task_id: str,
                name: str,
                inspection_id: str,
                **kwargs: Any,
            ) -> None:
                self.calls.append(("agent", task_id, name, inspection_id, kwargs))

        renderer = Renderer()
        adapter = RendererAdapter(renderer)
        adapter.emit(
            WorkflowEvent(
                phase="run_start",
                workflow_id="wf-1",
                workflow_name="decision_brief",
            )
        )
        adapter.emit(
            WorkflowEvent(
                phase="task_start",
                workflow_id="wf-1",
                task_id="task-1",
                name="analyse",
                step=2,
                input_state=None,
            )
        )
        adapter.emit(
            WorkflowEvent(
                phase="agent_start",
                workflow_id="wf-1",
                task_id="task-1",
                agent_id="agent-1",
                inspection_id="agent-1",
                name="researcher",
                task_input="research",
            )
        )

        self.assertEqual(renderer.calls[0], ("run", "wf-1", "decision_brief"))
        self.assertEqual(renderer.calls[1][1:5], ("task-1", "analyse", 2, None))
        self.assertEqual(renderer.calls[1][5]["workflow_id"], "wf-1")
        self.assertEqual(renderer.calls[2][1:4], ("task-1", "researcher", "agent-1"))
        self.assertEqual(renderer.calls[2][4]["task_input"], "research")


if __name__ == "__main__":
    unittest.main()
