"""Deterministic Workflow inspection and HITL integration coverage."""

from __future__ import annotations

import asyncio
import unittest
from typing import Annotated, Any, TypedDict

from deepagents.middleware.subagents import create_sub_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.stream.transformers import TasksTransformer
from langgraph.types import Command, interrupt
from pydantic import PrivateAttr

from core.execution.inspection.live import LiveInspectionStore
from core.execution.inspection.subagents import SubagentInspectionCoordinator
from core.execution.streams.output import capture_output, collect_interrupts
from core.execution.streams.subagents import consume_workflow_inspections
from core.execution.workflows import WorkflowCoordinator
from core.interface import FrontendEmitter, FrontendEvent, WorkflowEvent


class RecordingFrontend:
    def __init__(self) -> None:
        self.events: list[FrontendEvent] = []

    def emit(self, event: FrontendEvent) -> None:
        self.events.append(event)

    async def request(self, _request: Any) -> Any:
        raise AssertionError("This test resumes the native interrupt directly")


class AsyncItems:
    def __init__(self, items: list[Any]) -> None:
        self.items = items

    async def __aiter__(self):
        for item in self.items:
            yield item


class RecordingInspectionCoordinator(SubagentInspectionCoordinator):
    def __init__(self, store: LiveInspectionStore) -> None:
        super().__init__(store)
        self.starts: list[tuple[str, str, bool]] = []

    def standalone_started(
        self,
        subagent: Any,
        title: str,
        task: str,
        *,
        inspection_type: str = "subagent",
    ) -> tuple[str, str, bool]:
        started = super().standalone_started(
            subagent,
            title,
            task,
            inspection_type=inspection_type,
        )
        self.starts.append(started)
        return started


@tool
async def workflow_phase2_approval(value: str) -> str:
    """Interrupt once, then return the resumed approval value."""
    answer = interrupt(
        {
            "type": "ask_user",
            "question": "Approve the deterministic Workflow test?",
            "options": ["Approve"],
        }
    )
    return f"APPROVED:{value}:{answer!r}"


class WorkflowPhase2Model(BaseChatModel):
    """Call the interrupting tool once, then summarize its result."""

    _tool_names: list[str] = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "workflow-phase2-test"

    def bind_tools(
        self,
        tools: Any,
        *,
        tool_choice: Any | None = None,
        **kwargs: Any,
    ) -> "WorkflowPhase2Model":
        del tool_choice, kwargs
        self._tool_names = [
            str(
                item.get("function", {}).get("name")
                if isinstance(item, dict)
                else getattr(item, "name", "")
            )
            for item in tools
        ]
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        tool_result = next(
            (
                message
                for message in reversed(messages)
                if isinstance(message, ToolMessage)
            ),
            None,
        )
        if tool_result is not None:
            message = AIMessage(content=f"FINAL:{tool_result.content}")
        else:
            if "workflow_phase2_approval" not in self._tool_names:
                raise RuntimeError("The deterministic tool was not bound")
            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "workflow_phase2_approval",
                        "args": {"value": "phase2"},
                        "id": "approval-call",
                        "type": "tool_call",
                    }
                ],
            )
        return ChatResult(generations=[ChatGeneration(message=message)])


class WorkflowState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def build_interrupting_workflow():
    agent = create_sub_agent(
        {
            "name": "workflow-agent",
            "description": "Deterministic Workflow test agent",
            "system_prompt": "Run the deterministic tool.",
            "model": WorkflowPhase2Model(),
            "tools": [workflow_phase2_approval],
        }
    )
    graph = StateGraph(WorkflowState)
    graph.add_node("agent", agent)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    return graph.compile(checkpointer=InMemorySaver())


async def consume_protocol(run: Any, inspection: SubagentInspectionCoordinator) -> None:
    async for event in run:
        inspection.handle_protocol_event(event)


async def run_pass(
    graph: Any,
    payload: Any,
    config: dict[str, Any],
    workflow: WorkflowCoordinator,
    inspection: SubagentInspectionCoordinator,
) -> tuple[list[Any], Any]:
    inspection.begin_pass()
    run = await graph.astream_events(
        payload,
        config=config,
        version="v3",
        transformers=[TasksTransformer],
    )
    tasks = run.extensions.get("tasks")
    if tasks is None:
        raise AssertionError("TasksTransformer did not expose tasks")
    output: dict[str, Any] = {}
    async with run:
        await asyncio.gather(
            consume_protocol(run, inspection),
            workflow.consume(tasks),
            consume_workflow_inspections(
                run.subgraphs,
                inspection,
                workflow.bind_inspection,
            ),
            capture_output(run.output(), output),
        )
    interrupts = await collect_interrupts(run, output.get("value"))
    return interrupts, output.get("value")


class WorkflowInspectionIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_interrupted_child_reuses_task_row_and_inspection_on_resume(self) -> None:
        frontend = RecordingFrontend()
        workflow = WorkflowCoordinator(FrontendEmitter(frontend), workflow_id="test")
        store = LiveInspectionStore()
        inspection = RecordingInspectionCoordinator(store)
        graph = build_interrupting_workflow()
        config = {"configurable": {"thread_id": "workflow-phase2-test"}}
        workflow.start()

        interrupts, _output = await run_pass(
            graph,
            {"messages": [{"role": "user", "content": "Run the tool."}]},
            config,
            workflow,
            inspection,
        )
        self.assertEqual(len(interrupts), 1)
        task = workflow.tasks[0]
        self.assertEqual(task.status, "WAITING")
        first_inspection_id, first_row_id, first_start = inspection.starts[0]
        self.assertEqual(task.task_id, first_row_id)
        self.assertEqual(task.inspection_id, first_inspection_id)
        self.assertTrue(first_start)

        interrupts, _output = await run_pass(
            graph,
            Command(resume="Approve"),
            config,
            workflow,
            inspection,
        )
        self.assertEqual(interrupts, [])
        self.assertEqual(len(workflow.tasks), 1)
        self.assertEqual(workflow.tasks[0].status, "DONE")
        second_inspection_id, second_row_id, second_start = inspection.starts[1]
        self.assertEqual(second_row_id, first_row_id)
        self.assertEqual(second_inspection_id, first_inspection_id)
        self.assertFalse(second_start)

        phases = [
            event.phase
            for event in frontend.events
            if isinstance(event, WorkflowEvent) and event.task_id == task.task_id
        ]
        self.assertEqual(
            phases,
            [
                "task_start",
                "task_inspection",
                "task_waiting",
                "task_resume",
                "task_finish",
            ],
        )
        transcript = store.get(first_inspection_id)
        assert transcript is not None
        self.assertEqual(transcript.inspection_type, "workflow")
        self.assertEqual(transcript.status, "DONE")
        tool_calls = [
            event
            for event in transcript.events
            if event.kind == "tool_call" and event.call_id == "approval-call"
        ]
        tool_results = [
            event
            for event in transcript.events
            if event.kind == "tool_result" and event.call_id == "approval-call"
        ]
        assistants = [event for event in transcript.events if event.kind == "assistant"]
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(len(tool_results), 1)
        self.assertIn("APPROVED", tool_results[0].text)
        self.assertEqual(len(assistants), 1)
        self.assertIn("FINAL:", assistants[0].text)

    async def test_concurrent_same_name_children_bind_without_crossing(self) -> None:
        frontend = RecordingFrontend()
        workflow = WorkflowCoordinator(FrontendEmitter(frontend))
        await workflow.consume(
            AsyncItems(
                [
                    {"id": f"task-{index}", "name": "agent", "input": {}}
                    for index in range(3)
                ]
            )
        )
        store = LiveInspectionStore()
        inspection = RecordingInspectionCoordinator(store)

        async def result(index: int) -> str:
            return f"response-{index}"

        children = [
            type(
                "Child",
                (),
                {
                    "trigger_call_id": f"task-{index}",
                    "graph_name": "agent",
                    "task_input": f"request-{index}",
                    "path": (f"agent:task-{index}",),
                    "messages": AsyncItems([]),
                    "tool_calls": AsyncItems([]),
                    "output": result(index),
                    "status": "success",
                },
            )()
            for index in range(3)
        ]

        await consume_workflow_inspections(
            AsyncItems(children),
            inspection,
            workflow.bind_inspection,
        )

        self.assertEqual(
            [task.task_id for task in workflow.tasks],
            ["task-0", "task-1", "task-2"],
        )
        inspection_ids = [task.inspection_id for task in workflow.tasks]
        self.assertEqual(len(set(inspection_ids)), 3)
        for index, inspection_id in enumerate(inspection_ids):
            transcript = store.get(inspection_id)
            assert transcript is not None
            self.assertEqual(transcript.events[0].text, f"request-{index}")
            self.assertEqual(transcript.events[-1].text, f"response-{index}")
            self.assertNotIn(
                f"response-{(index + 1) % 3}",
                [event.text for event in transcript.events],
            )


if __name__ == "__main__":
    unittest.main()
