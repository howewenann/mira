"""Native Workflow agent inspection and HITL integration coverage."""

from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from typing import Annotated, Any, TypedDict

from deepagents.middleware.subagents import create_sub_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langgraph.stream.transformers import TasksTransformer
from langgraph.types import Command, interrupt
from pydantic import PrivateAttr

from core.execution.inspection.live import LiveInspectionStore
from core.execution.inspection.subagents import SubagentInspectionCoordinator
from core.execution.streams.output import capture_output, collect_interrupts
from core.execution.streams.subagents import consume_workflow_inspections
from core.execution.workflows import WorkflowCoordinator, execute_workflow
from core.interface import (
    ApprovalRequest,
    AskUserRequest,
    FrontendEmitter,
    FrontendEvent,
    MCPElicitationRequest,
    WorkflowEvent,
)


class RecordingFrontend:
    def __init__(self) -> None:
        self.events: list[FrontendEvent] = []

    def emit(self, event: FrontendEvent) -> None:
        self.events.append(event)

    async def request(self, _request: Any) -> Any:
        raise AssertionError("This test resumes the native interrupt directly")


class RespondingFrontend(RecordingFrontend):
    def __init__(self) -> None:
        super().__init__()
        self.requests: list[Any] = []

    async def request(self, request: Any) -> Any:
        self.requests.append(request)
        if isinstance(request, AskUserRequest):
            return "selected"
        if isinstance(request, MCPElicitationRequest):
            return {"responses": {"city": {"action": "accept", "content": "Osaka"}}}
        if isinstance(request, ApprovalRequest):
            return [{"type": "allow_once"}]
        raise AssertionError(f"Unexpected request: {request!r}")


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

    def workflow_started(
        self,
        subagent: Any,
        title: str,
        task: str,
    ) -> tuple[str, str, bool]:
        started = super().workflow_started(subagent, title, task)
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


class StaticWorkflowModel(BaseChatModel):
    """Return one deterministic response for nested-agent concurrency tests."""

    _response: str = PrivateAttr()

    def __init__(self, response: str) -> None:
        super().__init__()
        self._response = response

    @property
    def _llm_type(self) -> str:
        return "workflow-static-test"

    def bind_tools(
        self,
        _tools: Any,
        *,
        tool_choice: Any | None = None,
        **kwargs: Any,
    ) -> "StaticWorkflowModel":
        del tool_choice, kwargs
        return self

    def _generate(
        self,
        _messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self._response))]
        )


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


class ResumeState(TypedDict):
    result: Any


def build_resume_workflow(value: dict[str, Any]):
    graph = StateGraph(ResumeState)

    def pause(_state: ResumeState) -> dict[str, Any]:
        return {"result": interrupt(value)}

    graph.add_node("pause", pause)
    graph.add_edge(START, "pause")
    graph.add_edge("pause", END)
    return graph.compile(checkpointer=InMemorySaver())


class CountingRun:
    """Transparent stream wrapper that counts final-output consumption."""

    def __init__(self, run: Any) -> None:
        self.run = run
        self.extensions = run.extensions
        self.subgraphs = run.subgraphs
        self.output_calls = 0

    def __aiter__(self):
        return self.run.__aiter__()

    async def __aenter__(self):
        await self.run.__aenter__()
        return self

    async def __aexit__(self, *args: Any):
        return await self.run.__aexit__(*args)

    def output(self):
        self.output_calls += 1
        return self.run.output()

    async def interrupts(self):
        return []


class CountingGraph:
    def __init__(self, graph: Any) -> None:
        self.graph = graph
        self.runs: list[CountingRun] = []

    async def astream_events(self, *args: Any, **kwargs: Any) -> CountingRun:
        run = CountingRun(await self.graph.astream_events(*args, **kwargs))
        self.runs.append(run)
        return run


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
                workflow.agent_started,
                workflow.agent_waiting,
                workflow.agent_finished,
            ),
            capture_output(run.output(), output),
        )
    interrupts = await collect_interrupts(run, output.get("value"))
    return interrupts, output.get("value")


class WorkflowInspectionIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_parallel_nodes_each_keep_parallel_agents_with_native_owner(self) -> None:
        def agent(name: str, response: str) -> Any:
            return create_sub_agent(
                {
                    "name": name,
                    "description": f"Deterministic {name}",
                    "system_prompt": "Respond once.",
                    "model": StaticWorkflowModel(response),
                    "tools": [],
                }
            )

        agent_a = agent("researcher", "A complete")
        agent_b = agent("critic", "B complete")
        checker_a = agent("checker", "C complete")
        checker_b = agent("checker", "D complete")
        parent = StateGraph(MessagesState)

        async def analyse(state: MessagesState) -> dict[str, Any]:
            request = {"messages": list(state["messages"])}
            left, right = await asyncio.gather(
                agent_a.ainvoke(request),
                agent_b.ainvoke(request),
            )
            return {
                "messages": [
                    AIMessage(
                        content=(
                            f"{left['messages'][-1].content}; "
                            f"{right['messages'][-1].content}"
                        )
                    )
                ]
            }

        async def verify(state: MessagesState) -> dict[str, Any]:
            request = {"messages": list(state["messages"])}
            left, right = await asyncio.gather(
                checker_a.ainvoke(request),
                checker_b.ainvoke(request),
            )
            return {
                "messages": [
                    AIMessage(
                        content=(
                            f"{left['messages'][-1].content}; "
                            f"{right['messages'][-1].content}"
                        )
                    )
                ]
            }

        parent.add_node("analyse", analyse)
        parent.add_node("verify", verify)
        parent.add_edge(START, "analyse")
        parent.add_edge(START, "verify")
        parent.add_edge("analyse", END)
        parent.add_edge("verify", END)
        frontend = RecordingFrontend()
        store = LiveInspectionStore()
        output = await execute_workflow(
            parent.compile(),
            {"messages": [{"role": "user", "content": "Compare."}]},
            emitter=FrontendEmitter(frontend),
            inspection=SubagentInspectionCoordinator(store),
            workflow_id="parallel-agents",
            workflow_name="parallel-agents",
        )

        workflow_events = [
            event for event in frontend.events if isinstance(event, WorkflowEvent)
        ]
        root_starts = [
            event for event in workflow_events if event.phase == "task_start"
        ]
        agent_starts = [
            event for event in workflow_events if event.phase == "agent_start"
        ]
        agent_finishes = [
            event for event in workflow_events if event.phase == "agent_finish"
        ]
        self.assertEqual(len(root_starts), 2)
        self.assertEqual({event.step for event in root_starts}, {1})
        self.assertEqual(len(agent_starts), 4)
        self.assertEqual(len({event.agent_id for event in agent_starts}), 4)
        self.assertEqual(
            {event.task_id for event in agent_starts},
            {event.task_id for event in root_starts},
        )
        self.assertEqual(
            {
                task_id: sum(event.task_id == task_id for event in agent_starts)
                for task_id in {event.task_id for event in root_starts}
            },
            {event.task_id: 2 for event in root_starts},
        )
        self.assertEqual(len(agent_finishes), 4)
        combined = " ".join(str(message.content) for message in output["messages"])
        self.assertIn("A complete; B complete", combined)
        self.assertIn("C complete; D complete", combined)
        self.assertTrue(
            all(store.get(event.agent_id).inspection_type == "subagent" for event in agent_starts)
        )

    async def test_shared_executor_consumes_output_once_and_returns_native_state(self) -> None:
        graph = StateGraph(ResumeState)
        graph.add_node("done", lambda _state: {"result": ("native", {1, 2})})
        graph.add_edge(START, "done")
        graph.add_edge("done", END)
        counting = CountingGraph(graph.compile())
        frontend = RespondingFrontend()
        inspection = SubagentInspectionCoordinator(LiveInspectionStore())

        output = await execute_workflow(
            counting,
            {},
            emitter=FrontendEmitter(frontend),
            inspection=inspection,
            workflow_id="counting",
        )

        self.assertEqual(output["result"], ("native", {1, 2}))
        self.assertEqual(len(counting.runs), 1)
        self.assertEqual(counting.runs[0].output_calls, 1)

    async def test_shared_executor_resumes_ask_user_mcp_and_approval_interrupts(self) -> None:
        cases = [
            (
                {"type": "ask_user", "question": "Choose", "options": ["selected"]},
                AskUserRequest,
                "selected",
            ),
            (
                {"type": "mcp_elicitation", "message": "City", "requestedSchema": {}},
                MCPElicitationRequest,
                {"responses": {"city": {"action": "accept", "content": "Osaka"}}},
            ),
            (
                {"action_requests": [{"name": "shell", "args": {"command": "echo ok"}}]},
                ApprovalRequest,
                {"decisions": [{"type": "approve"}]},
            ),
        ]
        for index, (interrupt_value, request_type, expected) in enumerate(cases):
            with self.subTest(request=request_type.__name__):
                frontend = RespondingFrontend()
                output = await execute_workflow(
                    build_resume_workflow(interrupt_value),
                    {},
                    emitter=FrontendEmitter(frontend),
                    inspection=SubagentInspectionCoordinator(LiveInspectionStore()),
                    workflow_id=f"resume-{index}",
                    config={"configurable": {"thread_id": f"resume-{index}"}},
                    action_agent=SimpleNamespace(mira_backend=None),
                )

                self.assertEqual(output["result"], expected)
                self.assertEqual(len(frontend.requests), 1)
                self.assertIsInstance(frontend.requests[0], request_type)

    async def test_interrupted_child_reuses_task_and_agent_on_resume(self) -> None:
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
        first_inspection_id, owner_task_id, first_start = inspection.starts[0]
        self.assertEqual(task.task_id, owner_task_id)
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
        second_inspection_id, second_owner_task_id, second_start = inspection.starts[1]
        self.assertEqual(second_owner_task_id, owner_task_id)
        self.assertEqual(second_inspection_id, first_inspection_id)
        self.assertFalse(second_start)

        task_phases = [
            event.phase
            for event in frontend.events
            if isinstance(event, WorkflowEvent)
            and event.task_id == task.task_id
            and event.phase.startswith("task_")
        ]
        self.assertEqual(
            task_phases,
            [
                "task_start",
                "task_waiting",
                "task_resume",
                "task_finish",
            ],
        )
        agent_events = [
            event
            for event in frontend.events
            if isinstance(event, WorkflowEvent) and event.agent_id == first_inspection_id
        ]
        self.assertEqual(
            [event.phase for event in agent_events],
            ["agent_start", "agent_waiting", "agent_resume", "agent_finish"],
        )
        transcript = store.get(first_inspection_id)
        assert transcript is not None
        self.assertEqual(transcript.inspection_type, "subagent")
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
        self.assertEqual(len(tool_calls), 1, transcript.events)
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
            workflow.agent_started,
            workflow.agent_waiting,
            workflow.agent_finished,
        )

        self.assertEqual(
            [task.task_id for task in workflow.tasks],
            ["task-0", "task-1", "task-2"],
        )
        agent_starts = [
            event
            for event in frontend.events
            if isinstance(event, WorkflowEvent) and event.phase == "agent_start"
        ]
        inspection_ids = [event.agent_id for event in agent_starts]
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
