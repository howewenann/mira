"""Integration coverage for Plan/Goal response-status correction routing."""

from __future__ import annotations

import asyncio
import unittest
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import NodeCancelledError
from langgraph.types import Command, interrupt
from pydantic import Field

from agent.middleware import (
    CORRECTION_SOURCE,
    CorrectionMiddleware,
    PlanningStageEnforcementMiddleware,
)
from agent.middleware.file_references import FileReferenceMiddleware
from agent.planning.response_status import (
    PLANNING_RESPONSE_STATUS_FAILURE,
    PlanningResponseStatusRule,
)
from core.execution import runner
from core.execution.handoff import synchronize_conversation
from tests.test_runner import RunTurnRenderer


def correction_middleware(max_retries: int = 2) -> list[Any]:
    return [
        PlanningStageEnforcementMiddleware(),
        CorrectionMiddleware(
            rules=(
                PlanningResponseStatusRule(workflow="plan"),
                PlanningResponseStatusRule(workflow="goal"),
            ),
            max_retries=max_retries,
        ),
    ]


class BindableFakeModel(FakeMessagesListChatModel):
    """Fake model that accepts LangChain tool binding."""

    def bind_tools(self, tools: list[Any], *, tool_choice: Any = None, **kwargs: Any) -> Any:
        return self


class RecordingModel(BindableFakeModel):
    requests: list[list[Any]] = Field(default_factory=list)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.requests.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def system_text(messages: list[Any]) -> str:
    return "\n".join(m.text for m in messages if isinstance(m, SystemMessage))


@tool
def inspect_workspace(path: str) -> str:
    """Read one fake workspace path."""
    return f"contents:{path}"


@tool
def prepare_goal(objective: str) -> str:
    """Prepare one fake Goal."""
    return f"prepared:{objective}"


@tool
def prepare_plan(objective: str) -> str:
    """Prepare one fake Plan."""
    return f"prepared:{objective}"


@tool("finalize_plan")
def finalize_plan_stub(title: str) -> str:
    """Fake finalization control that must not run during research."""
    raise AssertionError(f"finalize_plan executed unexpectedly: {title}")


@tool("show_plan")
def show_plan_stub() -> str:
    """Fake retained Plan display control."""
    return "current Plan shown"


@tool("finalize_goal")
def finalize_goal_stub(title: str) -> str:
    """Fake finalization control that must not run during research."""
    raise AssertionError(f"finalize_goal executed unexpectedly: {title}")


@tool("show_goal")
def show_goal_stub() -> str:
    """Fake retained Goal display control."""
    return "current Goal shown"


@tool("ask_user")
def forbidden_finalize_ask_user(question: str) -> str:
    """Fail if finalization enforcement lets a hidden prompt execute."""
    raise AssertionError(f"ask_user executed during finalization: {question}")


@tool("finalize_plan")
def successful_finalize_plan(title: str) -> str:
    """Return a visible result after finalization enforcement accepts the call."""
    return f"finalized Plan:{title}"


@tool("finalize_goal")
def successful_finalize_goal(title: str) -> str:
    """Return a visible result after finalization enforcement accepts the call."""
    return f"finalized Goal:{title}"


class PlanningResponseStatusIntegrationTests(unittest.IsolatedAsyncioTestCase):
    """Exercise native after_agent revision routing with a real graph."""

    async def test_transient_retry_preserves_requests_checkpoints_events_and_handoff(self) -> None:
        for workflow in ("plan", "goal"):
            with self.subTest(workflow=workflow):
                model = RecordingModel(responses=[
                    AIMessage(content="Need research.\nRESPONSE_STATUS: NEEDS_RESEARCH"),
                    AIMessage(content="Done.\nRESPONSE_STATUS: COMPLETE"),
                    AIMessage(content="New answer.\nRESPONSE_STATUS: COMPLETE"),
                ])
                graph = create_agent(
                    model=model, tools=[], system_prompt=SystemMessage(
                        content=[{"type": "text", "text": "Original system instructions"}],
                        additional_kwargs={"original": True},
                    ),
                    middleware=[FileReferenceMiddleware(), *correction_middleware()],
                    checkpointer=InMemorySaver(),
                )
                config = {"configurable": {"thread_id": workflow}}
                user = HumanMessage(
                    content="Inspect @config.py", id="user", additional_kwargs={"original": True},
                )
                events = [event async for event in graph.astream(
                    {"messages": [user], "planning_stage": f"{workflow}_research"},
                    config, stream_mode="custom",
                )]
                self.assertEqual(len(model.requests), 2)
                rule = PlanningResponseStatusRule(workflow=workflow)
                feedback = rule.inspect(model.responses[0], {}).retry_prompt
                first, retry = model.requests
                self.assertNotIn(feedback, system_text(first))
                self.assertEqual(system_text(retry).count(feedback), 1)
                for request in model.requests:
                    self.assertIn("Original system instructions", system_text(request))
                    self.assertIn("/config.py", system_text(request))
                    self.assertEqual([m for m in request if isinstance(m, HumanMessage)], [user])
                    self.assertTrue(request[0].additional_kwargs["original"])
                self.assertEqual([e["attempt"] for e in events], [1])
                self.assertEqual(events[0]["protocol"], rule.protocol_id)
                self.assertEqual(events[0]["retry_prompt"], feedback)
                self.assertFalse(events[0]["exhausted"])
                state = (await graph.aget_state(config)).values
                self.assertEqual(len(state["messages"]), 3)
                self.assertEqual(state["messages"][0], user)
                self.assertEqual(state["_correction_pending"], {})
                self.assertEqual(state["_correction_retries"], {})
                # Inspect all intermediate checkpoints, not only the final one.
                async for snapshot in graph.aget_state_history(config):
                    if not snapshot.values.get("messages"):
                        continue  # The initial input checkpoint precedes message reduction.
                    self.assertEqual(
                        [m for m in snapshot.values.get("messages", []) if isinstance(m, HumanMessage)],
                        [user],
                    )
                target = create_agent(
                    model=RecordingModel(responses=[AIMessage(content="Act")]),
                    tools=[], checkpointer=InMemorySaver(),
                )
                self.assertTrue(await synchronize_conversation(graph, workflow, target, "act"))
                inherited = (await target.aget_state({"configurable": {"thread_id": "act"}})).values
                self.assertEqual(inherited["messages"], state["messages"])
                self.assertNotIn("_correction_pending", inherited)
                result = await graph.ainvoke(
                    {"messages": [HumanMessage(content="Independent request")]}, config,
                )
                self.assertNotIn(feedback, system_text(model.requests[-1]))
                self.assertNotIn("_correction_pending", result)

    def test_sync_native_retry_and_exhaustion_events(self) -> None:
        for workflow in ("plan", "goal"):
            with self.subTest(workflow=workflow):
                model = RecordingModel(responses=[AIMessage(content="Invalid") for _ in range(3)])
                graph = create_agent(
                    model=model, tools=[], middleware=correction_middleware(), checkpointer=InMemorySaver(),
                )
                config = {"configurable": {"thread_id": workflow}}
                events = list(graph.stream(
                    {"messages": [HumanMessage(content="Request")], "planning_stage": f"{workflow}_research"},
                    config, stream_mode="custom",
                ))
                self.assertEqual(len(model.requests), 3)
                self.assertEqual([e["attempt"] for e in events], [1, 2, 2])
                self.assertEqual([e["exhausted"] for e in events], [False, False, True])
                self.assertEqual(events[-1]["terminal_text"], PLANNING_RESPONSE_STATUS_FAILURE)
                feedback = events[0]["retry_prompt"]
                for request in model.requests[1:]:
                    self.assertEqual(system_text(request).count(feedback), 1)
                state = graph.get_state(config).values
                self.assertEqual(len(state["messages"]), 5)
                self.assertEqual(state["messages"][-1].content, PLANNING_RESPONSE_STATUS_FAILURE)
                self.assertEqual(state["_correction_pending"], {})
                self.assertEqual(state["_correction_retries"], {})
                self.assertEqual(sum(isinstance(m, HumanMessage) for m in state["messages"]), 1)

    async def test_tool_continuation_clears_pending_and_resets_retry_budget(self) -> None:
        model = RecordingModel(responses=[
            AIMessage(content="Need research.\nRESPONSE_STATUS: NEEDS_RESEARCH"),
            AIMessage(content="", tool_calls=[{
                "name": "inspect_workspace", "args": {"path": "x"}, "id": "read",
            }]),
            AIMessage(content="Need input.\nRESPONSE_STATUS: NEEDS_USER_INPUT"),
            AIMessage(content="Done.\nRESPONSE_STATUS: COMPLETE"),
        ])
        graph = create_agent(
            model=model, tools=[inspect_workspace], middleware=correction_middleware(1),
            checkpointer=InMemorySaver(),
        )
        config = {"configurable": {"thread_id": "tools"}}
        events = [e async for e in graph.astream(
            {"messages": [HumanMessage(content="Inspect")], "planning_stage": "plan_research"},
            config, stream_mode="custom",
        )]
        self.assertEqual(len(model.requests), 4)
        self.assertIn("Perform that research now", system_text(model.requests[1]))
        self.assertNotIn("Your previous response", system_text(model.requests[2]))
        self.assertIn("Call ask_user now", system_text(model.requests[3]))
        self.assertNotIn("Perform that research now", system_text(model.requests[3]))
        self.assertEqual([e["attempt"] for e in events], [1, 1])
        self.assertEqual((await graph.aget_state(config)).values["_correction_pending"], {})

    async def test_checkpoint_resume_keeps_only_the_intended_retry_guidance(self) -> None:
        saver = InMemorySaver()
        model = RecordingModel(responses=[
            AIMessage(content="Invalid"), AIMessage(content="Done.\nRESPONSE_STATUS: COMPLETE"),
        ])
        graph = create_agent(
            model=model, tools=[], middleware=correction_middleware(), checkpointer=saver,
            interrupt_after=["CorrectionMiddleware.after_agent"],
        )
        config = {"configurable": {"thread_id": "resume"}}
        await graph.ainvoke(
            {"messages": [HumanMessage(content="Inspect")], "planning_stage": "goal_research"}, config,
        )
        paused = await graph.aget_state(config)
        self.assertTrue(paused.next)
        self.assertTrue(paused.values["_correction_pending"])
        self.assertEqual(len(model.requests), 1)
        # Reconstruct middleware as on process restart: no mutable instance cache.
        resumed = create_agent(model=model, tools=[], middleware=correction_middleware(), checkpointer=saver)
        await resumed.ainvoke(None, config)
        self.assertEqual(len(model.requests), 2)
        self.assertEqual(system_text(model.requests[-1]).count("Your previous response"), 1)
        self.assertEqual((await resumed.aget_state(config)).values["_correction_pending"], {})

    async def test_cancelled_retry_does_not_contaminate_a_new_turn(self) -> None:
        class CancelRetry(AgentMiddleware):
            async def awrap_model_call(self, request, handler):
                if request.state.get("_correction_pending"):
                    raise asyncio.CancelledError
                return await handler(request)

        saver = InMemorySaver()
        model = RecordingModel(responses=[
            AIMessage(content="Invalid"), AIMessage(content="Done.\nRESPONSE_STATUS: COMPLETE"),
        ])
        graph = create_agent(
            model=model, tools=[], middleware=[*correction_middleware(), CancelRetry()], checkpointer=saver,
        )
        config = {"configurable": {"thread_id": "cancel"}}
        with self.assertRaises(NodeCancelledError):
            await graph.ainvoke(
                {"messages": [HumanMessage(content="Old request")], "planning_stage": "plan_research"}, config,
            )
        self.assertTrue((await graph.aget_state(config)).values["_correction_pending"])
        # A new input starts before_agent even though the previous invocation stopped.
        await graph.ainvoke({"messages": [HumanMessage(content="New request")]}, config)
        self.assertEqual(len(model.requests), 2)
        self.assertNotIn("Your previous response", system_text(model.requests[-1]))
        self.assertEqual((await graph.aget_state(config)).values["_correction_pending"], {})

    async def test_native_tool_interrupt_resume_has_no_stale_instruction(self) -> None:
        @tool("ask_user")
        def ask_user(question: str) -> str:
            """Collect a material decision from the user."""
            return interrupt(question)

        model = RecordingModel(responses=[
            AIMessage(content="Need input.\nRESPONSE_STATUS: NEEDS_USER_INPUT"),
            AIMessage(content="", tool_calls=[{
                "name": "ask_user", "args": {"question": "Which target?"}, "id": "ask",
            }]),
            AIMessage(content="Resolved.\nRESPONSE_STATUS: COMPLETE"),
        ])
        graph = create_agent(
            model=model, tools=[ask_user], middleware=correction_middleware(), checkpointer=InMemorySaver(),
        )
        config = {"configurable": {"thread_id": "hitl"}}
        await graph.ainvoke(
            {"messages": [HumanMessage(content="Prepare a Goal")], "planning_stage": "goal_research"}, config,
        )
        paused = await graph.aget_state(config)
        self.assertTrue(paused.tasks[0].interrupts)
        self.assertEqual(paused.values["_correction_pending"], {})
        self.assertEqual(paused.values["_correction_retries"], {})
        self.assertIn("Call ask_user now", system_text(model.requests[1]))
        await graph.ainvoke(Command(resume="Target A"), config)
        self.assertEqual(len(model.requests), 3)
        self.assertNotIn("Your previous response", system_text(model.requests[2]))
        self.assertEqual((await graph.aget_state(config)).values["_correction_pending"], {})

    async def test_invalid_plan_prose_retries_then_runs_tool_and_accepts_answer(self) -> None:
        model = BindableFakeModel(
            responses=[
                AIMessage(content="I'll inspect next.\nRESPONSE_STATUS: NEEDS_RESEARCH"),
                AIMessage(
                    content="Inspecting now.",
                    tool_calls=[
                        {
                            "name": "inspect_workspace",
                            "args": {"path": "session/store.py"},
                            "id": "call-read",
                        }
                    ],
                ),
                AIMessage(content="Investigation complete.\nRESPONSE_STATUS: COMPLETE"),
            ]
        )
        graph = create_agent(
            model=model,
            tools=[inspect_workspace],
            middleware=correction_middleware(),
        )

        result = await graph.ainvoke(
            {
                "messages": [HumanMessage(content="inspect")],
                "planning_stage": "plan_research",
            }
        )

        messages = result["messages"]
        self.assertEqual(
            [type(message) for message in messages],
            [HumanMessage, AIMessage, AIMessage, ToolMessage, AIMessage],
        )
        self.assertEqual(messages[2].tool_calls[0]["id"], "call-read")
        self.assertEqual(messages[3].content, "contents:session/store.py")
        self.assertEqual(
            messages[4].content,
            "Investigation complete.\nRESPONSE_STATUS: COMPLETE",
        )
        self.assertFalse(
            any(
                getattr(message, "additional_kwargs", {}).get("lc_source")
                == CORRECTION_SOURCE
                for message in messages
            )
        )

    async def test_goal_research_rejects_plan_marker_then_uses_prepare_goal(self) -> None:
        model = BindableFakeModel(
            responses=[
                AIMessage(content="Ready.\nRESPONSE_STATUS: READY_TO_PREPARE_PLAN"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "prepare_goal",
                            "args": {"objective": "Ship it"},
                            "id": "call-goal",
                        }
                    ],
                ),
                AIMessage(content="Goal prepared.\nRESPONSE_STATUS: COMPLETE"),
            ]
        )
        graph = create_agent(
            model=model,
            tools=[prepare_goal],
            middleware=correction_middleware(),
        )

        result = await graph.ainvoke(
            {
                "messages": [HumanMessage(content="prepare a goal")],
                "planning_stage": "goal_research",
            }
        )

        self.assertTrue(
            any(
                isinstance(message, AIMessage)
                and message.tool_calls
                and message.tool_calls[0]["name"] == "prepare_goal"
                for message in result["messages"]
            )
        )
        self.assertTrue(
            any("READY_TO_PREPARE_PLAN" in str(message.content) for message in result["messages"])
        )
        self.assertFalse(
            any(
                getattr(message, "additional_kwargs", {}).get("lc_source") == CORRECTION_SOURCE
                for message in result["messages"]
            )
        )

    async def test_configured_retry_caps_finish_with_explicit_incomplete_message(self) -> None:
        for cap in (1, 2):
            with self.subTest(cap=cap):
                model = BindableFakeModel(
                    responses=[
                        AIMessage(
                            content=(
                                f"Still unresolved {attempt}.\n"
                                "RESPONSE_STATUS: NEEDS_RESEARCH"
                            )
                        )
                        for attempt in range(cap + 1)
                    ]
                )
                graph = create_agent(
                    model=model,
                    tools=[],
                    middleware=correction_middleware(max_retries=cap),
                )

                result = await graph.ainvoke(
                    {
                        "messages": [HumanMessage(content="inspect")],
                        "planning_stage": "plan_research",
                    }
                )

                self.assertEqual(len(result["messages"]), cap + 3)
                self.assertEqual(
                    result["messages"][-1].content,
                    PLANNING_RESPONSE_STATUS_FAILURE,
                )

    async def test_wrong_stage_finalize_plan_returns_tool_error_then_model_uses_show_plan(self) -> None:
        model = BindableFakeModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "finalize_plan",
                            "args": {"title": "Existing Plan"},
                            "id": "call-wrong-present",
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "show_plan", "args": {}, "id": "call-show"}
                    ],
                ),
                AIMessage(content="The retained Plan is shown.\nRESPONSE_STATUS: COMPLETE"),
            ]
        )
        graph = create_agent(
            model=model,
            tools=[finalize_plan_stub, show_plan_stub],
            middleware=correction_middleware(),
        )

        result = await graph.ainvoke(
            {
                "messages": [HumanMessage(content="show me the plan again")],
                "planning_stage": "plan_research",
            }
        )

        wrong = next(
            message
            for message in result["messages"]
            if isinstance(message, ToolMessage) and message.tool_call_id == "call-wrong-present"
        )
        shown = next(
            message
            for message in result["messages"]
            if isinstance(message, ToolMessage) and message.tool_call_id == "call-show"
        )
        self.assertEqual(wrong.status, "error")
        self.assertIn("show_plan", str(wrong.content))
        self.assertEqual(shown.status, "success")
        self.assertEqual(shown.content, "current Plan shown")
        self.assertEqual(
            result["messages"][-1].content,
            "The retained Plan is shown.\nRESPONSE_STATUS: COMPLETE",
        )

    async def test_wrong_stage_finalize_goal_returns_tool_error_then_model_uses_show_goal(self) -> None:
        model = BindableFakeModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "finalize_goal",
                            "args": {"title": "Existing Goal"},
                            "id": "call-wrong-goal",
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "show_goal", "args": {}, "id": "call-goal-show"}
                    ],
                ),
                AIMessage(content="The retained Goal is shown.\nRESPONSE_STATUS: COMPLETE"),
            ]
        )
        graph = create_agent(
            model=model,
            tools=[finalize_goal_stub, show_goal_stub],
            middleware=correction_middleware(),
        )

        result = await graph.ainvoke(
            {
                "messages": [HumanMessage(content="show me the goal again")],
                "planning_stage": "goal_research",
            }
        )

        wrong = next(
            message
            for message in result["messages"]
            if isinstance(message, ToolMessage) and message.tool_call_id == "call-wrong-goal"
        )
        shown = next(
            message
            for message in result["messages"]
            if isinstance(message, ToolMessage) and message.tool_call_id == "call-goal-show"
        )
        self.assertEqual(wrong.status, "error")
        self.assertIn("show_goal", str(wrong.content))
        self.assertEqual(shown.content, "current Goal shown")
        self.assertEqual(
            result["messages"][-1].content,
            "The retained Goal is shown.\nRESPONSE_STATUS: COMPLETE",
        )

    async def test_valid_stage_missing_arguments_use_native_toolnode_retry(self) -> None:
        model = BindableFakeModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "prepare_plan", "args": {}, "id": "call-missing-arg"}
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "prepare_plan",
                            "args": {"objective": "Ship it"},
                            "id": "call-valid-arg",
                        }
                    ],
                ),
                AIMessage(content="Plan prepared.\nRESPONSE_STATUS: COMPLETE"),
            ]
        )
        graph = create_agent(
            model=model,
            tools=[prepare_plan],
            middleware=correction_middleware(),
        )

        result = await graph.ainvoke(
            {
                "messages": [HumanMessage(content="prepare a plan")],
                "planning_stage": "plan_research",
            }
        )

        invalid = next(
            message
            for message in result["messages"]
            if isinstance(message, ToolMessage) and message.tool_call_id == "call-missing-arg"
        )
        valid = next(
            message
            for message in result["messages"]
            if isinstance(message, ToolMessage) and message.tool_call_id == "call-valid-arg"
        )
        self.assertEqual(invalid.status, "error")
        self.assertIn("objective", str(invalid.content))
        self.assertEqual(valid.status, "success")
        self.assertEqual(valid.content, "prepared:Ship it")
        self.assertEqual(
            result["messages"][-1].content,
            "Plan prepared.\nRESPONSE_STATUS: COMPLETE",
        )

    async def test_finalization_rejects_hidden_ask_user_then_accepts_plan_finalizer(self) -> None:
        model = BindableFakeModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "ask_user",
                            "args": {"question": "Should I continue?"},
                            "id": "call-hidden-ask",
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "finalize_plan",
                            "args": {"title": "Safe Plan"},
                            "id": "call-finalize-plan",
                        }
                    ],
                ),
                AIMessage(content="Plan finalized."),
            ]
        )
        graph = create_agent(
            model=model,
            tools=[forbidden_finalize_ask_user, successful_finalize_plan],
            middleware=correction_middleware(),
        )

        result = await graph.ainvoke(
            {
                "messages": [HumanMessage(content="finish the plan")],
                "planning_stage": "plan_finalize",
            }
        )

        hidden = next(
            message
            for message in result["messages"]
            if isinstance(message, ToolMessage) and message.tool_call_id == "call-hidden-ask"
        )
        finalized = next(
            message
            for message in result["messages"]
            if isinstance(message, ToolMessage) and message.tool_call_id == "call-finalize-plan"
        )
        self.assertEqual(hidden.status, "error")
        self.assertIn("Only finalize_plan can be called now.", str(hidden.content))
        self.assertEqual(finalized.status, "success")
        self.assertEqual(finalized.content, "finalized Plan:Safe Plan")
        self.assertNotIn("_correction_retries", result)

    async def test_goal_finalizer_schema_error_retries_natively_without_correction(self) -> None:
        model = BindableFakeModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "finalize_goal", "args": {}, "id": "call-invalid-goal"}
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "finalize_goal",
                            "args": {"title": "Safe Goal"},
                            "id": "call-finalize-goal",
                        }
                    ],
                ),
                AIMessage(content="Goal finalized."),
            ]
        )
        graph = create_agent(
            model=model,
            tools=[successful_finalize_goal],
            middleware=correction_middleware(),
        )

        result = await graph.ainvoke(
            {
                "messages": [HumanMessage(content="finish the goal")],
                "planning_stage": "goal_finalize",
            }
        )

        invalid = next(
            message
            for message in result["messages"]
            if isinstance(message, ToolMessage) and message.tool_call_id == "call-invalid-goal"
        )
        finalized = next(
            message
            for message in result["messages"]
            if isinstance(message, ToolMessage) and message.tool_call_id == "call-finalize-goal"
        )
        self.assertEqual(invalid.status, "error")
        self.assertIn("title", str(invalid.content))
        self.assertEqual(finalized.content, "finalized Goal:Safe Goal")
        self.assertNotIn("_correction_retries", result)

    async def test_real_v3_stream_renders_finalization_rejection_live(self) -> None:
        model = BindableFakeModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "ask_user",
                            "args": {"question": "Should I continue?"},
                            "id": "call-live-hidden",
                        }
                    ],
                ),
                AIMessage(content="Finalization repaired."),
            ]
        )
        graph = create_agent(
            model=model,
            tools=[forbidden_finalize_ask_user, successful_finalize_plan],
            middleware=correction_middleware(),
        )
        renderer = RunTurnRenderer()

        result = await runner.run_turn(
            graph,
            "finish the plan",
            renderer,
            "real-v3-plan-finalize",
            planning_stage="plan_finalize",
        )

        self.assertEqual(result.final_text, "Finalization repaired.")
        self.assertEqual(result.tool_calls, ["ask_user"])
        self.assertEqual(len(result.tool_results), 1)
        self.assertIn("Only finalize_plan can be called now.", result.tool_results[0])
        self.assertEqual(
            [event[0] for event in renderer.events if event[0] in {"tool_call", "tool_error"}],
            ["tool_call", "tool_error"],
        )
        self.assertEqual(
            [event[3] for event in renderer.events if event[0] in {"tool_call", "tool_error"}],
            ["call-live-hidden", "call-live-hidden"],
        )
        self.assertEqual(renderer.ask_user_prompts, [])


if __name__ == "__main__":
    unittest.main()
