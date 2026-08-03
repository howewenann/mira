"""Integration coverage for Plan/Goal response-status correction routing."""

from __future__ import annotations

import unittest
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from agent.middleware import (
    CORRECTION_SOURCE,
    CorrectionMiddleware,
    PlanningStageEnforcementMiddleware,
)
from agent.planning.response_status import (
    PLANNING_RESPONSE_STATUS_FAILURE,
    PlanningResponseStatusRule,
)
from runtime import runner
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
            [HumanMessage, AIMessage, HumanMessage, AIMessage, ToolMessage, AIMessage],
        )
        self.assertEqual(messages[3].tool_calls[0]["id"], "call-read")
        self.assertEqual(messages[4].content, "contents:session/store.py")
        self.assertEqual(
            messages[5].content,
            "Investigation complete.\nRESPONSE_STATUS: COMPLETE",
        )
        self.assertTrue(
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
        self.assertTrue(
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

                self.assertEqual(len(result["messages"]), 2 * cap + 4)
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
