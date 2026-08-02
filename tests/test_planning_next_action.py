"""Integration coverage for Plan/Goal next-action retry routing."""

from __future__ import annotations

import unittest
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from agent.middleware import PlanningStageMiddleware
from agent.planning.policy import PLANNING_NEXT_ACTION_FAILURE, PLANNING_NEXT_ACTION_SOURCE


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


class PlanningNextActionIntegrationTests(unittest.IsolatedAsyncioTestCase):
    """Exercise native after_agent revision routing with a real graph."""

    async def test_invalid_plan_prose_retries_then_runs_tool_and_accepts_answer(self) -> None:
        model = BindableFakeModel(
            responses=[
                AIMessage(content="I'll inspect next.\nNEXT_ACTION: RESEARCH"),
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
                AIMessage(content="Investigation complete.\nNEXT_ACTION: ANSWER"),
            ]
        )
        graph = create_agent(
            model=model,
            tools=[inspect_workspace],
            middleware=[PlanningStageMiddleware(max_retries=2)],
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
            [HumanMessage, AIMessage, ToolMessage, AIMessage],
        )
        self.assertEqual(messages[1].tool_calls[0]["id"], "call-read")
        self.assertEqual(messages[2].content, "contents:session/store.py")
        self.assertEqual(messages[3].content, "Investigation complete.\n")
        self.assertFalse(
            any(
                getattr(message, "additional_kwargs", {}).get("lc_source")
                == PLANNING_NEXT_ACTION_SOURCE
                for message in messages
            )
        )

    async def test_goal_research_rejects_plan_marker_then_uses_prepare_goal(self) -> None:
        model = BindableFakeModel(
            responses=[
                AIMessage(content="Ready.\nNEXT_ACTION: PREPARE_PLAN"),
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
                AIMessage(content="Goal prepared.\nNEXT_ACTION: ANSWER"),
            ]
        )
        graph = create_agent(
            model=model,
            tools=[prepare_goal],
            middleware=[PlanningStageMiddleware(max_retries=2)],
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
        self.assertFalse(any("PREPARE_PLAN" in str(message.content) for message in result["messages"]))

    async def test_configured_retry_cap_finishes_with_explicit_incomplete_message(self) -> None:
        model = BindableFakeModel(
            responses=[
                AIMessage(content="Later.\nNEXT_ACTION: RESEARCH"),
                AIMessage(content="Still later.\nNEXT_ACTION: RESEARCH"),
            ]
        )
        graph = create_agent(
            model=model,
            tools=[],
            middleware=[PlanningStageMiddleware(max_retries=1)],
        )

        result = await graph.ainvoke(
            {
                "messages": [HumanMessage(content="inspect")],
                "planning_stage": "plan_research",
            }
        )

        self.assertEqual(len(result["messages"]), 2)
        self.assertEqual(result["messages"][-1].content, PLANNING_NEXT_ACTION_FAILURE)


if __name__ == "__main__":
    unittest.main()
