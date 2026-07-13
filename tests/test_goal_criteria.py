"""Tests for the standalone Definition-of-Done model pathway."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage

from agent.planning.criteria import GoalCriteriaService


class GoalCriteriaTests(unittest.IsolatedAsyncioTestCase):
    async def test_initial_generation_receives_objective_and_returns_markdown_only(self) -> None:
        model = type("Model", (), {})()
        model.ainvoke = AsyncMock(return_value=AIMessage(content="- Report exists\n- Evidence is cited"))
        with patch("agent.planning.criteria.get_llm", return_value=model):
            result = await GoalCriteriaService({}).generate("Compare two options")

        self.assertEqual(result, "- Report exists\n- Evidence is cited")
        messages = model.ainvoke.await_args.args[0]
        self.assertIn("Do not create an execution plan", messages[0].content)
        self.assertIn("<objective>\nCompare two options\n</objective>", messages[1].content)

    async def test_revision_has_no_plan_input_and_handles_plan_only_feedback(self) -> None:
        model = type("Model", (), {})()
        previous = "- The comparison covers both options"
        model.ainvoke = AsyncMock(return_value=AIMessage(content=previous))
        with patch("agent.planning.criteria.get_llm", return_value=model):
            result = await GoalCriteriaService({}).revise(
                "Compare two options",
                previous,
                "Make the plan shorter",
            )

        self.assertEqual(result, previous)
        messages = model.ainvoke.await_args.args[0]
        self.assertIn("feedback may mention \"the plan\"", messages[0].content)
        self.assertIn("<previous_criteria>", messages[1].content)
        self.assertNotIn("previous_plan", messages[1].content)

    async def test_blank_model_response_is_rejected(self) -> None:
        model = type("Model", (), {})()
        model.ainvoke = AsyncMock(return_value=AIMessage(content="  "))
        with patch("agent.planning.criteria.get_llm", return_value=model):
            with self.assertRaisesRegex(RuntimeError, "empty response"):
                await GoalCriteriaService({}).generate("Do work")
