"""Tests for the standalone Definition-of-Done model pathway."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage

from agent.planning.criteria import (
    INITIAL_CRITERIA_PROMPT,
    REVISION_CRITERIA_PROMPT,
    SUCCESS_CRITERIA_SOURCE,
    SuccessCriteriaService,
)


class GoalCriteriaTests(unittest.IsolatedAsyncioTestCase):
    async def test_initial_generation_receives_objective_and_returns_markdown_only(self) -> None:
        model = type("Model", (), {})()
        model.ainvoke = AsyncMock(return_value=AIMessage(content="- Report exists\n- Evidence is cited"))
        with patch("agent.planning.criteria.get_llm", return_value=model):
            result = await SuccessCriteriaService({}).generate("Compare two options")

        self.assertEqual(result, "- Report exists\n- Evidence is cited")
        messages = model.ainvoke.await_args.args[0]
        self.assertIn("Do not create an execution plan", messages[0].content)
        self.assertIn("<objective>\nCompare two options\n</objective>", messages[1].content)
        self.assertNotIn("<research_context>", messages[1].content)
        self.assertEqual(
            model.ainvoke.await_args.kwargs["config"]["metadata"]["lc_source"],
            SUCCESS_CRITERIA_SOURCE,
        )

    async def test_generation_receives_only_bounded_research_handoff(self) -> None:
        model = type("Model", (), {})()
        model.ainvoke = AsyncMock(return_value=AIMessage(content="- Existing storage is reused"))
        with patch("agent.planning.criteria.get_llm", return_value=model):
            await SuccessCriteriaService({}).generate(
                "Finish persistence",
                "Sessions are JSON-backed and normalized on save.",
            )

        messages = model.ainvoke.await_args.args[0]
        self.assertIn("<research_context>\nSessions are JSON-backed", messages[1].content)
        self.assertIn("objective is authoritative", messages[0].content.lower())

    async def test_generation_binds_polished_objective_to_authoritative_request(self) -> None:
        model = type("Model", (), {})()
        model.ainvoke = AsyncMock(return_value=AIMessage(content="- story.md contains the story"))
        with patch("agent.planning.criteria.get_llm", return_value=model):
            await SuccessCriteriaService({}).generate(
                "Create an approximately 20-word story and save it as story.md in the project root.",
                "The project root is writable.",
                authoritative_request="write me a story of about 20 words. save it to a .md file in the root directory",
            )

        messages = model.ainvoke.await_args.args[0]
        self.assertIn("authoritative request", messages[0].content.lower())
        self.assertIn(
            "<authoritative_request>\nwrite me a story of about 20 words. save it to a .md file in the root directory\n</authoritative_request>",
            messages[1].content,
        )
        self.assertIn(
            "<objective>\nCreate an approximately 20-word story and save it as story.md in the project root.\n</objective>",
            messages[1].content,
        )

    async def test_revision_has_no_plan_input_and_handles_plan_only_feedback(self) -> None:
        model = type("Model", (), {})()
        previous = "- The comparison covers both options"
        model.ainvoke = AsyncMock(return_value=AIMessage(content=previous))
        with patch("agent.planning.criteria.get_llm", return_value=model):
            result = await SuccessCriteriaService({}).revise(
                "Compare two options",
                previous,
                "Make the plan shorter",
                "The current proposal already uses the session store.",
            )

        self.assertEqual(result, previous)
        messages = model.ainvoke.await_args.args[0]
        self.assertIn("feedback may mention \"the plan\"", messages[0].content)
        self.assertIn("<previous_criteria>", messages[1].content)
        self.assertNotIn("previous_plan", messages[1].content)
        self.assertIn("<research_context>", messages[1].content)

    async def test_blank_model_response_is_rejected(self) -> None:
        model = type("Model", (), {})()
        model.ainvoke = AsyncMock(return_value=AIMessage(content="  "))
        with patch("agent.planning.criteria.get_llm", return_value=model):
            with self.assertRaisesRegex(RuntimeError, "empty response"):
                await SuccessCriteriaService({}).generate("Do work")

    async def test_diagnostic_mechanics_are_excluded_from_required_final_state(self) -> None:
        expected = (
            "- rubric_done.txt exists and contains exactly DONE\n"
            "- rubric_probe.txt exists and contains exactly MIRA_RUBRIC_PROBE_7F3A"
        )
        model = type("Model", (), {})()
        model.ainvoke = AsyncMock(return_value=AIMessage(content=expected))
        request = (
            "Diagnostic test. Do not modify, read, list, grep, or otherwise inspect "
            "rubric_probe.txt. Its state is intentionally acceptance-checker-only evidence. "
            "Create only rubric_done.txt containing exactly DONE. The no-inspection "
            "instruction is diagnostic mechanics and must not become a Success Criterion. "
            "Success Criteria must describe the required final state: rubric_done.txt exists "
            "and contains exactly DONE; rubric_probe.txt exists and contains exactly "
            "MIRA_RUBRIC_PROBE_7F3A."
        )
        objective = (
            "rubric_done.txt exists and contains exactly DONE; rubric_probe.txt exists and "
            "contains exactly MIRA_RUBRIC_PROBE_7F3A."
        )

        with patch("agent.planning.criteria.get_llm", return_value=model):
            result = await SuccessCriteriaService({}).generate(
                objective,
                authoritative_request=request,
            )

        self.assertEqual(result, expected)
        system_prompt = model.ainvoke.await_args.args[0][0].content
        self.assertIn("required final state, not diagnostic mechanics", system_prompt)
        self.assertIn("acceptance-checker-only evidence gap", system_prompt)
        self.assertNotIn("Do not inspect rubric_probe.txt", result)

    def test_initial_and_revision_prompts_share_final_state_guidance(self) -> None:
        for prompt in (INITIAL_CRITERIA_PROMPT, REVISION_CRITERIA_PROMPT):
            self.assertIn("required final state, not diagnostic mechanics", prompt)
            self.assertIn("control how the task is carried out", prompt)
