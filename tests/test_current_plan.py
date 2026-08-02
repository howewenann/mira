"""Focused coverage for conversational durable Plans."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from agent.middleware import PlanningStageEnforcementMiddleware
from agent.factory import ACT_SYSTEM_PROMPT
from agent.planning.criteria import SuccessCriteriaService
from agent.planning.policy import (
    PLANNING_STAGE_PLAN_FINALIZE,
    PLANNING_STAGE_PLAN_RESEARCH,
    PLANNING_STAGE_GOAL_FINALIZE,
    PLANNING_STAGE_GOAL_RESEARCH,
    SHARED_QUESTION_POLICY,
    plan_system_prompt,
)
from session.context import build_resume_context, normalize_session
from session.plans import (
    clear_current_plan,
    current_plan,
    finish_plan_attempt,
    plan_artifact,
    replace_current_plan,
    start_plan_attempt,
)
from session.store import SessionStore
from ui.repl import plan_command_prompt, plan_thread_id
from ui.repl import run_user_turn
from ui.app import MiraApp
from runtime.runner import TurnResult


def artifact(*, rubric: bool = False) -> dict:
    return plan_artifact(
        plan_id="plan-1",
        title="Durable Plan",
        objective="Deliver the requested result.",
        context_and_constraints="Use the current workspace and preserve unrelated work.",
        key_changes=["Inspect the relevant context.", "Produce the deliverable."],
        test_plan=["Check the observable result against the request."],
        assumptions=["No additional assumptions."],
        success_criteria="- The requested result is complete and verified.",
        rubric_enabled=rubric,
        rubric_iterations=4,
    )


class Request:
    def __init__(self, stage: str, tools: list[dict]) -> None:
        self.state = {"planning_stage": stage}
        self.tools = tools
        self.tool_choice = None

    def override(self, **updates):
        value = Request(self.state["planning_stage"], updates.get("tools", self.tools))
        value.tool_choice = updates.get("tool_choice", self.tool_choice)
        return value


class CurrentPlanTests(unittest.TestCase):
    def test_plan_prompt_is_conversational_general_purpose_and_rubric_independent(self) -> None:
        prompt = plan_system_prompt()
        for outcome in ("DISCUSSION", "NEEDS_DECISION", "PLAN_READY"):
            self.assertIn(outcome, prompt)
        self.assertIn("Imperative wording never authorizes execution", prompt)
        self.assertIn("research, analysis, writing, communication, data work", prompt)
        self.assertIn(SHARED_QUESTION_POLICY, prompt)
        self.assertIn(SHARED_QUESTION_POLICY, ACT_SYSTEM_PROMPT)
        self.assertNotIn("SAFE_CONVERSATION", prompt)

    def test_plan_stage_exposes_prepare_then_requires_present(self) -> None:
        tools = [
            {"name": "read_file"},
            {"name": "ask_user"},
            {"name": "prepare_goal"},
            {"name": "prepare_plan"},
            {"name": "present_plan"},
            {"name": "present_goal"},
            {"name": "goal_show"},
            {"name": "plan_show"},
        ]
        middleware = PlanningStageEnforcementMiddleware()
        research = middleware._stage_request(Request(PLANNING_STAGE_PLAN_RESEARCH, tools))
        self.assertEqual(
            [tool["name"] for tool in research.tools],
            ["read_file", "ask_user", "prepare_plan", "goal_show", "plan_show"],
        )
        final = middleware._stage_request(Request(PLANNING_STAGE_PLAN_FINALIZE, tools))
        self.assertEqual([tool["name"] for tool in final.tools], ["present_plan"])
        self.assertEqual(final.tool_choice, "required")
        goal_research = middleware._stage_request(Request(PLANNING_STAGE_GOAL_RESEARCH, tools))
        self.assertEqual(
            [tool["name"] for tool in goal_research.tools],
            ["read_file", "ask_user", "prepare_goal", "goal_show", "plan_show"],
        )
        goal_final = middleware._stage_request(Request(PLANNING_STAGE_GOAL_FINALIZE, tools))
        self.assertEqual([tool["name"] for tool in goal_final.tools], ["present_goal"])
        self.assertEqual(goal_final.tool_choice, "required")

    def test_current_plan_replaces_starts_completes_restarts_and_clears(self) -> None:
        record = {"events": []}
        first = replace_current_plan(record, artifact())
        self.assertEqual(first["status"], "proposed")
        active = start_plan_attempt(record)
        self.assertEqual(active["status"], "active")
        self.assertEqual(active["attempts"], 1)
        completed = finish_plan_attempt(record)
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["completion_source"], "agent-declared")
        restarted = start_plan_attempt(record)
        self.assertEqual(restarted["status"], "active")
        self.assertEqual(restarted["attempts"], 2)
        replacement = artifact(rubric=True)
        replacement["id"] = "plan-2"
        replace_current_plan(record, replacement)
        self.assertEqual(current_plan(record)["id"], "plan-2")
        self.assertEqual(clear_current_plan(record)["id"], "plan-2")
        self.assertIsNone(current_plan(record))

    def test_rubric_completion_requires_satisfied_and_preserves_max_iterations(self) -> None:
        record = {"events": [], "current_plan": artifact(rubric=True)}
        start_plan_attempt(record)
        paused = finish_plan_attempt(record, rubric_status="needs_revision")
        self.assertEqual(paused["status"], "paused")
        start_plan_attempt(record)
        capped = finish_plan_attempt(record, rubric_status="max_iterations_reached")
        self.assertEqual(capped["status"], "max_iterations_reached")
        start_plan_attempt(record)
        completed = finish_plan_attempt(record, rubric_status="satisfied")
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["completion_source"], "rubric-verified")

    def test_current_plan_persists_and_is_authoritative_resume_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            record = store.new("session-1", Path(directory))
            replace_current_plan(record, artifact())
            record["events"] = [
                {
                    "id": 1,
                    "type": "plan",
                    "created_at": record["created_at"],
                    "plan": {"id": "stale", "title": "Stale"},
                    "status": "superseded",
                }
            ]
            store.save(record)
            loaded = store.read(store.path("session-1"))
        self.assertEqual(current_plan(loaded), current_plan(record))
        resume = build_resume_context(loaded)
        self.assertIn("Authoritative current Plan:", resume)
        self.assertIn("Durable Plan", resume)
        self.assertNotIn("Stale", resume)

    def test_sessions_without_current_schema_fields_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "current schema"):
            normalize_session({"id": "empty", "events": []})

    def test_old_plan_field_names_are_not_accepted(self) -> None:
        value = artifact()
        value["criteria"] = value.pop("success_criteria")
        value.pop("rubric_enabled")
        value["automatic_evaluation"] = True

        record = SessionStore(Path(".")).new(session_id="old", workspace=Path("workspace"))
        record["current_plan"] = value

        with self.assertRaisesRegex(ValueError, "current_plan"):
            normalize_session(record)

    def test_current_plan_values_are_not_coerced(self) -> None:
        invalid = (("key_changes", "Add it."), ("attempts", "0"), ("rubric_iterations", True))
        for field, bad_value in invalid:
            value = artifact()
            value[field] = bad_value
            record = SessionStore(Path(".")).new("strict-plan", Path("workspace"))
            record["current_plan"] = value
            with self.assertRaisesRegex(ValueError, "current_plan"):
                normalize_session(record)

    def test_plan_command_suffix_is_a_normal_message_on_the_persistent_thread(self) -> None:
        session = {"id": "session-1"}
        self.assertEqual(plan_thread_id(session), "session-1:plan")
        self.assertEqual(plan_command_prompt("/plan"), "")
        self.assertEqual(plan_command_prompt("/plan    "), "")
        self.assertEqual(
            plan_command_prompt('/plan   "keep this quoted text"  '),
            '"keep this quoted text"',
        )
        self.assertIsNone(plan_command_prompt("/plan-show"))


class CriteriaFirstPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_prepare_plan_uses_same_success_criteria_service_for_both_settings(self) -> None:
        staged = []
        for rubric_enabled in (False, True):
            fake = SimpleNamespace(
                config={},
                mode={
                    "rubric_enabled": rubric_enabled,
                    "plan_revision": None,
                    "plan_staging": None,
                    "planning_stage": PLANNING_STAGE_PLAN_RESEARCH,
                },
                waiting_finished=lambda: None,
                waiting_started=lambda *args, **kwargs: None,
            )
            with patch(
                "ui.app.SuccessCriteriaService.generate",
                new=AsyncMock(return_value="- The observable result is complete."),
            ) as generate:
                result = await MiraApp.prepare_plan(
                    fake,
                    {
                        "type": "prepare_plan",
                        "objective": "Produce the deliverable.",
                        "context_and_constraints": "Keep it concise.",
                    },
                )
            generate.assert_awaited_once_with(
                "Produce the deliverable.",
                "Keep it concise.",
            )
            self.assertIn("<success_criteria>", result)
            self.assertEqual(fake.mode["planning_stage"], PLANNING_STAGE_PLAN_FINALIZE)
            staged.append(fake.mode["plan_staging"])
        self.assertEqual(staged[0], staged[1])

    async def test_execution_injects_exact_plan_criteria_and_uses_rubric_outcome(self) -> None:
        class Store:
            def save(self, record):
                return None

        value = artifact(rubric=True)
        value["status"] = "active"
        session = {
            "id": "session-1",
            "workspace": ".",
            "turns": 0,
            "events": [],
            "current_plan": value,
        }
        mode = {
            "planning": False,
            "executing_plan": True,
            "rubric_enabled": True,
            "rubric_max_iterations": 4,
        }
        captured = {}

        async def fake_run_turn(**kwargs):
            captured.update(kwargs)
            return TurnResult(rubric_status="satisfied")

        with patch("ui.repl.run_turn", fake_run_turn):
            await run_user_turn(
                agent=SimpleNamespace(),
                plan_agent=SimpleNamespace(),
                renderer=SimpleNamespace(),
                store=Store(),
                session=session,
                mode=mode,
                text="Implement the approved Plan.",
                record_user=False,
            )

        self.assertEqual(
            captured["rubric"],
            "- The requested result is complete and verified.",
        )
        self.assertEqual(captured["rubric_max_iterations"], 4)
        self.assertIn("Title: Durable Plan", captured["text"])
        self.assertIn("Success Criteria:", captured["text"])
        self.assertEqual(session["current_plan"]["status"], "completed")
        self.assertEqual(
            session["current_plan"]["completion_source"],
            "rubric-verified",
        )


if __name__ == "__main__":
    unittest.main()
