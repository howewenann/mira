"""Focused coverage for durable criteria-only Goals."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from runtime.runner import TurnResult
from session.context import build_resume_context, normalize_session
from session.goals import (
    clear_current_goal,
    current_goal,
    finish_goal_attempt,
    goal_artifact,
    replace_current_goal,
    start_goal_attempt,
)
from session.plans import plan_artifact, replace_current_plan
from ui.repl import action_request_text, run_user_turn


def artifact(*, rubric: bool = False, goal_id: str = "goal-1") -> dict:
    return goal_artifact(
        goal_id=goal_id,
        title="Durable Goal",
        objective="Deliver the requested result.",
        success_criteria="- The requested result exists and is verified.",
        rubric_enabled=rubric,
        rubric_iterations=4,
    )


class Store:
    def save(self, record):
        return None


class CurrentGoalTests(unittest.TestCase):
    def test_goal_has_only_title_objective_criteria_and_lifecycle(self) -> None:
        value = artifact()
        self.assertEqual(value["title"], "Durable Goal")
        self.assertEqual(value["status"], "proposed")
        self.assertNotIn("plan", value)
        self.assertNotIn("proposal_id", value)
        self.assertNotIn("original_objective", value)

    def test_agent_declared_and_rubric_verified_completion(self) -> None:
        plain = {"events": [], "current_goal": artifact()}
        start_goal_attempt(plain)
        self.assertEqual(finish_goal_attempt(plain)["completion_source"], "agent-declared")

        graded = {"events": [], "current_goal": artifact(rubric=True)}
        start_goal_attempt(graded)
        self.assertEqual(
            finish_goal_attempt(graded, rubric_status="satisfied")["completion_source"],
            "rubric-verified",
        )

    def test_max_iterations_is_resumable_and_clear_keeps_events(self) -> None:
        value = artifact(rubric=True)
        record = {"events": [{"type": "goal", "goal": value, "status": "proposed"}], "current_goal": value}
        start_goal_attempt(record)
        capped = finish_goal_attempt(record, rubric_status="max_iterations_reached")
        self.assertEqual(capped["status"], "max_iterations_reached")
        self.assertEqual(start_goal_attempt(record)["attempts"], 2)
        self.assertEqual(clear_current_goal(record)["id"], "goal-1")
        self.assertEqual(len(record["events"]), 1)

    def test_plan_and_goal_replacement_enforce_one_current_artifact(self) -> None:
        goal = artifact()
        record = {"events": [{"type": "goal", "goal": goal, "status": "proposed"}]}
        replace_current_goal(record, goal)
        plan = plan_artifact(
            plan_id="plan-1", title="Plan", objective="Do it.",
            context_and_constraints="No constraints.", key_changes=["Do it."],
            test_plan=["Verify it."], assumptions=["None."],
            success_criteria="- It is done.", rubric_enabled=False, rubric_iterations=3,
        )
        replace_current_plan(record, plan)
        self.assertIsNone(record["current_goal"])
        self.assertEqual(record["current_plan"]["id"], "plan-1")
        self.assertEqual(record["events"][0]["status"], "superseded")

    def test_legacy_active_goal_migrates_without_embedded_plan(self) -> None:
        normalized = normalize_session(
            {
                "id": "legacy",
                "events": [],
                "active_goal": {
                    "proposal_id": "proposal-1",
                    "objective": "Build search.",
                    "criteria": "- Search works.",
                    "plan": {"title": "Hidden legacy plan", "summary": ["Do not retain."]},
                    "status": "complete",
                    "last_rubric_status": "satisfied",
                    "rubric_iterations": 5,
                },
            }
        )
        goal = normalized["current_goal"]
        self.assertEqual(goal["title"], "Hidden legacy plan")
        self.assertEqual(goal["status"], "completed")
        self.assertEqual(goal["completion_source"], "rubric-verified")
        self.assertTrue(goal["rubric_enabled"])
        self.assertNotIn("plan", goal)
        self.assertNotIn("active_goal", normalized)

    def test_legacy_cleared_goal_is_not_restored(self) -> None:
        normalized = normalize_session(
            {"id": "legacy", "events": [], "active_goal": {"id": "old", "objective": "Old", "criteria": "- Done", "status": "cleared"}}
        )
        self.assertIsNone(normalized["current_goal"])

    def test_newer_artifact_wins_and_timestamp_ties_prefer_plan(self) -> None:
        goal = artifact()
        goal["updated_at"] = "2026-02-01T00:00:00+00:00"
        plan = plan_artifact(
            plan_id="plan-1", title="Plan", objective="Do it.",
            context_and_constraints="No constraints.", key_changes=["Do it."],
            test_plan=["Verify it."], assumptions=["None."], success_criteria="- Done.",
            rubric_enabled=False, rubric_iterations=3,
        )
        plan["updated_at"] = "2026-01-01T00:00:00+00:00"
        self.assertIsNone(normalize_session({"id": "s", "events": [], "current_plan": plan, "current_goal": goal})["current_plan"])
        plan["updated_at"] = goal["updated_at"]
        self.assertIsNone(normalize_session({"id": "s", "events": [], "current_plan": plan, "current_goal": goal})["current_goal"])

    def test_resume_context_labels_only_current_goal_authoritative(self) -> None:
        goal = artifact()
        record = {"events": [{"id": 1, "type": "goal", "goal": goal, "status": "proposed"}], "current_goal": goal}
        text = build_resume_context(record)
        self.assertIn("Authoritative current Goal", text)
        self.assertIn("Success Criteria", text)
        self.assertNotIn("Plan:", text)

    def test_action_context_contains_no_plan_fields(self) -> None:
        text = action_request_text({}, "Continue.", retained_goal=artifact())
        self.assertIn("<current_goal>", text)
        self.assertIn("Objective:", text)
        self.assertIn("Success Criteria:", text)
        self.assertNotIn("Key Changes", text)
        self.assertNotIn("approved Plan", text)


class GoalExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_explicit_attempt_injects_exact_criteria_and_completes_without_rubric(self) -> None:
        value = artifact()
        value["status"] = "active"
        session = {"id": "session-1", "workspace": ".", "turns": 0, "events": [], "current_goal": value}
        mode = {"planning": False, "executing_goal": True, "executing_plan": False, "rubric_enabled": False, "rubric_max_iterations": 3}
        captured = {}

        async def fake_run_turn(**kwargs):
            captured.update(kwargs)
            return TurnResult()

        with patch("ui.repl.run_turn", fake_run_turn):
            await run_user_turn(
                agent=SimpleNamespace(), plan_agent=SimpleNamespace(), renderer=SimpleNamespace(),
                store=Store(), session=session, mode=mode, text="Implement.", record_user=False,
            )

        self.assertIsNone(captured["rubric"])
        self.assertIn(value["success_criteria"], captured["text"])
        self.assertEqual(session["current_goal"]["status"], "completed")
        self.assertEqual(session["current_goal"]["completion_source"], "agent-declared")


if __name__ == "__main__":
    unittest.main()
