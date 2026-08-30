"""Focused coverage for durable criteria-only Goals."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from core.execution.runner import TurnResult
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
from session.store import SessionStore
from core.execution.turns import action_request_text
from tests.support.turns import run_user_turn
from agent.default_resources.tools.prepare_goal import prepare_goal
from agent.planning.tool_context import PlanningToolContext


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

    def test_polished_objective_round_trips_without_authoritative_request_field(self) -> None:
        polished = "Create an approximately 20-word story and save it as story.md."
        value = goal_artifact(
            goal_id="goal-polished",
            title="Short Story",
            objective=polished,
            success_criteria="- story.md contains the requested story.",
            rubric_enabled=False,
            rubric_iterations=3,
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store = SessionStore(root / "sessions")
            record = store.new(session_id="polished-goal", workspace=root)
            record["current_goal"] = value
            store.save(record)

            loaded = store.load("polished-goal", resume=False, workspace=root)

        self.assertEqual(loaded["current_goal"]["objective"], polished)
        self.assertNotIn("authoritative_request", loaded["current_goal"])
        self.assertNotIn("authoritative_objective", loaded["current_goal"])

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

    def test_active_goal_is_not_migrated(self) -> None:
        with self.assertRaisesRegex(ValueError, "current schema"):
            normalize_session({"id": "old-session", "events": [], "active_goal": {}})

    def test_old_goal_fields_and_values_are_not_coerced(self) -> None:
        for field, invalid in (("success_criteria", None), ("rubric_enabled", 1)):
            value = artifact()
            if invalid is None:
                value["criteria"] = value.pop(field)
            else:
                value[field] = invalid
            record = SessionStore(Path(".")).new("old", Path("workspace"))
            record["current_goal"] = value
            with self.assertRaisesRegex(ValueError, "current_goal"):
                normalize_session(record)

    def test_session_with_both_current_artifacts_is_rejected(self) -> None:
        goal = artifact()
        plan = plan_artifact(
            plan_id="plan-1", title="Plan", objective="Do it.",
            context_and_constraints="No constraints.", key_changes=["Do it."],
            test_plan=["Verify it."], assumptions=["None."], success_criteria="- Done.",
            rubric_enabled=False, rubric_iterations=3,
        )
        record = SessionStore(Path(".")).new(session_id="s", workspace=Path("workspace"))
        record["current_plan"] = plan
        record["current_goal"] = goal
        with self.assertRaisesRegex(ValueError, "cannot contain both"):
            normalize_session(record)

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
    async def test_prepare_goal_returns_normally_with_exact_generated_criteria(self) -> None:
        service = SimpleNamespace(
            generate=AsyncMock(return_value="- Exact generated criterion."),
            revise=AsyncMock(),
        )
        runtime = SimpleNamespace(
            context=PlanningToolContext(service),
            state={"planning_authoritative_request": "Deliver the exact outcome."},
            tool_call_id="call-goal",
        )

        result = await prepare_goal.coroutine(
            "Deliver the exact outcome.",
            runtime,
            "Keep it focused.",
            "Existing evidence.",
        )

        self.assertEqual(result.update["planning_stage"], "goal_finalize")
        self.assertEqual(
            result.update["planning_success_criteria"],
            "- Exact generated criterion.",
        )
        self.assertEqual(
            result.update["messages"][0].content.count("- Exact generated criterion."),
            1,
        )
        service.generate.assert_awaited_once()

    async def test_explicit_attempt_injects_exact_criteria_and_completes_without_rubric(self) -> None:
        value = artifact()
        value["status"] = "active"
        session = {"id": "session-1", "workspace": ".", "turns": 0, "events": [], "current_goal": value}
        mode = {"planning": False, "executing_goal": True, "executing_plan": False, "rubric_enabled": False, "rubric_max_iterations": 3}
        captured = {}

        async def fake_run_turn(**kwargs):
            captured.update(kwargs)
            return TurnResult()

        with patch("tests.support.turns.run_turn", fake_run_turn):
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
