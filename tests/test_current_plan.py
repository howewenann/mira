"""Focused coverage for conversational durable Plans."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from agent.middleware import PlanningStageEnforcementMiddleware
from agent.factory import ACT_SYSTEM_PROMPT
from agent.planning.tool_context import PlanningToolContext
from agent.default_resources.tools.prepare_goal import prepare_goal
from agent.default_resources.tools.prepare_plan import prepare_plan
from agent.default_resources.tools.finalize_plan import finalize_plan
from agent.default_resources.tools.finalize_goal import finalize_goal
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
from session.goals import goal_artifact, replace_current_goal, start_goal_attempt
from session.store import SessionStore
from core.execution.turns import plan_command_prompt, plan_thread_id
from tests.support.turns import run_user_turn
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from core.execution.runner import TurnResult


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


class BindablePlanningModel(FakeMessagesListChatModel):
    """Fake model that records native model passes while accepting tool binding."""

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        object.__setattr__(self, "bound_tool_names", [tool.name for tool in tools])
        object.__setattr__(self, "bound_tool_choice", tool_choice)
        return self


class CurrentPlanTests(unittest.TestCase):
    def test_plan_prompt_is_conversational_general_purpose_and_rubric_independent(self) -> None:
        prompt = plan_system_prompt()
        for outcome in ("DISPLAY_RETAINED", "DISCUSSION", "NEEDS_DECISION", "PLAN_READY"):
            self.assertIn(outcome, prompt)
        self.assertIn("Imperative wording never authorizes execution", prompt)
        self.assertIn("immediately call the applicable show_plan or show_goal tool", prompt)
        self.assertIn("Do not research, reproduce the artifact in prose", prompt)
        self.assertIn("research, analysis, writing, communication, data work", prompt)
        self.assertIn(SHARED_QUESTION_POLICY, prompt)
        self.assertIn(SHARED_QUESTION_POLICY, ACT_SYSTEM_PROMPT)
        self.assertNotIn("SAFE_CONVERSATION", prompt)

    def test_plan_stage_prioritizes_show_then_requires_finalize(self) -> None:
        tools = [
            {"name": "read_file"},
            {"name": "ask_user"},
            {"name": "prepare_goal"},
            {"name": "prepare_plan"},
            {"name": "finalize_plan"},
            {"name": "finalize_goal"},
            {"name": "show_goal"},
            {"name": "show_plan"},
        ]
        middleware = PlanningStageEnforcementMiddleware()
        research = middleware._stage_request(Request(PLANNING_STAGE_PLAN_RESEARCH, tools))
        self.assertEqual(
            [tool["name"] for tool in research.tools],
            ["show_plan", "read_file", "ask_user", "prepare_plan", "show_goal"],
        )
        final = middleware._stage_request(Request(PLANNING_STAGE_PLAN_FINALIZE, tools))
        self.assertEqual([tool["name"] for tool in final.tools], ["finalize_plan"])
        self.assertEqual(final.tool_choice, "required")
        goal_research = middleware._stage_request(Request(PLANNING_STAGE_GOAL_RESEARCH, tools))
        self.assertEqual(
            [tool["name"] for tool in goal_research.tools],
            ["show_goal", "read_file", "ask_user", "prepare_goal", "show_plan"],
        )
        goal_final = middleware._stage_request(Request(PLANNING_STAGE_GOAL_FINALIZE, tools))
        self.assertEqual([tool["name"] for tool in goal_final.tools], ["finalize_goal"])
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
    def test_finalizers_are_native_terminal_tools(self) -> None:
        self.assertTrue(finalize_plan.return_direct)
        self.assertTrue(finalize_goal.return_direct)
        self.assertNotIn("runtime", finalize_plan.tool_call_schema.model_fields)
        self.assertNotIn("runtime", finalize_goal.tool_call_schema.model_fields)

    async def test_prepare_then_finalizer_uses_one_post_prepare_inference_and_ends_on_resume(self) -> None:
        criteria = "- Exact generated criterion."
        service = SimpleNamespace(
            generate=AsyncMock(return_value=criteria),
            revise=AsyncMock(),
        )
        model = BindablePlanningModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "prepare_plan",
                            "args": {
                                "objective": "Ship the result.",
                                "context_and_constraints": "Keep it focused.",
                            },
                            "id": "call-prepare",
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "finalize_plan",
                            "args": {
                                "title": "Ship Result",
                                "key_changes": ["Make the change."],
                                "test_plan": ["Run focused checks."],
                                "assumptions": ["None."],
                            },
                            "id": "call-finalize",
                        }
                    ],
                ),
                AIMessage(content="unexpected post-finalizer inference"),
            ]
        )
        graph = create_agent(
            model=model,
            tools=[prepare_plan, finalize_plan],
            middleware=[PlanningStageEnforcementMiddleware()],
            context_schema=PlanningToolContext,
            checkpointer=InMemorySaver(),
        )
        config = {"configurable": {"thread_id": "criteria-first-plan"}}
        context = PlanningToolContext(service)

        interrupted = await graph.ainvoke(
            {
                "messages": [HumanMessage(content="Create the Plan.")],
                "planning_stage": PLANNING_STAGE_PLAN_RESEARCH,
                "planning_authoritative_request": "Ship the result.",
            },
            config=config,
            context=context,
        )

        self.assertEqual(len(interrupted["__interrupt__"]), 1)
        review = interrupted["__interrupt__"][0].value
        self.assertEqual(review["success_criteria"], criteria)
        self.assertEqual(review["objective"], "Ship the result.")
        self.assertEqual(model.bound_tool_names, ["finalize_plan"])
        self.assertEqual(model.bound_tool_choice, "required")
        service.generate.assert_awaited_once()

        completed = await graph.ainvoke(
            Command(resume={"action": "close"}),
            config=config,
            context=context,
        )

        self.assertNotIn("unexpected post-finalizer inference", str(completed["messages"]))
        self.assertEqual(model.i, 2)

    async def test_goal_prepare_then_finalizer_has_one_finalization_inference(self) -> None:
        criteria = "- Exact Goal criterion."
        service = SimpleNamespace(
            generate=AsyncMock(return_value=criteria),
            revise=AsyncMock(),
        )
        model = BindablePlanningModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "prepare_goal",
                            "args": {
                                "objective": "Deliver the outcome.",
                                "context_and_constraints": "Keep it focused.",
                                "research_evidence": "The existing path is known.",
                            },
                            "id": "call-prepare-goal",
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "finalize_goal",
                            "args": {"title": "Deliver Outcome"},
                            "id": "call-finalize-goal",
                        }
                    ],
                ),
                AIMessage(content="unexpected post-finalizer inference"),
            ]
        )
        graph = create_agent(
            model=model,
            tools=[prepare_goal, finalize_goal],
            middleware=[PlanningStageEnforcementMiddleware()],
            context_schema=PlanningToolContext,
            checkpointer=InMemorySaver(),
        )
        config = {"configurable": {"thread_id": "criteria-first-goal"}}
        context = PlanningToolContext(service)

        interrupted = await graph.ainvoke(
            {
                "messages": [HumanMessage(content="Create the Goal.")],
                "planning_stage": PLANNING_STAGE_GOAL_RESEARCH,
                "planning_authoritative_request": "Deliver the outcome.",
            },
            config=config,
            context=context,
        )

        review = interrupted["__interrupt__"][0].value
        self.assertEqual(review["success_criteria"], criteria)
        self.assertEqual(review["objective"], "Deliver the outcome.")
        self.assertEqual(model.bound_tool_names, ["finalize_goal"])
        self.assertEqual(model.bound_tool_choice, "required")
        service.generate.assert_awaited_once()

        completed = await graph.ainvoke(
            Command(resume={"action": "close"}),
            config=config,
            context=context,
        )

        self.assertNotIn("unexpected post-finalizer inference", str(completed["messages"]))
        self.assertEqual(model.i, 2)

    async def test_prepare_plan_uses_same_success_criteria_service_for_both_settings(self) -> None:
        staged = []
        for rubric_enabled in (False, True):
            service = SimpleNamespace(
                generate=AsyncMock(return_value="- The observable result is complete."),
                revise=AsyncMock(),
            )
            runtime = SimpleNamespace(
                context=PlanningToolContext(service),
                state={"planning_authoritative_request": "Produce the deliverable."},
                tool_call_id="call-prepare",
            )
            result = await prepare_plan.coroutine(
                "Produce the deliverable.",
                runtime,
                "Keep it concise.",
            )
            service.generate.assert_awaited_once_with(
                "Produce the deliverable.",
                "Keep it concise.",
                authoritative_request="Produce the deliverable.",
            )
            self.assertIsInstance(result, Command)
            self.assertEqual(result.update["planning_stage"], PLANNING_STAGE_PLAN_FINALIZE)
            self.assertIn("<success_criteria>", result.update["messages"][0].content)
            staged.append(result.update["planning_success_criteria"])
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

        with patch("tests.support.turns.run_turn", fake_run_turn):
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


class SingleTurnWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_plan_revise_implement_and_act_share_one_outer_turn(self) -> None:
        class Store:
            def save(self, record):
                return None

        session = {
            "id": "session-1",
            "workspace": ".",
            "turns": 0,
            "events": [],
            "current_plan": None,
            "current_goal": None,
        }
        mode = {
            "planning": True,
            "planning_stage": PLANNING_STAGE_PLAN_RESEARCH,
            "plan_thread_id": "session-1:plan",
            "rubric_enabled": True,
            "rubric_max_iterations": 4,
            "executing_plan": False,
            "executing_goal": False,
        }
        first = artifact(rubric=True)
        first["id"] = "plan-b"
        final = artifact(rubric=True)
        final["id"] = "plan-c"
        final["success_criteria"] = "- Exact revised criterion.\n- Focused checks pass."
        calls: list[dict] = []

        async def fake_run_turn(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return TurnResult(
                    formal_review={
                        "action": "revise",
                        "artifact": first,
                        "feedback": "Change the completion conditions.",
                    }
                )
            if len(calls) == 2:
                replace_current_plan(session, final)
                start_plan_attempt(session)
                mode.update(
                    planning=False,
                    executing_plan=True,
                    current_plan=session["current_plan"],
                )
                return TurnResult(
                    formal_review={"action": "implement", "artifact": final}
                )
            return TurnResult(rubric_status="satisfied")

        with patch("tests.support.turns.run_turn", fake_run_turn):
            await run_user_turn(
                agent=SimpleNamespace(),
                plan_agent=SimpleNamespace(),
                renderer=SimpleNamespace(),
                store=Store(),
                session=session,
                mode=mode,
                text="Build the requested feature.",
            )

        self.assertEqual(len(calls), 3)
        self.assertEqual([call["planning_stage"] for call in calls[:2]], ["plan_research"] * 2)
        self.assertIsNone(calls[2]["planning_stage"])
        self.assertEqual(calls[2]["rubric"], final["success_criteria"])
        self.assertIn(final["success_criteria"], calls[2]["text"])
        self.assertEqual(session["turns"], 1)
        self.assertEqual(len([event for event in session["events"] if event.get("type") == "user"]), 1)
        self.assertEqual(session["current_plan"]["status"], "completed")

    async def test_close_ends_same_outer_turn_without_act(self) -> None:
        class Store:
            def save(self, record):
                return None

        session = {
            "id": "session-close",
            "workspace": ".",
            "turns": 0,
            "events": [],
            "current_plan": None,
            "current_goal": None,
        }
        mode = {
            "planning": True,
            "planning_stage": PLANNING_STAGE_PLAN_RESEARCH,
            "plan_thread_id": "session-close:plan",
            "rubric_enabled": False,
            "rubric_max_iterations": 3,
            "executing_plan": False,
            "executing_goal": False,
        }
        accepted = artifact()
        calls: list[dict] = []

        async def fake_run_turn(**kwargs):
            calls.append(kwargs)
            replace_current_plan(session, accepted)
            mode["current_plan"] = session["current_plan"]
            return TurnResult(formal_review={"action": "close", "artifact": accepted})

        with patch("tests.support.turns.run_turn", fake_run_turn):
            await run_user_turn(
                agent=SimpleNamespace(),
                plan_agent=SimpleNamespace(),
                renderer=SimpleNamespace(),
                store=Store(),
                session=session,
                mode=mode,
                text="Create a Plan, then retain it.",
            )

        self.assertEqual(len(calls), 1)
        self.assertEqual(session["current_plan"]["id"], accepted["id"])
        self.assertEqual(session["turns"], 1)

    async def test_revise_then_clear_discards_proposals_and_preserves_previous_plan(self) -> None:
        class Store:
            def save(self, record):
                return None

        previous = artifact()
        previous["id"] = "plan-a"
        draft = artifact()
        draft["id"] = "plan-b"
        revised = artifact()
        revised["id"] = "plan-c"
        session = {
            "id": "session-clear",
            "workspace": ".",
            "turns": 0,
            "events": [],
            "current_plan": previous,
            "current_goal": None,
        }
        mode = {
            "planning": True,
            "planning_stage": PLANNING_STAGE_PLAN_RESEARCH,
            "plan_thread_id": "session-clear:plan",
            "rubric_enabled": False,
            "rubric_max_iterations": 3,
            "executing_plan": False,
            "executing_goal": False,
        }
        calls: list[dict] = []

        async def fake_run_turn(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return TurnResult(
                    formal_review={
                        "action": "revise",
                        "artifact": draft,
                        "feedback": "Tighten the checks.",
                    }
                )
            return TurnResult(formal_review={"action": "clear", "artifact": revised})

        with patch("tests.support.turns.run_turn", fake_run_turn):
            await run_user_turn(
                agent=SimpleNamespace(),
                plan_agent=SimpleNamespace(),
                renderer=SimpleNamespace(),
                store=Store(),
                session=session,
                mode=mode,
                text="Propose a replacement Plan.",
            )

        self.assertEqual(len(calls), 2)
        self.assertEqual(session["current_plan"]["id"], "plan-a")
        self.assertEqual(session["current_plan"]["status"], "proposed")
        self.assertEqual(session["turns"], 1)

    async def test_clear_without_previous_artifact_ends_without_act(self) -> None:
        class Store:
            def save(self, record):
                return None

        session = {
            "id": "session-empty-clear",
            "workspace": ".",
            "turns": 0,
            "events": [],
            "current_plan": None,
            "current_goal": None,
        }
        mode = {
            "planning": True,
            "planning_stage": PLANNING_STAGE_PLAN_RESEARCH,
            "plan_thread_id": "session-empty-clear:plan",
            "rubric_enabled": False,
            "rubric_max_iterations": 3,
            "executing_plan": False,
            "executing_goal": False,
        }
        calls: list[dict] = []

        async def fake_run_turn(**kwargs):
            calls.append(kwargs)
            return TurnResult(
                formal_review={"action": "clear", "artifact": artifact()}
            )

        with patch("tests.support.turns.run_turn", fake_run_turn):
            await run_user_turn(
                agent=SimpleNamespace(),
                plan_agent=SimpleNamespace(),
                renderer=SimpleNamespace(),
                store=Store(),
                session=session,
                mode=mode,
                text="Propose and discard a Plan.",
            )

        self.assertEqual(len(calls), 1)
        self.assertIsNone(session["current_plan"])
        self.assertIsNone(session["current_goal"])
        self.assertEqual(session["turns"], 1)

    async def test_goal_multiple_revisions_and_implement_share_one_outer_turn(self) -> None:
        class Store:
            def save(self, record):
                return None

        session = {
            "id": "session-goal",
            "workspace": ".",
            "turns": 0,
            "events": [],
            "current_plan": None,
            "current_goal": None,
        }
        mode = {
            "planning": False,
            "planning_stage": PLANNING_STAGE_GOAL_RESEARCH,
            "plan_thread_id": "session-goal:plan",
            "plan_runs": 1,
            "goal_staging": {
                "authoritative_objective": "Deliver the outcome.",
                "thread_id": "session-goal:plan:1",
            },
            "rubric_enabled": True,
            "rubric_max_iterations": 3,
            "executing_plan": False,
            "executing_goal": False,
        }
        drafts = [
            goal_artifact(
                goal_id=f"goal-{index}",
                title=f"Goal {index}",
                objective="Deliver the outcome.",
                success_criteria=f"- Exact criteria {index}.",
                rubric_enabled=True,
                rubric_iterations=3,
            )
            for index in range(1, 4)
        ]
        calls: list[dict] = []

        async def fake_run_turn(**kwargs):
            calls.append(kwargs)
            if len(calls) < 3:
                return TurnResult(
                    formal_review={
                        "action": "revise",
                        "artifact": drafts[len(calls) - 1],
                        "feedback": f"Revision {len(calls)}.",
                    }
                )
            if len(calls) == 3:
                replace_current_goal(session, drafts[2])
                start_goal_attempt(session)
                mode.update(
                    executing_goal=True,
                    current_goal=session["current_goal"],
                )
                return TurnResult(
                    formal_review={"action": "implement", "artifact": drafts[2]}
                )
            return TurnResult(rubric_status="satisfied")

        with patch("tests.support.turns.run_turn", fake_run_turn):
            await run_user_turn(
                agent=SimpleNamespace(),
                plan_agent=SimpleNamespace(),
                renderer=SimpleNamespace(),
                store=Store(),
                session=session,
                mode=mode,
                text="Deliver the outcome.",
            )

        self.assertEqual(len(calls), 4)
        self.assertEqual(calls[3]["rubric"], drafts[2]["success_criteria"])
        self.assertEqual(session["turns"], 1)
        self.assertEqual(session["current_goal"]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
