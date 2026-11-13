"""Application-level handoff checks using native agents and SQLite checkpoints."""

from __future__ import annotations

import asyncio
import gc
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.graph import END

from agent.planning.policy import PLAN_BEHAVIOR_POLICY, PLAN_DISABLED_TOOLS
from agent.middleware.mode_transition import MODE_NOTICES, MODE_TRANSITION_SOURCE
from core.execution.compaction import commit_forced_compaction
from core.execution.handoff import effective_conversation
from core.execution.runner import TurnResult
from core.interface.requests import ApprovalRequest, ArtifactReviewRequest
from tests import test_sqlite_sessions as sqlite_tests
from tests.test_workflows import WorkflowModel
from ui.textual.commands.dispatcher import handle_command


def config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def content(messages: list) -> list[str]:
    return [str(message.content) for message in messages]


class ReviewFrontend:
    def __init__(self, *actions: str) -> None:
        self.actions = list(actions)
        self.requests: list[ArtifactReviewRequest] = []
        self.events: list[object] = []

    def emit(self, event: object) -> None:
        self.events.append(event)

    async def request(self, request: object) -> dict:
        if not isinstance(request, ArtifactReviewRequest):
            raise AssertionError(f"unexpected frontend request: {request}")
        self.requests.append(request)
        action = self.actions.pop(0)
        return {"action": action, **({"feedback": "Make the revised Goal clearer"} if action == "revise" else {})}


class DisconnectedReviewFrontend(ReviewFrontend):
    def __init__(self) -> None:
        super().__init__()

    async def request(self, request: object) -> dict:
        if not isinstance(request, ArtifactReviewRequest):
            raise AssertionError(f"unexpected frontend request: {request}")
        self.requests.append(request)
        raise RuntimeError("frontend disconnected during review")


class ScriptedModel(WorkflowModel):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        del stop, run_manager, kwargs
        names = self._bound_tool_names
        last = next((str(message.content) for message in reversed(messages) if isinstance(message, HumanMessage)), "")
        if "finalize_goal" in names and "prepare_goal" not in names:
            name, args = "finalize_goal", {"title": "Prepared Goal"}
        elif "finalize_plan" in names and "prepare_plan" not in names:
            name, args = "finalize_plan", {
                "title": "Prepared Plan", "key_changes": ["Make change"],
                "test_plan": ["Verify change"], "assumptions": ["None"],
            }
        elif "prepare_goal" in names and ("Goal request" in last or "Revise this complete Goal" in last):
            name, args = "prepare_goal", {"objective": "Complete the requested Goal"}
        elif "prepare_plan" in names and "Build a formal Plan" in last:
            name, args = "prepare_plan", {"objective": "Build a formal Plan"}
        else:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="- Verified result"))])
        message = AIMessage(content="", tool_calls=[{
            "name": name, "args": args, "id": f"call-{uuid4().hex}", "type": "tool_call",
        }])
        return ChatResult(generations=[ChatGeneration(message=message)])


class ConversationHandoffTests(unittest.IsolatedAsyncioTestCase):
    start = sqlite_tests.SqliteSessionTests.start
    stop = sqlite_tests.SqliteSessionTests.stop

    async def asyncTearDown(self) -> None:
        # QuickJS tool middleware closes its worker synchronously at GC time.
        # Collect while the test event loop and worker are still available.
        gc.collect()

    async def test_bidirectional_inheritance_and_independent_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = await self.start(Path(directory), todos=True)
            try:
                session = await app.open_session("handoff")
                await session.prompt("act one")
                await session.agent.aupdate_state(config(session.id), {"todos": [{
                    "content": "Act task", "status": "pending", "id": "act-task",
                }]})
                act_updated = await session.agent.aget_state(config(session.id))
                if act_updated.next:
                    await session.agent.aupdate_state(act_updated.config, None, as_node=END)
                act_before = await session.agent.aget_state(config(session.id))
                await session.set_mode("plan")
                plan_before = await session.plan_agent.aget_state(config("handoff:plan"))
                self.assertEqual(content(plan_before.values["messages"]), content(act_before.values["messages"]))
                checkpoint_id = plan_before.config
                await session.set_mode("plan")
                self.assertEqual((await session.plan_agent.aget_state(config("handoff:plan"))).config, checkpoint_id)
                self.assertNotIn("write_file", {item["name"] for item in session.snapshot().tools})
                await session.prompt("plan two")
                await session.plan_agent.aupdate_state(config("handoff:plan"), {"todos": [{
                    "content": "Plan task", "status": "pending", "id": "plan-task",
                }]})
                plan_updated = await session.plan_agent.aget_state(config("handoff:plan"))
                if plan_updated.next:
                    await session.plan_agent.aupdate_state(plan_updated.config, None, as_node=END)
                await session.set_mode("act")
                act_after = await session.agent.aget_state(config(session.id))
                plan_after = await session.plan_agent.aget_state(config("handoff:plan"))
                self.assertEqual(content(act_after.values["messages"]), content(plan_after.values["messages"]))
                self.assertEqual(act_after.values["todos"][0]["content"], "Act task")
                self.assertEqual(plan_after.values["todos"][0]["content"], "Plan task")
                human = [str(message.content) for message in act_after.values["messages"] if isinstance(message, HumanMessage)]
                self.assertEqual(sum("act one" in text for text in human), 1)
                self.assertEqual(sum("plan two" in text for text in human), 1)
                self.assertIn("write_file", {item["name"] for item in session.snapshot().tools})
                self.assertIsNone(session.recovery)
            finally:
                await self.stop(app)

    async def test_tui_mode_commands_use_session_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = await self.start(Path(directory))
            try:
                session = await app.open_session("commands")
                await session.prompt("Act command marker")

                class Renderer:
                    core_session = session

                    def system_message(self, text: str, *, kind: str = "system") -> None:
                        pass

                renderer = Renderer()
                self.assertTrue(await handle_command("/plan", renderer, session.record, "test", session.mode))
                plan = await session.plan_agent.aget_state(config("commands:plan"))
                self.assertTrue(any("Act command marker" in text for text in content(plan.values["messages"])))
                await session.prompt("Plan command marker")
                self.assertTrue(await handle_command("/act", renderer, session.record, "test", session.mode))
                act = await session.agent.aget_state(config(session.id))
                self.assertTrue(any("Plan command marker" in text for text in content(act.values["messages"])))
            finally:
                await self.stop(app)

    async def test_plan_act_roundtrip_keeps_policy_out_of_history_and_allows_act_writes(self) -> None:
        requests: list[tuple[list, set[str]]] = []
        discussion = "Discuss note.txt containing 'roundtrip context'."

        class WriteModel(WorkflowModel):
            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                names = set(self._bound_tool_names)
                requests.append((list(messages), names))
                if isinstance(messages[-1], ToolMessage):
                    message = AIMessage(content="Written.\nRESPONSE_STATUS: COMPLETE")
                elif "write_file" in names:
                    # Resolve the requested content from inherited Plan context.
                    original = next(m.content for m in messages if isinstance(m, HumanMessage)
                                    and m.name != MODE_TRANSITION_SOURCE)
                    message = AIMessage(content="", tool_calls=[{
                        "name": "write_file", "id": "write-note", "type": "tool_call",
                        "args": {"file_path": "/note.txt", "content": original.split("'")[1]},
                    }])
                else:
                    message = AIMessage(content="Read-only discussion; no files changed.\nRESPONSE_STATUS: COMPLETE")
                return ChatResult(generations=[ChatGeneration(message=message)])

        class ApprovingFrontend(ReviewFrontend):
            async def request(self, request):
                if isinstance(request, ApprovalRequest):
                    return [{"type": "approve"}]
                raise AssertionError(f"unexpected frontend request: {request}")

        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            app = await self.start(workspace, model=WriteModel(), frontend=ApprovingFrontend())
            try:
                session = await app.open_session("roundtrip")
                await session.set_mode("plan")
                await session.prompt(discussion)
                plan_messages, plan_tools = requests[-1]
                self.assertTrue(set(PLAN_DISABLED_TOOLS).isdisjoint(plan_tools))
                self.assertEqual([m.content for m in plan_messages if isinstance(m, HumanMessage)],
                                 [MODE_NOTICES["plan"], discussion])
                self.assertIn(PLAN_BEHAVIOR_POLICY, next(m.text for m in plan_messages if isinstance(m, SystemMessage)))
                self.assertFalse((workspace / "note.txt").exists())

                source = await session.plan_agent.aget_state(config("roundtrip:plan"))
                await session.set_mode("act")
                inherited = await session.agent.aget_state(config(session.id))
                self.assertEqual(inherited.values["messages"], source.values["messages"])
                self.assertEqual([m.content for m in inherited.values["messages"] if isinstance(m, HumanMessage)], [discussion])
                self.assertTrue(all(PLAN_BEHAVIOR_POLICY not in m.text for m in inherited.values["messages"]))
                result = await session.prompt("Write note.txt with the content we discussed.")
                self.assertIn("write_file", result.tool_calls)
                self.assertEqual((workspace / "note.txt").read_text(), "roundtrip context")
                act_messages, act_tools = requests[-1]
                self.assertIn("write_file", act_tools)
                self.assertTrue(all(PLAN_BEHAVIOR_POLICY not in m.text for m in act_messages))

                await session.set_mode("plan")
                result = await session.prompt("Write blocked.txt containing 'blocked'.")
                restored_messages, restored_tools = requests[-1]
                self.assertTrue(set(PLAN_DISABLED_TOOLS).isdisjoint(restored_tools))
                self.assertIn(PLAN_BEHAVIOR_POLICY, next(m.text for m in restored_messages if isinstance(m, SystemMessage)))
                self.assertNotIn("write_file", result.tool_calls)
                self.assertFalse((workspace / "blocked.txt").exists())
                self.assertEqual((workspace / "note.txt").read_text(), "roundtrip context")

                before = await session.plan_agent.aget_state(config("roundtrip:plan"))
                calls_before = len(requests)
                for _ in range(4):
                    await session.set_mode("act")
                    act = await session.agent.aget_state(config(session.id))
                    self.assertEqual(act.values["messages"], before.values["messages"])
                    await session.set_mode("plan")
                    plan = await session.plan_agent.aget_state(config("roundtrip:plan"))
                    self.assertEqual(plan.values["messages"], before.values["messages"])
                self.assertEqual(len(requests), calls_before)
            finally:
                await self.stop(app)

    async def test_forced_compaction_handoff_and_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            model = ScriptedModel()
            model.profile = {"max_input_tokens": 5000}
            app = await self.start(workspace, model=model)
            try:
                session = await app.open_session("compact")
                for index in range(12):
                    await session.prompt(f"history {index}")
                model.profile = {"max_input_tokens": 100}
                snapshot = await session.agent.aget_state(config(session.id))
                plan = await session.agent.mira_compaction.aplan_forced_compaction_update(snapshot.values)
                result = await commit_forced_compaction(session.agent, session.id, snapshot, plan)
                self.assertTrue(result.compacted)
                source = await session.agent.aget_state(config(session.id))
                expected = content(effective_conversation(session.agent, source.values))
                await session.set_mode("plan")
                inherited = await session.plan_agent.aget_state(config("compact:plan"))
                self.assertEqual(content(inherited.values["messages"]), expected)
                self.assertIsNone(inherited.values.get("_summarization_event"))
                self.assertIsNone(inherited.values.get("_summarization_session_id"))
                archive = await session.plan_agent.mira_backend.adownload_files([result.file_path])
                self.assertIsNone(archive[0].error)
            finally:
                await self.stop(app)
            reopened_app = await self.start(workspace, model=model)
            try:
                reopened = await reopened_app.open_session("compact")
                self.assertTrue(reopened.mode["planning"])
                await reopened.set_mode("act")
                state = await reopened.agent.aget_state(config(reopened.id))
                self.assertEqual(content(effective_conversation(reopened.agent, state.values)), expected)
            finally:
                await self.stop(reopened_app)

    async def test_plan_compaction_handoff_to_act(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = ScriptedModel()
            model.profile = {"max_input_tokens": 100000}
            app = await self.start(Path(directory), model=model)
            try:
                session = await app.open_session("plan-compact")
                await session.prompt("Act marker")
                await session.set_mode("plan")
                for index in range(12):
                    await session.prompt(f"plan history {index}")
                model.profile = {"max_input_tokens": 100}
                snapshot = await session.plan_agent.aget_state(config("plan-compact:plan"))
                planned = await session.plan_agent.mira_compaction.aplan_forced_compaction_update(snapshot.values)
                result = await commit_forced_compaction(session.plan_agent, "plan-compact:plan", snapshot, planned)
                self.assertTrue(result.compacted)
                source = await session.plan_agent.aget_state(config("plan-compact:plan"))
                expected = content(effective_conversation(session.plan_agent, source.values))
                await session.set_mode("act")
                target = await session.agent.aget_state(config(session.id))
                self.assertEqual(content(effective_conversation(session.agent, target.values)), expected)
                self.assertIsNone(target.values.get("_summarization_event"))
                archive = await session.agent.mira_backend.adownload_files([result.file_path])
                self.assertIsNone(archive[0].error)
            finally:
                await self.stop(app)

    async def test_automatic_compaction_is_inherited_by_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = ScriptedModel()
            model.profile = {"max_input_tokens": 10000}
            app = await self.start(Path(directory), model=model)
            try:
                session = await app.open_session("auto-compact")
                state = None
                for index in range(12):
                    await session.prompt(f"history {index}: " + "context " * 400)
                    state = await session.agent.aget_state(config(session.id))
                    if state.values.get("_summarization_event"):
                        break
                self.assertIsNotNone(state.values.get("_summarization_event"))
                expected = content(effective_conversation(session.agent, state.values))
                await session.set_mode("plan")
                plan_state = await session.plan_agent.aget_state(config("auto-compact:plan"))
                self.assertEqual(content(plan_state.values["messages"]), expected)
                self.assertIsNone(plan_state.values.get("_summarization_event"))
                for index in range(8):
                    await session.prompt(f"plan continuation {index}: " + "context " * 400)
                    plan_state = await session.plan_agent.aget_state(config("auto-compact:plan"))
                    if plan_state.values.get("_summarization_event"):
                        break
                self.assertIsNotNone(plan_state.values.get("_summarization_event"))
                expected = content(effective_conversation(session.plan_agent, plan_state.values))
                await session.set_mode("act")
                act_state = await session.agent.aget_state(config(session.id))
                self.assertEqual(content(effective_conversation(session.agent, act_state.values)), expected)
                archive_path = plan_state.values["_summarization_event"]["file_path"]
                archive = await session.agent.mira_backend.adownload_files([archive_path])
                self.assertIsNone(archive[0].error)
            finally:
                await self.stop(app)

    async def test_active_turn_blocks_all_mode_entry_points(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = await self.start(Path(directory))
            entered, release = asyncio.Event(), asyncio.Event()

            async def blocked_turn(**kwargs):
                entered.set()
                await release.wait()
                return TurnResult(final_text="done")

            try:
                session = await app.open_session("busy")
                with patch("core.execution.turns.run_turn", blocked_turn):
                    task = asyncio.create_task(session.prompt("working"))
                    await entered.wait()
                    for action in (
                        session.set_mode("plan"), session.begin_goal("Goal"),
                        session.begin_artifact_revision("goal", {"objective": "Goal"}, "revise"),
                        session.start_artifact("plan"), session.clear_artifact("plan"),
                        session.prompt("second"),
                    ):
                        with self.assertRaisesRegex(RuntimeError, "active"):
                            await action
                    self.assertFalse(session.mode["planning"])
                    release.set()
                    await task
            finally:
                release.set()
                await self.stop(app)

    async def test_mode_switch_is_rejected_during_native_compaction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = ScriptedModel()
            model.profile = {"max_input_tokens": 10000}
            app = await self.start(Path(directory), model=model)
            entered, release = asyncio.Event(), asyncio.Event()
            try:
                session = await app.open_session("compacting")
                summarization = session.agent.mira_summarization
                original = summarization._acreate_summary

                async def blocked_summary(messages):
                    entered.set()
                    await release.wait()
                    return await original(messages)

                with patch.object(summarization, "_acreate_summary", blocked_summary):
                    for index in range(12):
                        turn = asyncio.create_task(session.prompt(f"history {index}: " + "context " * 400))
                        started = asyncio.create_task(entered.wait())
                        done, pending = await asyncio.wait({turn, started}, return_when=asyncio.FIRST_COMPLETED)
                        for pending_task in pending:
                            if pending_task is started:
                                pending_task.cancel()
                        if started in pending:
                            await asyncio.gather(started, return_exceptions=True)
                        if started in done and entered.is_set():
                            with self.assertRaisesRegex(RuntimeError, "active operation"):
                                await session.set_mode("plan")
                            with self.assertRaisesRegex(RuntimeError, "active operation"):
                                await session.begin_goal("new Goal")
                            release.set()
                            await turn
                            break
                        await turn
                    else:
                        self.fail("automatic compaction did not start")
                self.assertFalse(session.mode["planning"])
            finally:
                release.set()
                await self.stop(app)

    async def test_plan_turn_blocks_mode_switch_and_goal_preparation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = await self.start(Path(directory))
            entered, release = asyncio.Event(), asyncio.Event()

            async def blocked_turn(**kwargs):
                entered.set()
                await release.wait()
                return TurnResult(final_text="done")

            try:
                session = await app.open_session("plan-busy")
                await session.set_mode("plan")
                with patch("core.execution.turns.run_turn", blocked_turn):
                    turn = asyncio.create_task(session.prompt("planning"))
                    await entered.wait()
                    with self.assertRaisesRegex(RuntimeError, "active operation"):
                        await session.set_mode("act")
                    with self.assertRaisesRegex(RuntimeError, "active operation"):
                        await session.begin_goal("new Goal")
                    self.assertTrue(session.mode["planning"])
                    release.set()
                    await turn
            finally:
                release.set()
                await self.stop(app)

    async def test_independent_sessions_prompt_concurrently(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = await self.start(Path(directory))
            try:
                first = await app.open_session("parallel-a")
                second = await app.open_session("parallel-b")
                await asyncio.gather(first.prompt("first only"), second.prompt("second only"))
                first_state = await first.agent.aget_state(config(first.id))
                second_state = await second.agent.aget_state(config(second.id))
                self.assertTrue(any("first only" in text for text in content(first_state.values["messages"])))
                self.assertFalse(any("second only" in text for text in content(first_state.values["messages"])))
                self.assertTrue(any("second only" in text for text in content(second_state.values["messages"])))
                self.assertNotEqual(app.store.database_path(first.id), app.store.database_path(second.id))
            finally:
                await self.stop(app)

    async def test_pending_destination_refuses_handoff_without_changing_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = await self.start(Path(directory))
            try:
                session = await app.open_session("pending-target")
                await session.prompt("safe origin")
                source = await session.agent.aget_state(config(session.id))
                await session.plan_agent.aupdate_state(
                    config("pending-target:plan"), {"messages": [HumanMessage(content="pending plan input")]},
                )
                pending = await session.plan_agent.aget_state(config("pending-target:plan"))
                self.assertTrue(pending.next)
                with self.assertRaisesRegex(RuntimeError, "pending execution"):
                    await session.set_mode("plan")
                self.assertFalse(session.mode["planning"])
                self.assertEqual((await session.agent.aget_state(config(session.id))).config, source.config)
                self.assertEqual((await session.plan_agent.aget_state(config("pending-target:plan"))).config, pending.config)
            finally:
                await self.stop(app)

    async def test_goal_review_implement_uses_frontend_and_runs_act_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frontend = ReviewFrontend("implement")
            model = ScriptedModel()
            app = await self.start(Path(directory), model=model, frontend=frontend)
            try:
                session = await app.open_session("goal")
                await session.prompt("Before Goal")
                await session.begin_goal("Goal request: finish the feature")
                with patch("agent.planning.criteria.get_llm", return_value=model):
                    await session.prompt("Goal request: finish the feature")
                self.assertEqual(len(frontend.requests), 1)
                self.assertEqual(frontend.requests[0].artifact_type, "goal")
                self.assertEqual(session.record["current_goal"]["attempts"], 1)
                self.assertEqual(session.record["current_goal"]["status"], "completed")
                self.assertFalse(session.mode["executing_goal"])
                act = await session.agent.aget_state(config(session.id))
                self.assertTrue(any("prepare_goal" in str(message.content) for message in act.values["messages"]))
                call_ids = {
                    call["id"] for message in act.values["messages"] if isinstance(message, AIMessage)
                    for call in message.tool_calls
                }
                self.assertTrue(call_ids)
                self.assertTrue(all(
                    message.tool_call_id in call_ids
                    for message in act.values["messages"] if isinstance(message, ToolMessage)
                ))
                self.assertIsNone(session.recovery)
            finally:
                await self.stop(app)

    async def test_goal_close_and_clear_leave_origin_unchanged(self) -> None:
        for action in ("close", "clear"):
            with self.subTest(action=action), tempfile.TemporaryDirectory() as directory:
                frontend = ReviewFrontend(action)
                model = ScriptedModel()
                app = await self.start(Path(directory), model=model, frontend=frontend)
                try:
                    session = await app.open_session(action)
                    await session.prompt("Origin marker")
                    before = await session.agent.aget_state(config(session.id))
                    await session.begin_goal("Goal request: finish the feature")
                    with patch("agent.planning.criteria.get_llm", return_value=model):
                        await session.prompt("Goal request: finish the feature")
                    after = await session.agent.aget_state(config(session.id))
                    self.assertEqual(content(after.values["messages"]), content(before.values["messages"]))
                    self.assertIsNone(session.mode["goal_staging"])
                    self.assertFalse(session.mode["executing_goal"])
                    self.assertEqual(len(frontend.requests), 1)
                    if action == "close":
                        self.assertIsNotNone(session.record["current_goal"])
                    if action == "clear":
                        self.assertIsNone(session.record["current_goal"])
                    await session.prompt("After Goal")
                    self.assertEqual(len(frontend.requests), 1)
                finally:
                    await self.stop(app)

    async def test_goal_started_in_plan_inherits_plan_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frontend = ReviewFrontend("close")
            model = ScriptedModel()
            app = await self.start(Path(directory), model=model, frontend=frontend)
            try:
                session = await app.open_session("goal-from-plan")
                await session.prompt("Act marker")
                await session.set_mode("plan")
                await session.prompt("Plan marker")
                plan_before = await session.plan_agent.aget_state(config("goal-from-plan:plan"))
                act_before = await session.agent.aget_state(config(session.id))
                await session.begin_goal("Goal request: finish the feature")
                staged = await session.plan_agent.aget_state(config("goal-from-plan:plan:1"))
                self.assertEqual(content(staged.values["messages"]), content(plan_before.values["messages"]))
                self.assertEqual((await session.plan_agent.aget_state(config("goal-from-plan:plan"))).config, plan_before.config)
                with patch("agent.planning.criteria.get_llm", return_value=model):
                    await session.prompt("Goal request: finish the feature")
                self.assertTrue(session.mode["planning"])
                self.assertEqual((await session.agent.aget_state(config(session.id))).config, act_before.config)
                self.assertEqual(len(frontend.requests), 1)
            finally:
                await self.stop(app)

    async def test_goal_revise_preserves_feedback_and_prior_authority(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frontend = ReviewFrontend("revise", "close")
            model = ScriptedModel()
            app = await self.start(Path(directory), model=model, frontend=frontend)
            try:
                session = await app.open_session("revise")
                await session.prompt("Origin marker")
                before = await session.agent.aget_state(config(session.id))
                await session.begin_goal("Goal request: finish the feature")
                with patch("agent.planning.criteria.get_llm", return_value=model):
                    await session.prompt("Goal request: finish the feature")
                self.assertEqual(len(frontend.requests), 2)
                self.assertEqual([event["status"] for event in session.record["events"] if event["type"] == "goal"], ["rejected", "closed"])
                revised = await session.plan_agent.aget_state(config("revise:plan:2"))
                self.assertTrue(any("Make the revised Goal clearer" in text for text in content(revised.values["messages"])))
                after = await session.agent.aget_state(config(session.id))
                self.assertEqual(content(after.values["messages"]), content(before.values["messages"]))
                self.assertIsNotNone(session.record["current_goal"])
                self.assertIsNone(session.mode["goal_staging"])
            finally:
                await self.stop(app)

    async def test_retained_goal_survives_rejected_replacement_and_is_revised(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frontend = ReviewFrontend("close", "clear", "close")
            model = ScriptedModel()
            app = await self.start(Path(directory), model=model, frontend=frontend)
            try:
                session = await app.open_session("retained")
                await session.begin_goal("Goal request: first result")
                with patch("agent.planning.criteria.get_llm", return_value=model):
                    await session.prompt("Goal request: first result")
                original = dict(session.record["current_goal"])
                await session.begin_goal("Goal request: rejected replacement")
                with patch("agent.planning.criteria.get_llm", return_value=model):
                    await session.prompt("Goal request: rejected replacement")
                self.assertEqual(session.record["current_goal"]["id"], original["id"])
                await session.begin_artifact_revision("goal", original, "Make the revised Goal clearer")
                with patch("agent.planning.criteria.get_llm", return_value=model):
                    await session.prompt("Make the revised Goal clearer")
                self.assertEqual(len(frontend.requests), 3)
                self.assertNotEqual(session.record["current_goal"]["id"], original["id"])
                self.assertEqual(session.record["current_goal"]["status"], "proposed")
                self.assertTrue(any(
                    event.get("type") == "goal" and event.get("goal", {}).get("id") == original["id"]
                    and event.get("status") == "superseded"
                    for event in session.record["events"]
                ))
            finally:
                await self.stop(app)

    async def test_closed_goal_context_survives_restart_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            model = ScriptedModel()
            first_frontend = ReviewFrontend("close")
            first = await self.start(workspace, model=model, frontend=first_frontend)
            try:
                session = await first.open_session("retained-goal")
                await session.prompt("Origin marker")
                await session.begin_goal("Goal request: finish the feature")
                with patch("agent.planning.criteria.get_llm", return_value=model):
                    await session.prompt("Goal request: finish the feature")
                self.assertEqual(session.mode["goal_context_thread_id"], "retained-goal:plan:1")
            finally:
                await self.stop(first)
            second = await self.start(workspace, model=model, frontend=ReviewFrontend())
            try:
                resumed = await second.open_session("retained-goal")
                self.assertEqual(resumed.mode["goal_context_thread_id"], "retained-goal:plan:1")
                started = await resumed.start_artifact("goal")
                self.assertEqual(started["attempts"], 1)
                source = await resumed.plan_agent.aget_state(config("retained-goal:plan:1"))
                target = await resumed.agent.aget_state(config(resumed.id))
                self.assertEqual(content(target.values["messages"]), content(source.values["messages"]))
                await resumed.prompt("Implement the retained Goal")
                self.assertEqual(resumed.record["current_goal"]["status"], "completed")
            finally:
                await self.stop(second)

    async def test_pending_goal_review_recovers_once_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            model = ScriptedModel()
            disconnected = DisconnectedReviewFrontend()
            first = await self.start(workspace, model=model, frontend=disconnected)
            try:
                session = await first.open_session("pending-goal")
                await session.begin_goal("Goal request: finish the feature")
                with patch("agent.planning.criteria.get_llm", return_value=model):
                    with self.assertRaisesRegex(RuntimeError, "frontend disconnected"):
                        await session.prompt("Goal request: finish the feature")
                self.assertEqual(len(disconnected.requests), 1)
                self.assertIsNotNone(session.recovery)
            finally:
                await self.stop(first)
            approved = ReviewFrontend("implement")
            second = await self.start(workspace, model=model, frontend=approved)
            try:
                resumed = await second.open_session("pending-goal")
                self.assertIsNotNone(resumed.recovery)
                await resumed.resume_pending()
                self.assertEqual(len(approved.requests), 1)
                self.assertEqual(resumed.record["current_goal"]["attempts"], 1)
                self.assertEqual(resumed.record["current_goal"]["status"], "completed")
                self.assertIsNone(resumed.recovery)
            finally:
                await self.stop(second)

    async def test_formal_plan_approval_executes_in_act(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frontend = ReviewFrontend("implement")
            model = ScriptedModel()
            app = await self.start(Path(directory), model=model, frontend=frontend)
            try:
                session = await app.open_session("plan")
                await session.prompt("Origin marker")
                await session.set_mode("plan")
                with patch("agent.planning.criteria.get_llm", return_value=model):
                    await session.prompt("Build a formal Plan")
                self.assertEqual(len(frontend.requests), 1)
                self.assertEqual(frontend.requests[0].artifact_type, "plan")
                self.assertFalse(session.mode["planning"])
                self.assertEqual(session.record["current_plan"]["attempts"], 1)
                self.assertEqual(session.record["current_plan"]["status"], "completed")
                act = await session.agent.aget_state(config(session.id))
                called_tools = {
                    call["name"] for message in act.values["messages"] if isinstance(message, AIMessage)
                    for call in message.tool_calls
                }
                self.assertTrue({"prepare_plan", "finalize_plan"} <= called_tools)
                self.assertIn("Build a formal Plan", [
                    message.content for message in act.values["messages"] if isinstance(message, HumanMessage)
                ])
            finally:
                await self.stop(app)
