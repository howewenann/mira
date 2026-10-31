"""Cold restart and isolation checks for native session checkpoints."""

from __future__ import annotations

import sqlite3
import asyncio
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import AsyncMock, patch

from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
from langgraph.checkpoint.serde.types import _DeltaSnapshot
from langgraph.types import interrupt
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
import aiosqlite

from config.metadata import ModelMetadata
from config.settings import normalize_settings, set_rubric_enabled
from core.application.app import MiraApplication
from core.interface.requests import ApprovalRequest
from session.goals import goal_artifact, replace_current_goal
from session.plans import plan_artifact, replace_current_plan
from tests.test_workflows import WorkflowModel
from tests.test_checkpoint import BindableFakeMessagesListChatModel, Finding
from tests.test_runner import RunTurnRenderer
from core.execution.runner import run_turn


class SqliteSessionTests(unittest.IsolatedAsyncioTestCase):
    async def start(self, workspace: Path, *, model=None, frontend=None, rubric=False, todos=False) -> MiraApplication:
        model = model or WorkflowModel()
        settings = normalize_settings({"system": {"planning_todos": {"enabled": True}}} if todos else {})
        if rubric:
            settings = set_rubric_enabled(settings, True)
        config = {
            "session_dir": str(workspace / ".mira" / "_sessions"),
            "settings": settings,
            "settings_valid": True,
        }
        metadata = ModelMetadata(context_tokens=32768, context_source="test")
        with (
            patch("agent.llm.active_model_issues", return_value=[]),
            patch("agent.llm.get_llm", return_value=model),
            patch("agent.factory.get_llm", return_value=model),
            patch("agent.factory.get_rubric_model_name", return_value="sqlite-test"),
            patch("agent.llm.get_model_name", return_value="sqlite-test"),
            patch("config.metadata.infer_model_metadata", new=AsyncMock(return_value=metadata)),
        ):
            application = await MiraApplication.start(workspace=workspace, config=config, frontend=frontend)
        # Opening also builds a separate Act/Plan pair for the session.
        application._test_model_patch = patch("agent.factory.get_llm", return_value=model)
        application._test_model_patch.start()
        application._test_rubric_name_patch = patch(
            "agent.factory.get_rubric_model_name", return_value="sqlite-test"
        )
        application._test_rubric_name_patch.start()
        return application

    async def stop(self, application: MiraApplication) -> None:
        try:
            await application.shutdown()
        finally:
            application._test_rubric_name_patch.stop()
            application._test_model_patch.stop()

    async def test_cold_restart_restores_messages_and_session_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            first = await self.start(workspace)
            session = await first.open_session("one")
            self.assertIs(type(session.checkpointer), AsyncSqliteSaver)
            self.assertIs(type(session.checkpointer.serde), JsonPlusSerializer)
            await session.prompt("first message")
            await session.set_mode("plan")
            await session.prompt("plan something")
            database = first.store.database_path("one")
            self.assertTrue(database.is_file())
            self.assertEqual(database.name, "checkpoints.db")
            self.assertEqual(first.store.path("one"), database.parent / "session.json")
            self.assertEqual([path.name for path in database.parent.glob("*.json")], ["session.json"])
            self.assertTrue(any(event.get("type") == "user" for event in session.record["events"]))
            with closing(sqlite3.connect(database)) as connection:
                self.assertEqual(connection.execute("PRAGMA journal_mode").fetchone()[0], "wal")
                self.assertGreater(connection.execute("SELECT count(*) FROM checkpoints").fetchone()[0], 0)
            await self.stop(first)

            second = await self.start(workspace)
            try:
                restored = await second.open_session("one")
                state = await restored.agent.aget_state({"configurable": {"thread_id": "one"}})
                self.assertTrue(any(isinstance(message, HumanMessage) and message.content == "first message" for message in state.values["messages"]))
                self.assertTrue(restored.mode["planning"])
                plan_state = await restored.plan_agent.aget_state(
                    {"configurable": {"thread_id": "one:plan"}}
                )
                self.assertTrue(plan_state.values.get("messages"))
                self.assertIsNone(restored.recovery)
                await restored.set_mode("act")
                await restored.prompt("second message")
                continued = await restored.agent.aget_state({"configurable": {"thread_id": "one"}})
                human = [message.content for message in continued.values["messages"] if isinstance(message, HumanMessage)]
                self.assertEqual(human.count("first message"), 1)
                self.assertEqual(human.count("second message"), 1)
            finally:
                await self.stop(second)

    async def test_two_sessions_use_distinct_databases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            application = await self.start(Path(directory))
            try:
                first = await application.open_session("one")
                second = await application.open_session("two")
                await first.prompt("alpha")
                await second.prompt("beta")
                self.assertIsNot(first.checkpointer, second.checkpointer)
                self.assertTrue(application.store.database_path("one").is_file())
                self.assertTrue(application.store.database_path("two").is_file())
                self.assertEqual(
                    {path.name for path in application.store.root.iterdir()}, {"one", "two"}
                )
                state = await second.agent.aget_state({"configurable": {"thread_id": "two"}})
                self.assertFalse(any(getattr(message, "content", "") == "alpha" for message in state.values["messages"]))
                application._policy_refresh_generation += 1
                await application.refresh_agents_for_policy()
                self.assertIs(first.agent.checkpointer, first.checkpointer)
                self.assertIs(second.agent.checkpointer, second.checkpointer)
                # Windows will refuse rmtree while SQLite still owns these files.
                for suffix in ("-wal", "-shm", "-journal"):
                    Path(f"{application.store.database_path('one')}{suffix}").touch()
                self.assertTrue(await application.delete_session("one"))
                self.assertEqual(first.runtime_state, "closed")
                self.assertFalse(application.store.session_directory("one").exists())
                self.assertFalse(application.store.database_path("one").exists())
                self.assertFalse(any(
                    Path(f"{application.store.database_path('one')}{suffix}").exists()
                    for suffix in ("-wal", "-shm", "-journal")
                ))
                self.assertTrue(application.store.database_path("two").exists())
                await second.prompt("still here")
                self.assertEqual(await application.clear_sessions(), 1)
                self.assertEqual(second.runtime_state, "closed")
                self.assertFalse(application.store.session_directory("two").exists())
                self.assertFalse(application.store.database_path("two").exists())
            finally:
                await self.stop(application)

    async def test_native_rubric_channels_survive_cold_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            first = await self.start(workspace, rubric=True)
            session = await first.open_session("rubric")
            await session.prompt("first task")
            await session.agent.aupdate_state(
                {"configurable": {"thread_id": session.id}},
                {
                    "_rubric_status": "max_iterations_reached",
                    "_rubric_iterations": 2,
                    "_rubric_criteria": ["First task is complete"],
                },
            )
            await self.stop(first)

            second = await self.start(workspace, rubric=True)
            try:
                reopened = await second.open_session("rubric")
                state = await reopened.agent.aget_state(
                    {"configurable": {"thread_id": reopened.id}}
                )
                self.assertEqual(state.values["_rubric_status"], "max_iterations_reached")
                self.assertEqual(state.values["_rubric_iterations"], 2)
                self.assertEqual(state.values["_rubric_criteria"], ["First task is complete"])
            finally:
                await self.stop(second)

    async def test_formal_artifact_lifecycle_survives_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            first = await self.start(workspace)
            session = await first.open_session("formal")
            plan = plan_artifact(
                plan_id="plan-1", title="Test plan", objective="Build it",
                context_and_constraints="Keep it small", key_changes=["One change"],
                test_plan=["Run tests"], assumptions=["None"],
                success_criteria="It works", rubric_enabled=True, rubric_iterations=2,
            )
            replace_current_plan(session.record, plan)
            await asyncio.wait_for(session.persist_graph_state(), timeout=5)
            first.store.save(session.record)
            await session.start_artifact("plan")
            await self.stop(first)

            second = await self.start(workspace)
            restored = await second.open_session("formal")
            self.assertEqual(restored.record["current_plan"]["key_changes"], ["One change"])
            self.assertEqual(restored.record["current_plan"]["attempts"], 1)
            self.assertEqual(restored.record["current_plan"]["status"], "paused")
            restored.pause_active_artifacts()
            await restored.persist_graph_state()
            await restored.clear_artifact("plan")
            await restored.begin_goal("New objective")
            goal = goal_artifact(
                goal_id="goal-1", title="Test goal", objective="New objective",
                success_criteria="Done", rubric_enabled=False, rubric_iterations=3,
            )
            replace_current_goal(restored.record, goal)
            await restored.persist_graph_state()
            second.store.save(restored.record)
            await self.stop(second)

            third = await self.start(workspace)
            try:
                reopened = await third.open_session("formal")
                self.assertIsNone(reopened.record["current_plan"])
                self.assertEqual(reopened.record["current_goal"]["success_criteria"], "Done")
                self.assertEqual(reopened.mode["goal_staging"]["thread_id"], "formal:plan:1")
            finally:
                await self.stop(third)

    async def test_saved_tool_approval_is_redisplayed_before_native_resume(self) -> None:
        effects: list[str] = []

        @tool
        def side_effect(value: str) -> str:
            """Record one visible external action."""
            effects.append(value)
            return "done"

        with tempfile.TemporaryDirectory() as directory:
            database = str(Path(directory) / "hitl.db")
            config = {"configurable": {"thread_id": "hitl"}}
            connection = await aiosqlite.connect(database)
            saver = AsyncSqliteSaver(connection)
            first = create_agent(
                model=BindableFakeMessagesListChatModel(responses=[AIMessage(
                    content="", tool_calls=[{"name": "side_effect", "args": {"value": "x"}, "id": "call-1"}]
                )]),
                tools=[side_effect],
                middleware=[HumanInTheLoopMiddleware(interrupt_on={"side_effect": True})],
                checkpointer=saver,
            )
            stream = await first.astream_events({"messages": [{"role": "user", "content": "do it"}]}, config=config, version="v3")
            await stream.output()
            saved_interrupts = await stream.interrupts()
            self.assertEqual(len(saved_interrupts), 1)
            self.assertEqual(effects, [])
            await connection.close()

            connection = await aiosqlite.connect(database)
            try:
                saver = AsyncSqliteSaver(connection)
                second = create_agent(
                    model=BindableFakeMessagesListChatModel(responses=[AIMessage(content="finished")]),
                    tools=[side_effect],
                    middleware=[HumanInTheLoopMiddleware(interrupt_on={"side_effect": True})],
                    checkpointer=saver,
                )
                state = await second.aget_state(config)
                restored_interrupts = [interrupt for task in state.tasks for interrupt in task.interrupts]
                self.assertEqual(len(restored_interrupts), 1)
                renderer = RunTurnRenderer(decisions=[{"type": "reject", "message": "no"}])
                await run_turn(second, "", renderer, "hitl", resume_interrupts=restored_interrupts)
                self.assertEqual(len(renderer.approvals), 1)
                self.assertEqual(effects, [])
            finally:
                await connection.close()

            approved_config = {"configurable": {"thread_id": "hitl-approved"}}
            connection = await aiosqlite.connect(database)
            try:
                saver = AsyncSqliteSaver(connection)
                approving_first = create_agent(
                    model=BindableFakeMessagesListChatModel(responses=[AIMessage(
                        content="", tool_calls=[{"name": "side_effect", "args": {"value": "x"}, "id": "call-2"}]
                    )]),
                    tools=[side_effect],
                    middleware=[HumanInTheLoopMiddleware(interrupt_on={"side_effect": True})],
                    checkpointer=saver,
                )
                stream = await approving_first.astream_events(
                    {"messages": [{"role": "user", "content": "do it"}]},
                    config=approved_config, version="v3",
                )
                await stream.output()
                self.assertEqual(len(await stream.interrupts()), 1)
            finally:
                await connection.close()

            connection = await aiosqlite.connect(database)
            try:
                saver = AsyncSqliteSaver(connection)
                approving_second = create_agent(
                    model=BindableFakeMessagesListChatModel(responses=[AIMessage(content="finished")]),
                    tools=[side_effect],
                    middleware=[HumanInTheLoopMiddleware(interrupt_on={"side_effect": True})],
                    checkpointer=saver,
                )
                state = await approving_second.aget_state(approved_config)
                interrupts = [interrupt for task in state.tasks for interrupt in task.interrupts]
                renderer = RunTurnRenderer(decisions=[{"type": "approve"}])
                await run_turn(
                    approving_second, "", renderer, "hitl-approved", resume_interrupts=interrupts
                )
                self.assertEqual(len(renderer.approvals), 1)
                self.assertEqual(effects, ["x"])
            finally:
                await connection.close()

    async def test_sqlite_stock_serializer_restores_checkpoint_and_pending_writes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connection = await aiosqlite.connect(str(Path(directory) / "serde.db"))
            try:
                saver = AsyncSqliteSaver(connection)
                self.assertIs(type(saver), AsyncSqliteSaver)
                self.assertIs(type(saver.serde), JsonPlusSerializer)
                config = {"configurable": {"thread_id": "serde", "checkpoint_ns": ""}}
                checkpoint = {
                    "v": 4, "ts": "2026-01-01T00:00:00+00:00", "id": "checkpoint-1",
                    "channel_values": {
                        "messages": [AIMessage(content="stored")],
                        "state": _DeltaSnapshot({"finding": Finding(summary="saved"),
                                                 "flags": {"enabled": True}}),
                    },
                    "channel_versions": {"messages": "1", "state": "1"},
                    "versions_seen": {}, "pending_sends": [],
                }
                saved = await saver.aput(config, checkpoint, {}, {"messages": "1", "state": "1"})
                await saver.aput_writes(saved, [("state", {"finding": Finding(summary="queued")})], "task-1")
                loaded = await saver.aget_tuple(saved)
                self.assertEqual(loaded.checkpoint["channel_values"]["messages"][0].content, "stored")
                state = loaded.checkpoint["channel_values"]["state"]
                self.assertIsInstance(state, _DeltaSnapshot)
                self.assertIsInstance(state.value["finding"], Finding)
                self.assertEqual(state.value["finding"].summary, "saved")
                self.assertEqual(state.value["flags"], {"enabled": True})
                self.assertEqual(loaded.pending_writes, [
                    ("task-1", "state", {"finding": Finding(summary="queued")}),
                ])
            finally:
                await connection.close()

    async def test_incomplete_non_hitl_checkpoint_requires_explicit_continuation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            first = await self.start(workspace)
            session = await first.open_session("incomplete")
            await session.agent.aupdate_state(
                {"configurable": {"thread_id": session.id}},
                {"messages": [HumanMessage(content="unfinished")]},
                as_node="__start__",
            )
            await self.stop(first)

            second = await self.start(workspace)
            try:
                reopened = await second.open_session("incomplete")
                self.assertIsNotNone(reopened.recovery)
                self.assertTrue(reopened.recovery["incomplete"])
                state = await reopened.agent.aget_state({"configurable": {"thread_id": reopened.id}})
                self.assertTrue(state.next)
                self.assertEqual([m.content for m in state.values["messages"]], ["unfinished"])
                with self.assertRaisesRegex(RuntimeError, "explicitly continue"):
                    await reopened.prompt("new request")
                unchanged = await reopened.agent.aget_state({"configurable": {"thread_id": reopened.id}})
                self.assertEqual(unchanged.config, state.config)
            finally:
                await self.stop(second)

    async def test_application_redisplays_saved_approval_after_cold_restart(self) -> None:
        effects: list[str] = []

        @tool
        def side_effect(value: str) -> str:
            """Record an approved action."""
            effects.append(value)
            return "done"

        class RejectingFrontend:
            def __init__(self) -> None:
                self.approvals = 0

            def emit(self, event) -> None:
                pass

            async def request(self, request):
                if isinstance(request, ApprovalRequest):
                    self.approvals += 1
                    return [{"type": "reject_once"}]
                raise AssertionError(type(request))

        first_model = BindableFakeMessagesListChatModel(responses=[AIMessage(
            content="", tool_calls=[{"name": "side_effect", "args": {"value": "x"}, "id": "call-1"}]
        )])
        second_model = BindableFakeMessagesListChatModel(responses=[AIMessage(content="stopped")])
        active_model = first_model

        def pair(*, checkpointer, **kwargs):
            del kwargs
            graph = create_agent(
                model=active_model, tools=[side_effect],
                middleware=[HumanInTheLoopMiddleware(interrupt_on={"side_effect": True})],
                checkpointer=checkpointer,
            )
            return graph, graph

        with tempfile.TemporaryDirectory() as directory, patch(
            "core.application.app.build_agent_pair", side_effect=pair
        ):
            workspace = Path(directory)
            first = await self.start(workspace)
            session = await first.open_session("approval")
            session.mode["goal_staging"] = {
                "thread_id": "approval:plan:1", "authoritative_objective": "Do the task"
            }
            await session.persist_graph_state()
            stream = await session.agent.astream_events(
                {"messages": [{"role": "user", "content": "run side effect"}]},
                config={"configurable": {"thread_id": session.id}}, version="v3",
            )
            await stream.output()
            self.assertEqual(len(await stream.interrupts()), 1)
            await self.stop(first)

            active_model = second_model
            frontend = RejectingFrontend()
            second = await self.start(workspace, frontend=frontend)
            try:
                reopened = await second.open_session("approval")
                self.assertIsNotNone(reopened.recovery)
                self.assertEqual(len(reopened.recovery["interrupts"]), 1)
                self.assertEqual(frontend.approvals, 0)
                with self.assertRaisesRegex(RuntimeError, "pending checkpoint request"):
                    await reopened.set_mode("plan")
                with self.assertRaisesRegex(RuntimeError, "pending checkpoint request"):
                    await reopened.begin_goal("Another Goal")
                await reopened.resume_pending()
                self.assertEqual(frontend.approvals, 1)
                self.assertEqual(effects, [])
                self.assertIsNone(reopened.mode["goal_staging"])
                state = await reopened.agent.aget_state({"configurable": {"thread_id": reopened.id}})
                self.assertFalse(state.next)
            finally:
                await self.stop(second)

    async def test_saved_ask_user_interrupt_uses_native_resume(self) -> None:
        answers: list[str] = []

        @tool
        def ask_user() -> str:
            """Ask for one decision."""
            answer = interrupt({"type": "ask_user", "question": "Choose A or B", "options": ["A", "B"]})
            answers.append(answer)
            return answer

        with tempfile.TemporaryDirectory() as directory:
            database = str(Path(directory) / "ask.db")
            config = {"configurable": {"thread_id": "ask"}}
            connection = await aiosqlite.connect(database)
            try:
                saver = AsyncSqliteSaver(connection)
                first = create_agent(
                    model=BindableFakeMessagesListChatModel(responses=[AIMessage(
                        content="", tool_calls=[{"name": "ask_user", "args": {}, "id": "ask-1"}]
                    )]),
                    tools=[ask_user], checkpointer=saver,
                )
                stream = await first.astream_events(
                    {"messages": [{"role": "user", "content": "ask"}]}, config=config, version="v3"
                )
                await stream.output()
                self.assertEqual(len(await stream.interrupts()), 1)
                self.assertEqual(answers, [])
            finally:
                await connection.close()

            connection = await aiosqlite.connect(database)
            try:
                saver = AsyncSqliteSaver(connection)
                second = create_agent(
                    model=BindableFakeMessagesListChatModel(responses=[AIMessage(content="done")]),
                    tools=[ask_user], checkpointer=saver,
                )
                state = await second.aget_state(config)
                saved_interrupts = [item for task in state.tasks for item in task.interrupts]
                renderer = RunTurnRenderer(ask_user_answer="B")
                await run_turn(second, "", renderer, "ask", resume_interrupts=saved_interrupts)
                self.assertEqual(len(renderer.ask_user_prompts), 1)
                self.assertEqual(answers, ["B"])
            finally:
                await connection.close()
