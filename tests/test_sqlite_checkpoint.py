"""Native SQLite checkpoint and chat lifecycle regression tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain.agents import create_agent
from langchain.agents.middleware.human_in_the_loop import HumanInTheLoopMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from deepagents import create_deep_agent
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel

from core.application.app import MiraApplication
from core.execution.runner import run_turn
from core.interface import NullFrontend
from session.checkpoint import session_checkpointer
from session.store import SessionStore
from tests.test_checkpoint import BindableFakeMessagesListChatModel, toy
from tests.test_runner import RunTurnRenderer


def conversation(saver):
    def respond(state: MessagesState):
        return {"messages": [AIMessage(content=f"seen {len(state['messages'])} messages")]}

    graph = StateGraph(MessagesState)
    graph.add_node("respond", respond)
    graph.add_edge(START, "respond")
    graph.add_edge("respond", END)
    return graph.compile(checkpointer=saver)


class SqliteCheckpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_deepagents_subagent_interrupt_resumes_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chat.sqlite"
            config = {"configurable": {"thread_id": "act"}}

            def agent(saver, parent_responses, child_responses):
                return create_deep_agent(
                    model=BindableFakeMessagesListChatModel(responses=parent_responses),
                    subagents=[{
                        "name": "worker", "description": "Runs toy commands",
                        "model": BindableFakeMessagesListChatModel(responses=child_responses),
                        "tools": [toy], "interrupt_on": {"toy": True},
                    }],
                    checkpointer=saver,
                )

            async with session_checkpointer(path) as saver:
                graph = agent(
                    saver,
                    [AIMessage(content="", tool_calls=[{
                        "name": "task", "args": {
                            "description": "run toy", "subagent_type": "worker",
                        }, "id": "task-1",
                    }])],
                    [AIMessage(content="", tool_calls=[{
                        "name": "toy", "args": {"command": "safe"}, "id": "toy-1",
                    }])],
                )
                result = await graph.ainvoke(
                    {"messages": [HumanMessage(content="delegate a toy command")]}, config
                )
                self.assertIn("__interrupt__", result)
            async with session_checkpointer(path) as saver:
                graph = agent(
                    saver, [AIMessage(content="parent finished")],
                    [AIMessage(content="child finished")],
                )
                pending = await graph.ainvoke(None, config)
                self.assertIn("__interrupt__", pending)
                result = await graph.ainvoke(
                    Command(resume={"decisions": [{"type": "approve"}]}), config
                )
                self.assertEqual(result["messages"][-1].content, "parent finished")
                self.assertIn("child finished", result["messages"][-2].content)

    async def test_native_hitl_approval_resumes_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chat.sqlite"
            config = {"configurable": {"thread_id": "act"}}

            def agent(saver, responses):
                return create_agent(
                    model=BindableFakeMessagesListChatModel(responses=responses),
                    tools=[toy],
                    middleware=[HumanInTheLoopMiddleware(interrupt_on={"toy": True})],
                    checkpointer=saver,
                )

            async with session_checkpointer(path) as saver:
                graph = agent(saver, [AIMessage(content="", tool_calls=[{
                    "name": "toy", "args": {"command": "safe"}, "id": "call-1",
                }])])
                result = await graph.ainvoke(
                    {"messages": [HumanMessage(content="run toy")]}, config
                )
                self.assertIn("__interrupt__", result)
                self.assertTrue((await graph.aget_state(config)).next)
            async with session_checkpointer(path) as saver:
                graph = agent(saver, [AIMessage(content="finished")])
                renderer = RunTurnRenderer()
                result = await run_turn(graph, "", renderer, "act", resume_existing=True)
                self.assertEqual(len(renderer.approvals), 1)
                state = await graph.aget_state(config)
                self.assertEqual([message.type for message in state.values["messages"]], [
                    "human", "ai", "tool", "ai",
                ])
                self.assertEqual(state.values["messages"][2].content, "ran safe")
                self.assertEqual(result.final_text, "finished")

    async def test_serializer_safeguards_cover_sqlite_pending_writes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chat.sqlite"
            config = {"configurable": {"thread_id": "act", "checkpoint_ns": ""}}
            checkpoint = {
                "v": 4, "ts": "2026-01-01T00:00:00+00:00", "id": "checkpoint-1",
                "channel_values": {"state": {"internal": BaseModel.__pydantic_serializer__}},
                "channel_versions": {"state": "1"}, "versions_seen": {}, "pending_sends": [],
            }
            async with session_checkpointer(path) as saver:
                saved = await saver.aput(config, checkpoint, {}, {"state": "1"})
                await saver.aput_writes(saved, [("state", {"schema": AIMessage})], "task-1")
            async with session_checkpointer(path) as saver:
                loaded = await saver.aget_tuple(saved)
                self.assertIsNotNone(loaded)
                assert loaded is not None
                self.assertEqual(
                    loaded.checkpoint["channel_values"]["state"]["internal"]["__mira_pydantic_internal__"],
                    "pydantic._internal._mock_val_ser.MockValSer",
                )
                self.assertEqual(
                    loaded.pending_writes[0][2]["schema"]["__mira_type__"],
                    "langchain_core.messages.ai.AIMessage",
                )

    async def test_turns_and_restart_preserve_native_messages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chat.sqlite"
            config = {"configurable": {"thread_id": "act"}}
            async with session_checkpointer(path) as saver:
                graph = conversation(saver)
                await graph.ainvoke({"messages": [HumanMessage(content="one")]}, config)
                await graph.ainvoke({"messages": [HumanMessage(content="two")]}, config)
                before = await graph.aget_state(config)
                self.assertEqual([item.content for item in before.values["messages"]], [
                    "one", "seen 1 messages", "two", "seen 3 messages",
                ])
            async with session_checkpointer(path) as saver:
                graph = conversation(saver)
                after = await graph.aget_state(config)
                self.assertEqual(after.values, before.values)
                self.assertEqual(after.next, before.next)
                await graph.ainvoke({"messages": [HumanMessage(content="three")]}, config)
                self.assertEqual(len((await graph.aget_state(config)).values["messages"]), 6)

    async def test_interrupt_and_nested_subgraph_resume_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chat.sqlite"
            config = {"configurable": {"thread_id": "act"}}

            def build(saver):
                child = StateGraph(MessagesState)

                def ask(state: MessagesState):
                    answer = interrupt("approve")
                    return {"messages": [AIMessage(content=f"approved {answer}")]}

                child.add_node("ask", ask)
                child.add_edge(START, "ask")
                child.add_edge("ask", END)
                parent = StateGraph(MessagesState)
                parent.add_node("subagent", child.compile())
                parent.add_edge(START, "subagent")
                parent.add_edge("subagent", END)
                return parent.compile(checkpointer=saver)

            async with session_checkpointer(path) as saver:
                graph = build(saver)
                result = await graph.ainvoke({"messages": [HumanMessage(content="start")]}, config)
                self.assertIn("__interrupt__", result)
                snapshot = await graph.aget_state(config, subgraphs=True)
                self.assertEqual(snapshot.next, ("subagent",))
                self.assertTrue(snapshot.tasks[0].state)
            async with session_checkpointer(path) as saver:
                graph = build(saver)
                snapshot = await graph.aget_state(config, subgraphs=True)
                self.assertEqual(snapshot.next, ("subagent",))
                result = await graph.ainvoke(Command(resume="yes"), config)
                self.assertEqual(result["messages"][-1].content, "approved yes")

    async def test_chat_switch_and_deletion_are_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            store = SessionStore(workspace / ".mira" / "_sessions")
            app = MiraApplication(
                frontend=NullFrontend(), workspace=workspace,
                store=store, config={"settings": {}}, action_resources=object(),
                agent_metadata=None, model_name="fake", context_limit_tokens=100,
                context_limit_source="test", mcp_manager=None,
                agent_unavailable_message="", agent=None, plan_agent=None,
            )
            with patch("core.application.app.build_agent_pair", side_effect=lambda **kw: (
                conversation(kw["checkpointer"]), conversation(kw["checkpointer"])
            )):
                first = await app.open_session()
                first_id = first.id
                await first.agent.ainvoke({"messages": [HumanMessage(content="first")]},
                                          {"configurable": {"thread_id": first_id}})
                first.record["turns"] = 1
                store.save(first.record)
                await first.close()
                second = await app.open_session()
                await second.agent.ainvoke({"messages": [HumanMessage(content="second")]},
                                           {"configurable": {"thread_id": second.id}})
                self.assertNotEqual(first_id, second.id)
                await second.close()
                restored = await app.open_session(first_id, resume=True)
                state = await restored.agent.aget_state({"configurable": {"thread_id": first_id}})
                self.assertEqual(state.values["messages"][0].content, "first")
                self.assertTrue(await app.delete_session(first_id))
                self.assertEqual(restored.runtime_state, "closed")
                self.assertFalse(store.checkpoint_path(first_id).exists())
                await app.shutdown()

    async def test_restart_preserves_plan_threads_and_allocates_new_goal_thread(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            store = SessionStore(workspace / ".mira" / "_sessions")

            def application():
                return MiraApplication(
                    frontend=NullFrontend(), workspace=workspace,
                    store=store, config={"settings": {}}, action_resources=object(),
                    agent_metadata=None, model_name="fake", context_limit_tokens=100,
                    context_limit_source="test", mcp_manager=None,
                    agent_unavailable_message="", agent=None, plan_agent=None,
                )

            with patch("core.application.app.build_agent_pair", side_effect=lambda **kw: (
                conversation(kw["checkpointer"]), conversation(kw["checkpointer"])
            )):
                first_app = application()
                session = await first_app.open_session()
                await session.agent.ainvoke(
                    {"messages": [HumanMessage(content="act one")]},
                    {"configurable": {"thread_id": session.id}},
                )
                await session.plan_agent.ainvoke(
                    {"messages": [HumanMessage(content="plan one")]},
                    {"configurable": {"thread_id": f"{session.id}:plan"}},
                )
                session.begin_goal("first objective")
                first_thread = session.mode["goal_staging"]["thread_id"]
                await session.plan_agent.ainvoke(
                    {"messages": [HumanMessage(content="goal one")]},
                    {"configurable": {"thread_id": first_thread}},
                )
                session_id = session.id
                await first_app.shutdown()

                second_app = application()
                restored = await second_app.open_session(session_id, resume=True)
                act = await restored.agent.aget_state(
                    {"configurable": {"thread_id": session_id}}
                )
                plan = await restored.plan_agent.aget_state(
                    {"configurable": {"thread_id": f"{session_id}:plan"}}
                )
                old = await restored.plan_agent.aget_state(
                    {"configurable": {"thread_id": first_thread}}
                )
                self.assertEqual(act.values["messages"][0].content, "act one")
                self.assertEqual(plan.values["messages"][0].content, "plan one")
                self.assertEqual(old.values["messages"][0].content, "goal one")
                restored.begin_goal("second objective")
                self.assertNotEqual(restored.mode["goal_staging"]["thread_id"], first_thread)
                self.assertEqual(restored.record["plan_runs"], 2)
                await second_app.shutdown()

    async def test_empty_checkpoint_database_does_not_restore_json_as_graph_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            store = SessionStore(workspace / ".mira" / "_sessions")
            record = store.load(None, resume=False, workspace=workspace)
            record["turns"] = 1
            record["events"] = [{"type": "user", "text": "old JSON only"}]
            store.save(record)
            async with session_checkpointer(store.checkpoint_path(record["id"])):
                pass
            app = MiraApplication(
                frontend=NullFrontend(), workspace=workspace,
                store=store, config={"settings": {}}, action_resources=None,
                agent_metadata=None, model_name="fake", context_limit_tokens=100,
                context_limit_source="test", mcp_manager=None,
                agent_unavailable_message="", agent=None, plan_agent=None,
            )
            session = await app.open_session(record["id"], resume=True)
            self.assertIsNone(session.agent)
            with self.assertRaisesRegex(RuntimeError, "no LangGraph checkpoint"):
                await session.prompt("continue")
            await app.shutdown()

    async def test_json_history_remains_viewable_when_first_checkpoint_was_never_written(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            store = SessionStore(workspace / ".mira" / "_sessions")
            record = store.load(None, resume=False, workspace=workspace)
            record["events"] = [{"type": "user", "text": "Attempted request"}]
            store.save(record)
            app = MiraApplication(
                frontend=NullFrontend(), workspace=workspace, store=store,
                config={"settings": {}}, action_resources=None,
                agent_metadata=None, model_name="fake", context_limit_tokens=100,
                context_limit_source="test", mcp_manager=None,
                agent_unavailable_message="", agent=None, plan_agent=None,
            )
            session = await app.open_session(record["id"], resume=True)
            self.assertEqual(session.snapshot().transcript[0]["text"], "Attempted request")
            with self.assertRaisesRegex(RuntimeError, "no LangGraph checkpoint"):
                await session.prompt("continue")
            self.assertFalse(store.checkpoint_path(record["id"]).exists())
            await app.shutdown()

    async def test_application_reopens_and_resumes_pending_hitl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            store = SessionStore(workspace / ".mira" / "_sessions")

            class ApprovingFrontend:
                def emit(self, event):
                    pass

                async def request(self, request):
                    return [{"type": "approve"}]

            def application(frontend):
                return MiraApplication(
                    frontend=frontend, workspace=workspace,
                    store=store, config={"settings": {}}, action_resources=object(),
                    agent_metadata=None, model_name="fake", context_limit_tokens=100,
                    context_limit_source="test", mcp_manager=None,
                    agent_unavailable_message="", agent=None, plan_agent=None,
                )

            def graph(saver, responses):
                return create_agent(
                    model=BindableFakeMessagesListChatModel(responses=responses),
                    tools=[toy],
                    middleware=[HumanInTheLoopMiddleware(interrupt_on={"toy": True})],
                    checkpointer=saver,
                )

            with patch("core.application.app.build_agent_pair", side_effect=lambda **kw: (
                graph(kw["checkpointer"], [AIMessage(content="", tool_calls=[{
                    "name": "toy", "args": {"command": "safe"}, "id": "call-1",
                }])]),
                graph(kw["checkpointer"], [AIMessage(content="plan")]),
            )):
                first_app = application(NullFrontend())
                first = await first_app.open_session()
                config = {"configurable": {"thread_id": first.id}}
                result = await first.agent.ainvoke(
                    {"messages": [HumanMessage(content="run toy")]}, config
                )
                self.assertIn("__interrupt__", result)
                first.record["events"] = [{"type": "user", "text": "run toy"}]
                store.save(first.record)
                session_id = first.id
                await first_app.shutdown()

            with patch("core.application.app.build_agent_pair", side_effect=lambda **kw: (
                graph(kw["checkpointer"], [AIMessage(content="finished")]),
                graph(kw["checkpointer"], [AIMessage(content="plan")]),
            )):
                second_app = application(ApprovingFrontend())
                restored = await second_app.open_session(session_id, resume=True)
                self.assertEqual(restored.pending_thread_id, session_id)
                await restored.resume_pending()
                self.assertIsNone(restored.pending_thread_id)
                state = await restored.agent.aget_state(config)
                self.assertEqual(state.values["messages"][-1].content, "finished")
                await second_app.shutdown()

    async def test_goal_research_interrupt_keeps_checkpointed_objective_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            store = SessionStore(workspace / ".mira" / "_sessions")

            def planning_graph(saver, responses):
                from agent.middleware.planning_stage_enforcement import PlanningStageEnforcementMiddleware

                return create_agent(
                    model=BindableFakeMessagesListChatModel(responses=responses),
                    tools=[toy],
                    middleware=[
                        PlanningStageEnforcementMiddleware(),
                        HumanInTheLoopMiddleware(interrupt_on={"toy": True}),
                    ],
                    checkpointer=saver,
                )

            class ApprovingFrontend:
                def emit(self, event):
                    pass

                async def request(self, request):
                    return [{"type": "approve"}]

            def application():
                return MiraApplication(
                    frontend=ApprovingFrontend(), workspace=workspace, store=store,
                    config={"settings": {}}, action_resources=object(),
                    agent_metadata=None, model_name="fake", context_limit_tokens=100,
                    context_limit_source="test", mcp_manager=None,
                    agent_unavailable_message="", agent=None, plan_agent=None,
                )

            builds = 0

            def build_pair(**kw):
                nonlocal builds
                builds += 1
                responses = ([AIMessage(content="", tool_calls=[{
                    "name": "toy", "args": {"command": "safe"}, "id": "research-call",
                }])] if builds == 1 else [
                    AIMessage(content="Research complete"),
                    AIMessage(content="Plan continued"),
                ])
                return (
                    create_agent(
                        model=BindableFakeMessagesListChatModel(responses=[AIMessage(content="Action happened")]),
                        tools=[], checkpointer=kw["checkpointer"],
                    ),
                    planning_graph(kw["checkpointer"], responses),
                )

            with patch("core.application.app.build_agent_pair", side_effect=build_pair):
                first_app = application()
                first = await first_app.open_session()
                first.begin_goal("Original objective")
                thread_id = first.mode["goal_staging"]["thread_id"]
                result = await first.plan_agent.ainvoke(
                    {"messages": [HumanMessage(content="Create Goal")],
                     "planning_stage": "goal_research",
                     "planning_authoritative_request": "Original objective"},
                    {"configurable": {"thread_id": thread_id}},
                )
                self.assertIn("__interrupt__", result)
                first.record["events"] = [{"type": "user", "text": "Create Goal"}]
                store.save(first.record)
                session_id = first.id
                await first_app.shutdown()

                second_app = application()
                try:
                    restored = await second_app.open_session(session_id, resume=True)
                    self.assertEqual(restored.pending_thread_id, thread_id)
                    resumed = await restored.resume_pending()
                    self.assertEqual(resumed.final_text, "Research complete")
                    continuation = await restored.prompt("Continue the Goal")
                    self.assertEqual(continuation.final_text, "Action happened")
                    state = await restored.plan_agent.aget_state(
                        {"configurable": {"thread_id": thread_id}}
                    )
                    self.assertEqual(state.values["planning_authoritative_request"], "Original objective")
                finally:
                    await second_app.shutdown()

    async def test_paused_goal_and_plan_execution_finish_after_checkpoint_resume(self) -> None:
        from session.goals import goal_artifact, replace_current_goal
        from session.plans import plan_artifact, replace_current_plan

        for kind in ("goal", "plan"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                workspace = Path(directory)
                store = SessionStore(workspace / ".mira" / "_sessions")

                class ApprovingFrontend:
                    def emit(self, event):
                        pass

                    async def request(self, request):
                        return [{"type": "approve"}]

                def application():
                    return MiraApplication(
                        frontend=ApprovingFrontend(), workspace=workspace, store=store,
                        config={"settings": {}}, action_resources=object(),
                        agent_metadata=None, model_name="fake", context_limit_tokens=100,
                        context_limit_source="test", mcp_manager=None,
                        agent_unavailable_message="", agent=None, plan_agent=None,
                    )

                builds = 0

                def build_pair(**kw):
                    nonlocal builds
                    builds += 1
                    response = (AIMessage(content="", tool_calls=[{
                        "name": "toy", "args": {"command": "safe"}, "id": "execution-call",
                    }]) if builds == 1 else AIMessage(content="Finished"))
                    return (
                        create_agent(
                            model=BindableFakeMessagesListChatModel(responses=[response]),
                            tools=[toy],
                            middleware=[HumanInTheLoopMiddleware(interrupt_on={"toy": True})],
                            checkpointer=kw["checkpointer"],
                        ),
                        conversation(kw["checkpointer"]),
                    )

                with patch("core.application.app.build_agent_pair", side_effect=build_pair):
                    first_app = application()
                    first = await first_app.open_session()
                    if kind == "goal":
                        replace_current_goal(first.record, goal_artifact(
                            goal_id="goal-1", title="Finish Goal", objective="Exact objective",
                            success_criteria="- Exact criterion", rubric_enabled=False,
                            rubric_iterations=3,
                        ))
                    else:
                        replace_current_plan(first.record, plan_artifact(
                            plan_id="plan-1", title="Finish Plan", objective="Exact objective",
                            context_and_constraints="Constraint", key_changes=["Change"],
                            test_plan=["Test"], assumptions=["None"],
                            success_criteria="- Exact criterion", rubric_enabled=False,
                            rubric_iterations=3,
                        ))
                    first.start_artifact(kind)
                    result = await first.agent.ainvoke(
                        {"messages": [HumanMessage(content="Implement")],
                         "rubric": None},
                        {"configurable": {"thread_id": first.id}},
                    )
                    self.assertIn("__interrupt__", result)
                    first.record["events"] = [{"type": "user", "text": "Implement"}]
                    first.pause_active_artifacts()
                    session_id = first.id
                    await first_app.shutdown()

                    second_app = application()
                    try:
                        restored = await second_app.open_session(session_id, resume=True)
                        self.assertEqual(restored.pending_thread_id, session_id)
                        await restored.resume_pending()
                        self.assertEqual(restored.record[f"current_{kind}"]["status"], "completed")
                    finally:
                        await second_app.shutdown()

    async def test_goal_and_plan_formal_review_after_restart(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        from agent.middleware.planning_stage_enforcement import PlanningStageEnforcementMiddleware
        from agent.planning.tool_context import PlanningToolContext
        from agent.resources.defaults.tools.finalize_goal import finalize_goal
        from agent.resources.defaults.tools.finalize_plan import finalize_plan
        from agent.resources.defaults.tools.prepare_goal import prepare_goal
        from agent.resources.defaults.tools.prepare_plan import prepare_plan
        from core.interface import ArtifactReviewRequest
        from tests.test_current_plan import BindablePlanningModel

        for kind, first_decision in (
            ("goal", "implement"), ("plan", "implement"),
            ("goal", "revise"), ("plan", "revise"),
        ):
            with self.subTest(kind=kind, decision=first_decision), tempfile.TemporaryDirectory() as directory:
                workspace = Path(directory)
                store = SessionStore(workspace / ".mira" / "_sessions")
                criteria = f"- Exact {kind} criterion"
                revised_criteria = f"- Revised {kind} criterion"
                service = SimpleNamespace(
                    generate=AsyncMock(return_value=criteria),
                    revise=AsyncMock(return_value=revised_criteria),
                )

                class ReviewFrontend:
                    reviews = 0

                    def emit(self, event):
                        pass

                    async def request(self, request):
                        if isinstance(request, ArtifactReviewRequest):
                            self.reviews += 1
                            if first_decision == "revise" and self.reviews == 1:
                                return {"action": "revise", "feedback": "Keep the objective; tighten the criterion."}
                            return {"action": "close" if first_decision == "revise" else "implement"}
                        raise AssertionError(f"unexpected request: {request}")

                def application():
                    return MiraApplication(
                        frontend=ReviewFrontend(), workspace=workspace, store=store,
                        config={"settings": {}}, action_resources=object(),
                        agent_metadata=None, model_name="fake", context_limit_tokens=100,
                        context_limit_source="test", mcp_manager=None,
                        agent_unavailable_message="", agent=None, plan_agent=None,
                    )

                builds = 0

                def build_pair(**kw):
                    nonlocal builds
                    builds += 1
                    prepare = prepare_goal if kind == "goal" else prepare_plan
                    finalize = finalize_goal if kind == "goal" else finalize_plan
                    prepare_args = {
                        "objective": "Exact objective",
                        "context_and_constraints": "Constraint",
                    }
                    if kind == "goal":
                        prepare_args["research_evidence"] = "Evidence"
                    finalize_args = {"title": f"Exact {kind.title()}"}
                    if kind == "plan":
                        finalize_args.update({
                            "key_changes": ["Change"], "test_plan": ["Test"],
                            "assumptions": ["None"],
                        })
                    proposal_responses = [
                        AIMessage(content="", tool_calls=[{
                            "name": prepare.name, "args": prepare_args, "id": "prepare-call",
                        }]),
                        AIMessage(content="", tool_calls=[{
                            "name": finalize.name, "args": finalize_args, "id": "finalize-call",
                        }]),
                    ]
                    responses = (
                        proposal_responses if builds == 1 or first_decision == "revise"
                        else [AIMessage(content="Unexpected planner call")]
                    )
                    planner = create_agent(
                        model=BindablePlanningModel(responses=responses),
                        tools=[prepare, finalize],
                        middleware=[PlanningStageEnforcementMiddleware()],
                        context_schema=PlanningToolContext,
                        checkpointer=kw["checkpointer"],
                    )
                    planner.mira_context_factory = lambda: PlanningToolContext(service)
                    action = create_agent(
                        model=BindableFakeMessagesListChatModel(
                            responses=[AIMessage(content="Action completed")]
                        ),
                        tools=[], checkpointer=kw["checkpointer"],
                    )
                    return action, planner

                with patch("core.application.app.build_agent_pair", side_effect=build_pair):
                    first_app = application()
                    first = await first_app.open_session()
                    if kind == "goal":
                        first.begin_goal("Exact objective")
                        thread_id = first.mode["goal_staging"]["thread_id"]
                    else:
                        await first.set_mode("plan")
                        thread_id = first.mode["plan_thread_id"]
                    result = await first.plan_agent.ainvoke(
                        {"messages": [HumanMessage(content="Create artifact")],
                         "planning_stage": f"{kind}_research",
                         "planning_authoritative_request": "Exact objective"},
                        {"configurable": {"thread_id": thread_id}},
                        context=PlanningToolContext(service),
                    )
                    self.assertIn("__interrupt__", result)
                    first.record["events"] = [{"type": "user", "text": "Create artifact"}]
                    store.save(first.record)
                    session_id = first.id
                    await first_app.shutdown()

                    second_app = application()
                    try:
                        restored = await second_app.open_session(session_id, resume=True)
                        self.assertEqual(restored.pending_thread_id, thread_id)
                        await restored.resume_pending()
                        artifact = restored.record[f"current_{kind}"]
                        self.assertEqual(artifact["objective"], "Exact objective")
                        self.assertEqual(
                            artifact["success_criteria"],
                            revised_criteria if first_decision == "revise" else criteria,
                        )
                        if first_decision == "revise":
                            service.revise.assert_awaited_once()
                            self.assertIn(
                                "Keep the objective; tighten the criterion.",
                                service.revise.await_args.args,
                            )
                            if kind == "goal":
                                self.assertNotEqual(
                                    restored.mode["goal_staging"]["thread_id"], thread_id
                                )
                            else:
                                self.assertEqual(restored.mode["plan_thread_id"], thread_id)
                        else:
                            self.assertEqual(artifact["status"], "completed")
                            action_state = await restored.agent.aget_state(
                                {"configurable": {"thread_id": session_id}}
                            )
                            self.assertEqual(action_state.values["messages"][-1].content, "Action completed")
                    finally:
                        await second_app.shutdown()

    async def test_clear_all_removes_orphaned_checkpoint_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SessionStore(root)
            record = store.load(None, resume=False, workspace=root)
            async with session_checkpointer(store.checkpoint_path(record["id"])) as saver:
                await conversation(saver).ainvoke(
                    {"messages": [HumanMessage(content="remove")]},
                    {"configurable": {"thread_id": record["id"]}},
                )
            orphan = store.checkpoint_path("orphan")
            async with session_checkpointer(orphan) as saver:
                await conversation(saver).ainvoke(
                    {"messages": [HumanMessage(content="orphan")]},
                    {"configurable": {"thread_id": "orphan"}},
                )
            self.assertEqual(store.clear_all(), 1)
            self.assertFalse(store.checkpoint_path(record["id"]).exists())
            self.assertFalse(orphan.exists())

    async def test_application_clear_all_closes_open_windows_connections(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            store = SessionStore(workspace / ".mira" / "_sessions")
            app = MiraApplication(
                frontend=NullFrontend(), workspace=workspace,
                store=store, config={"settings": {}}, action_resources=object(),
                agent_metadata=None, model_name="fake", context_limit_tokens=100,
                context_limit_source="test", mcp_manager=None,
                agent_unavailable_message="", agent=None, plan_agent=None,
            )
            with patch("core.application.app.build_agent_pair", side_effect=lambda **kw: (
                conversation(kw["checkpointer"]), conversation(kw["checkpointer"])
            )):
                first = await app.open_session()
                second = await app.open_session()
                await first.agent.ainvoke(
                    {"messages": [HumanMessage(content="one")]},
                    {"configurable": {"thread_id": first.id}},
                )
                await second.agent.ainvoke(
                    {"messages": [HumanMessage(content="two")]},
                    {"configurable": {"thread_id": second.id}},
                )
                self.assertEqual(await app.clear_all_sessions(), 2)
                self.assertEqual(first.runtime_state, "closed")
                self.assertEqual(second.runtime_state, "closed")
                self.assertFalse(store.checkpoint_path(first.id).exists())
                self.assertFalse(store.checkpoint_path(second.id).exists())
                await app.shutdown()

    async def test_cleared_chat_id_has_no_resurrected_graph_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chat.sqlite"
            config = {"configurable": {"thread_id": "same-chat"}}
            async with session_checkpointer(path) as saver:
                await conversation(saver).ainvoke(
                    {"messages": [HumanMessage(content="before clear")]}, config
                )
            path.unlink()
            async with session_checkpointer(path) as saver:
                graph = conversation(saver)
                self.assertEqual((await graph.aget_state(config)).values, {})
                await graph.ainvoke({"messages": [HumanMessage(content="after clear")]}, config)
                state = await graph.aget_state(config)
                self.assertEqual(state.values["messages"][0].content, "after clear")

    async def test_recompiled_agent_reads_same_session_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chat.sqlite"
            config = {"configurable": {"thread_id": "act"}}
            async with session_checkpointer(path) as saver:
                original = conversation(saver)
                await original.ainvoke({"messages": [HumanMessage(content="before reload")]}, config)
                rebuilt = conversation(saver)
                self.assertEqual(
                    (await rebuilt.aget_state(config)).values,
                    (await original.aget_state(config)).values,
                )
                await rebuilt.ainvoke({"messages": [HumanMessage(content="after reload")]}, config)
                self.assertEqual(len((await rebuilt.aget_state(config)).values["messages"]), 4)


if __name__ == "__main__":
    unittest.main()
