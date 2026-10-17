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

    async def test_missing_checkpoint_is_not_restored_from_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            store = SessionStore(workspace / ".mira" / "_sessions")
            record = store.load(None, resume=False, workspace=workspace)
            record["turns"] = 1
            record["events"] = [{"type": "user", "text": "old JSON only"}]
            store.save(record)
            app = MiraApplication(
                frontend=NullFrontend(), workspace=workspace,
                store=store, config={"settings": {}}, action_resources=None,
                agent_metadata=None, model_name="fake", context_limit_tokens=100,
                context_limit_source="test", mcp_manager=None,
                agent_unavailable_message="", agent=None, plan_agent=None,
            )
            with self.assertRaisesRegex(RuntimeError, "no LangGraph checkpoint"):
                await app.open_session(record["id"], resume=True)
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
