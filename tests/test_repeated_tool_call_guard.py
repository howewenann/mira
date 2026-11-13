"""Standalone tests adapted from kxleee's DeepAgents PR #3581, plus graphs."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware, hook_config
from langchain.tools import ToolRuntime
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import ToolException, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from agent.middleware.repeated_tool_call_guard import RepeatedToolCallMiddleware


class ScriptedModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def call(index, name="read_file", **args):
    return {"name": name, "args": args or {"file_path": "/palindrome.py"}, "id": f"call-{index}"}


def responses(calls):
    return [AIMessage(content="", tool_calls=[item]) for item in calls] + [AIMessage(content="Done.")]


def assert_paired(test, messages):
    pending = []
    for message in messages:
        if isinstance(message, AIMessage):
            test.assertEqual(pending, [])
            pending.extend(item["id"] for item in message.tool_calls)
        elif isinstance(message, ToolMessage):
            test.assertIn(message.tool_call_id, pending)
            pending.remove(message.tool_call_id)
        elif isinstance(message, HumanMessage):
            test.assertEqual(pending, [])
    test.assertEqual(pending, [])


class RepeatedToolCallTests(unittest.TestCase):
    def setUp(self):
        self.executed = []

        @tool
        def read_file(file_path: str, offset: int = 0, limit: int = 100) -> str:
            """Read one file page."""
            self.executed.append((file_path, offset, limit))
            return "file contents"

        @tool
        def write_file(file_path: str, content: str = "changed") -> str:
            """Modify one file."""
            self.executed.append(("write", file_path))
            return "written"

        self.read = read_file
        self.write = write_file

    def graph(self, calls, *, middleware=None, **kwargs):
        return create_agent(
            ScriptedModel(responses=responses(calls)), tools=[self.read, self.write],
            middleware=[RepeatedToolCallMiddleware()] if middleware is None else middleware,
            **kwargs,
        )

    def invoke(self, graph, **kwargs):
        return graph.invoke({"messages": [HumanMessage(content="Inspect the files.")]}, **kwargs)

    def test_two_stages_in_real_graph_and_disabled_baseline(self):
        calls = [call(i) for i in range(9)]
        result = self.invoke(self.graph(calls))
        results = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        self.assertEqual([m.status for m in results], ["success"] * 3 + ["error"] * 3)
        self.assertEqual(len(self.executed), 3)
        self.assertIn("current agent turn has been stopped", result["messages"][-1].content)
        self.assertIn("Recovery attempts exhausted (2)", result["messages"][-1].content)
        assert_paired(self, result["messages"])
        self.executed.clear()
        result = self.invoke(self.graph(calls, middleware=[]))
        self.assertEqual(len(self.executed), 9)
        self.assertEqual(result["messages"][-1].content, "Done.")

    def test_many_files_and_pages_are_unrestricted(self):
        calls = [call(i, file_path=f"/{i}.py") for i in range(100)]
        calls += [call(100+i, file_path="/large.py", offset=i*100, limit=100+i) for i in range(20)]
        result = self.invoke(self.graph(calls), config={"recursion_limit": 600})
        self.assertEqual(len(self.executed), 120)
        self.assertEqual(result["messages"][-1].content, "Done.")

    def test_independent_custom_execution_and_recovery_thresholds(self):
        graph = self.graph([call(i) for i in range(9)], middleware=[
            RepeatedToolCallMiddleware(max_repeats=2, max_blocked_attempts=1),
        ])
        result = self.invoke(graph)
        self.assertEqual(len(self.executed), 2)
        self.assertEqual([m.status for m in result["messages"] if isinstance(m, ToolMessage)],
                         ["success", "success", "error", "error"])
        self.assertIn("Recovery attempts exhausted (1)", result["messages"][-1].content)

    def test_different_tool_write_and_different_arguments_reset(self):
        calls = [call(i) for i in range(3)]
        calls += [call(3, "write_file"), *[call(i) for i in range(4, 7)]]
        calls += [call(7, file_path="/other.py"), call(8)]
        result = self.invoke(self.graph(calls))
        self.assertEqual(len(self.executed), 9)
        self.assertTrue(all(m.status == "success" for m in result["messages"] if isinstance(m, ToolMessage)))

    def test_schema_defaults_coercions_and_dictionary_order(self):
        calls = [call(0), call(1, limit="100", offset="0", file_path="/palindrome.py"), call(2)]
        result = self.invoke(self.graph(calls, middleware=[RepeatedToolCallMiddleware(max_repeats=1)]))
        results = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        self.assertEqual([m.status for m in results], ["success", "error", "error"])
        self.assertEqual(len(self.executed), 1)

    def test_injected_runtime_does_not_prevent_normalization(self):
        @tool
        def inspect_file(file_path: str, runtime: ToolRuntime, offset: int = 0) -> str:
            """Read with injected runtime."""
            self.executed.append(file_path)
            return "ok"

        graph = create_agent(
            ScriptedModel(responses=responses([
                call(0, "inspect_file"), call(1, "inspect_file", file_path="/palindrome.py", offset="0"),
            ])), tools=[inspect_file], middleware=[RepeatedToolCallMiddleware(max_repeats=1)],
        )
        result = self.invoke(graph)
        self.assertEqual(len(self.executed), 1)
        self.assertEqual([m.status for m in result["messages"] if isinstance(m, ToolMessage)], ["success", "error"])

    def test_original_schema_custom_validators_are_preserved(self):
        from pydantic import BaseModel, field_validator

        class Arguments(BaseModel):
            file_path: str

            @field_validator("file_path")
            @classmethod
            def lowercase(cls, value):
                return value.lower()

        @tool(args_schema=Arguments)
        def read_file(file_path: str) -> str:
            """Read a normalized path."""
            self.executed.append(file_path)
            return "ok"

        graph = create_agent(ScriptedModel(responses=responses([
            call(0, file_path="/PALINDROME.PY"), call(1),
        ])), tools=[read_file], middleware=[RepeatedToolCallMiddleware(max_repeats=1)])
        result = self.invoke(graph)
        self.assertEqual(self.executed, ["/palindrome.py"])
        self.assertEqual([m.status for m in result["messages"] if isinstance(m, ToolMessage)], ["success", "error"])

    def test_failed_tool_results_also_count(self):
        @tool
        def read_file(file_path: str) -> str:
            """Fail consistently."""
            self.executed.append(file_path)
            raise ToolException("Unavailable")

        read_file.handle_tool_error = True
        graph = create_agent(ScriptedModel(responses=responses([call(i) for i in range(9)])),
                             tools=[read_file], middleware=[RepeatedToolCallMiddleware()])
        result = self.invoke(graph)
        self.assertEqual(len(self.executed), 3)
        self.assertIn("has been stopped", result["messages"][-1].content)
        assert_paired(self, result["messages"])

    def test_parallel_declaration_order_and_complete_pairing(self):
        batch = [call(i) for i in range(9)]
        model = ScriptedModel(responses=[AIMessage(content="", tool_calls=batch), AIMessage(content="Unexpected")])
        graph = create_agent(model, tools=[self.read], middleware=[RepeatedToolCallMiddleware()])
        result = self.invoke(graph)
        self.assertEqual(len(self.executed), 3)
        self.assertEqual(model.i, 1)  # No model call after hard termination.
        self.assertEqual(len([m for m in result["messages"] if isinstance(m, ToolMessage)]), 9)
        assert_paired(self, result["messages"])
        self.executed.clear()
        batch = [call(i, file_path=f"/{i}.py") for i in range(20)]
        graph = create_agent(ScriptedModel(responses=[AIMessage(content="", tool_calls=batch), AIMessage(content="Done")]),
                             tools=[self.read], middleware=[RepeatedToolCallMiddleware(max_repeats=1)])
        result = self.invoke(graph)
        self.assertEqual(len(self.executed), 20)
        assert_paired(self, result["messages"])

    def test_reused_call_ids_are_resolved_from_latest_history(self):
        # Duplicate IDs across different assistant messages occur with local models.
        result = self.invoke(self.graph([call(0)] * 9))
        self.assertEqual(len(self.executed), 3)
        self.assertIn("has been stopped", result["messages"][-1].content)

    def test_new_user_turn_and_separate_threads_reset(self):
        graph = self.graph([call(i) for i in range(9)], checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "one"}}
        result = self.invoke(graph, config=config)
        self.assertEqual(len(self.executed), 3)
        # Start the script again to compare the same calls on the retained thread.
        graph = self.graph([call(i) for i in range(9)], checkpointer=graph.checkpointer)
        result = self.invoke(graph, config=config)
        self.assertEqual(len(self.executed), 6)
        assert_paired(self, result["messages"])
        result = self.invoke(graph, config={"configurable": {"thread_id": "two"}})
        self.assertGreater(len(self.executed), 6)

    def test_sqlite_checkpoint_resume_at_terminal_tool_result(self):
        with tempfile.TemporaryDirectory() as directory, SqliteSaver.from_conn_string(str(Path(directory) / "state.db")) as saver:
            config = {"configurable": {"thread_id": "checkpoint"}}
            graph = self.graph([call(i) for i in range(9)], checkpointer=saver, interrupt_after=["tools"])
            result = self.invoke(graph, config=config)
            for _ in range(5):
                result = graph.invoke(None, config)
            self.assertEqual(len(self.executed), 3)
            self.assertEqual(graph.get_state(config).next, ("RepeatedToolCallMiddleware.before_model",))
            # Reconstruct the graph: decision is in serialized history, not the instance.
            graph = self.graph([call(50)], checkpointer=saver)
            result = graph.invoke(None, config)
            self.assertIn("has been stopped", result["messages"][-1].content)
            self.assertEqual(graph.get_state(config).next, ())
            assert_paired(self, result["messages"])
            result = self.invoke(graph, config=config)
            self.assertEqual(result["messages"][-1].content, "Done.")
            self.assertEqual(len(self.executed), 4)
            assert_paired(self, result["messages"])

    def test_constructor_validation(self):
        for key in ("max_repeats", "max_blocked_attempts"):
            for value in (0, -1, True, False, "3", 1.5, None):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    RepeatedToolCallMiddleware(**{key: value})

    def test_native_end_route_preserves_after_agent_revision_policy(self):
        # Pinned LangChain runs after-agent hooks on jump_to='end'. This is
        # an explicit limitation: a revision hook can start another attempt.
        revisions = []

        class Revision(AgentMiddleware):
            @hook_config(can_jump_to=["model"])
            def after_agent(self, state, runtime):
                if not revisions:
                    revisions.append(True)
                    return {"jump_to": "model", "messages": [HumanMessage(content="Revise this response.")]}
                return None

        model = ScriptedModel(responses=responses([call(i) for i in range(9)]))
        graph = create_agent(model, tools=[self.read], middleware=[RepeatedToolCallMiddleware(), Revision()])
        result = self.invoke(graph)
        self.assertEqual(len(self.executed), 6)
        self.assertEqual(result["messages"][-1].content, "Done.")
        assert_paired(self, result["messages"])

    def test_direct_wrappers_stable_json_and_no_handler_execution(self):
        messages = [HumanMessage(content="Search")]
        calls = []
        guard = RepeatedToolCallMiddleware(max_repeats=1)
        for i, args in enumerate(({"query": "x", "limit": 5}, {"limit": 5, "query": "x"})):
            current = call(i, "search", **args)
            messages.append(AIMessage(content="", tool_calls=[current]))
            request = ToolCallRequest(tool_call=current, tool=None, state={"messages": messages}, runtime=None)

            def handler(req):
                calls.append(req)
                return ToolMessage(content="ok", tool_call_id=req.tool_call["id"])

            result = guard.wrap_tool_call(request, handler)
            messages.append(result)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result.status, "error")
        self.assertEqual(result.tool_call_id, "call-1")


class AsyncRepeatedToolCallTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_sqlite_terminal_checkpoint_and_subsequent_user_turn(self):
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        executed = []

        @tool
        async def read_file(file_path: str) -> str:
            """Read asynchronously."""
            executed.append(file_path)
            return "ok"

        with tempfile.TemporaryDirectory() as directory:
            async with AsyncSqliteSaver.from_conn_string(str(Path(directory) / "state.db")) as saver:
                config = {"configurable": {"thread_id": "async-checkpoint"}}
                graph = create_agent(ScriptedModel(responses=responses([call(i) for i in range(9)])),
                                     tools=[read_file], middleware=[RepeatedToolCallMiddleware()],
                                     checkpointer=saver, interrupt_after=["tools"])
                await graph.ainvoke({"messages": [HumanMessage(content="Inspect")]}, config)
                for _ in range(5):
                    await graph.ainvoke(None, config)
                graph = create_agent(ScriptedModel(responses=responses([call(50)])), tools=[read_file],
                                     middleware=[RepeatedToolCallMiddleware()], checkpointer=saver)
                result = await graph.ainvoke(None, config)
                self.assertEqual(len(executed), 3)
                self.assertIn("has been stopped", result["messages"][-1].content)
                self.assertEqual((await graph.aget_state(config)).next, ())
                assert_paired(self, result["messages"])
                result = await graph.ainvoke({"messages": [HumanMessage(content="Read again")]}, config)
                self.assertEqual(len(executed), 4)
                self.assertEqual(result["messages"][-1].content, "Done.")
                assert_paired(self, result["messages"])

    async def test_async_graph_parallel_completion_order_and_thread_isolation(self):
        executed = []

        @tool
        async def read_file(file_path: str, offset: int = 0) -> str:
            """Read asynchronously."""
            await asyncio.sleep(0.001 if file_path == "/slow.py" else 0)
            executed.append(file_path)
            return "ok"

        guard = RepeatedToolCallMiddleware()
        async def run(thread):
            graph = create_agent(ScriptedModel(responses=responses([call(i) for i in range(9)])),
                                 tools=[read_file], middleware=[guard], checkpointer=InMemorySaver())
            return await graph.ainvoke({"messages": [HumanMessage(content="Inspect")]},
                                      {"configurable": {"thread_id": thread}})

        results = await asyncio.gather(run("a"), run("b"))
        self.assertEqual(len(executed), 6)
        for result in results:
            assert_paired(self, result["messages"])
            self.assertIn("has been stopped", result["messages"][-1].content)

        executed.clear()
        batch = [call(i, file_path="/slow.py" if i % 2 == 0 else "/fast.py") for i in range(20)]
        graph = create_agent(ScriptedModel(responses=[AIMessage(content="", tool_calls=batch), AIMessage(content="Done")]),
                             tools=[read_file], middleware=[RepeatedToolCallMiddleware(max_repeats=1)])
        result = await graph.ainvoke({"messages": [HumanMessage(content="Inspect")]})
        self.assertEqual(len(executed), 20)
        assert_paired(self, result["messages"])

    async def test_cancellation_propagates_and_checkpoint_can_resume(self):
        entered = asyncio.Event()
        release = asyncio.Event()
        executed = []

        @tool
        async def read_file(file_path: str) -> str:
            """Wait until released."""
            entered.set()
            await release.wait()
            executed.append(file_path)
            return "ok"

        graph = create_agent(ScriptedModel(responses=responses([call(i) for i in range(9)])), tools=[read_file],
                             middleware=[RepeatedToolCallMiddleware()], checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "cancel"}}
        task = asyncio.create_task(graph.ainvoke({"messages": [HumanMessage(content="Inspect")]}, config))
        await asyncio.wait_for(entered.wait(), 5)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        release.set()
        result = await graph.ainvoke(None, config)
        self.assertEqual(len(executed), 3)
        self.assertIn("has been stopped", result["messages"][-1].content)
        assert_paired(self, result["messages"])
        result = await graph.ainvoke({"messages": [HumanMessage(content="A new request")]}, config)
        self.assertEqual(result["messages"][-1].content, "Done.")
        self.assertEqual(len(executed), 6)
        assert_paired(self, result["messages"])


if __name__ == "__main__":
    unittest.main()
