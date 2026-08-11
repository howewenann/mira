"""Tests for model-visible workspace tool failures."""

from __future__ import annotations

import asyncio
import unittest
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphBubbleUp, GraphInterrupt

from agent.factory import subagents_with_project_tool_errors
from agent.middleware.project_tool_errors import ProjectToolErrorMiddleware


class Request:
    def __init__(self, name: str = "read_file_as_bytes", call_id: str = "call-read") -> None:
        self.tool_call = {"name": name, "args": {"path": "/missing.bin"}, "id": call_id}


class BindableFakeModel(FakeMessagesListChatModel):
    def bind_tools(self, tools: list[Any], *, tool_choice: Any = None, **kwargs: Any) -> Any:
        return self


class ProjectToolErrorMiddlewareTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.middleware = ProjectToolErrorMiddleware({"read_file_as_bytes"})

    def test_sync_project_exception_becomes_native_tool_error(self) -> None:
        def fail(_request: Any) -> Any:
            raise FileNotFoundError("missing.bin")

        result = self.middleware.wrap_tool_call(Request(), fail)

        self.assertIsInstance(result, ToolMessage)
        self.assertEqual(result.status, "error")
        self.assertEqual(result.name, "read_file_as_bytes")
        self.assertEqual(result.tool_call_id, "call-read")
        self.assertEqual(
            result.content,
            "read_file_as_bytes failed in the MIRA runtime.\nFileNotFoundError: missing.bin",
        )

    async def test_async_project_exception_becomes_native_tool_error(self) -> None:
        async def fail(_request: Any) -> Any:
            raise RuntimeError("broken reader")

        result = await self.middleware.awrap_tool_call(Request(), fail)

        self.assertIsInstance(result, ToolMessage)
        self.assertEqual(result.status, "error")
        self.assertIn("RuntimeError: broken reader", str(result.content))

    def test_non_project_exception_keeps_native_behavior(self) -> None:
        def fail(_request: Any) -> Any:
            raise FileNotFoundError("missing.txt")

        with self.assertRaises(FileNotFoundError):
            self.middleware.wrap_tool_call(Request(name="read_file"), fail)

    def test_graph_control_flow_is_not_converted(self) -> None:
        for error_type in (GraphBubbleUp, GraphInterrupt):
            with self.subTest(error_type=error_type.__name__):
                def interrupt(_request: Any) -> Any:
                    raise error_type()

                with self.assertRaises(error_type):
                    self.middleware.wrap_tool_call(Request(), interrupt)

    def test_process_level_exceptions_are_not_converted(self) -> None:
        for error_type in (KeyboardInterrupt, SystemExit):
            with self.subTest(error_type=error_type.__name__):
                def stop(_request: Any) -> Any:
                    raise error_type()

                with self.assertRaises(error_type):
                    self.middleware.wrap_tool_call(Request(), stop)

    async def test_async_cancellation_is_not_converted(self) -> None:
        async def cancel(_request: Any) -> Any:
            raise asyncio.CancelledError

        with self.assertRaises(asyncio.CancelledError):
            await self.middleware.awrap_tool_call(Request(), cancel)

    def test_raw_subagents_receive_middleware_without_mutation(self) -> None:
        raw = {"name": "general-purpose", "middleware": ["existing"]}
        compiled = {"name": "compiled", "runnable": object()}
        remote = {"name": "remote", "graph_id": "agent"}

        prepared = subagents_with_project_tool_errors(
            [raw, compiled, remote],
            frozenset({"read_file_as_bytes"}),
        )

        self.assertEqual(raw["middleware"], ["existing"])
        self.assertIsNot(prepared[0], raw)
        self.assertEqual(prepared[0]["middleware"][0], "existing")
        self.assertIsInstance(prepared[0]["middleware"][1], ProjectToolErrorMiddleware)
        self.assertIs(prepared[1], compiled)
        self.assertIs(prepared[2], remote)

    async def test_handled_failure_reaches_model_and_leaves_no_dangling_call(self) -> None:
        @tool
        def read_file_as_bytes(path: str) -> str:
            """Read bytes from a file."""
            raise FileNotFoundError(path)

        model = BindableFakeModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "read_file_as_bytes",
                            "args": {"path": "/missing.bin"},
                            "id": "call-read",
                        }
                    ],
                ),
                AIMessage(content="The file is missing."),
                AIMessage(content="The next turn is intact."),
            ]
        )
        graph = create_agent(
            model,
            tools=[read_file_as_bytes],
            middleware=[ProjectToolErrorMiddleware({"read_file_as_bytes"})],
            checkpointer=InMemorySaver(),
        )
        config = {"configurable": {"thread_id": "thread-1"}}

        first = await graph.ainvoke(
            {"messages": [HumanMessage(content="Read the missing file.")]},
            config=config,
        )
        second = await graph.ainvoke(
            {"messages": [HumanMessage(content="Continue.")]},
            config=config,
        )

        errors = [message for message in first["messages"] if isinstance(message, ToolMessage)]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].status, "error")
        self.assertIn("FileNotFoundError", str(errors[0].content))
        self.assertEqual(first["messages"][-1].content, "The file is missing.")
        self.assertEqual(second["messages"][-1].content, "The next turn is intact.")
        self.assertFalse(
            any(
                "was cancelled - another message came in" in str(message.content)
                for message in second["messages"]
                if isinstance(message, ToolMessage)
            )
        )


if __name__ == "__main__":
    unittest.main()
