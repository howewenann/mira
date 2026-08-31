"""Focused tests for model tool visibility and execution enforcement."""

from __future__ import annotations

import unittest
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from agent.middleware.model_tool_visibility import ModelToolVisibilityMiddleware
from agent.tools.specs import tool_name


class ModelRequest:
    def __init__(self, tools: list[Any]) -> None:
        self.tools = tools

    def override(self, **kwargs: Any) -> "ModelRequest":
        return ModelRequest(kwargs.get("tools", self.tools))


class ToolRequest:
    def __init__(self, name: str, call_id: str = "call-1") -> None:
        self.tool_call = {"name": name, "args": {}, "id": call_id}


class BindableFakeModel(FakeMessagesListChatModel):
    def bind_tools(self, tools: list[Any], *, tool_choice: Any = None, **kwargs: Any) -> Any:
        return self


class ModelToolVisibilityMiddlewareTests(unittest.IsolatedAsyncioTestCase):
    def test_excluded_tool_is_removed_from_model_request(self) -> None:
        middleware = ModelToolVisibilityMiddleware(("write_file",))

        filtered = middleware._filter_request(
            ModelRequest([{"name": "read_file"}, {"name": "write_file"}])
        )

        self.assertEqual([tool_name(tool) for tool in filtered.tools], ["read_file"])

    def test_excluded_tool_call_is_rejected_without_reaching_handler(self) -> None:
        middleware = ModelToolVisibilityMiddleware(("write_file",))
        called = False

        def handler(_request: Any) -> ToolMessage:
            nonlocal called
            called = True
            return ToolMessage(content="wrote", tool_call_id="call-1")

        result = middleware.wrap_tool_call(ToolRequest("write_file"), handler)

        self.assertFalse(called)
        self.assertIsInstance(result, ToolMessage)
        self.assertEqual(result.status, "error")
        self.assertEqual(result.content, "Error: write_file is not available.")

    def test_allowed_tool_call_reaches_handler(self) -> None:
        middleware = ModelToolVisibilityMiddleware(allowed_tools=("read_file",))
        expected = ToolMessage(content="read", name="read_file", tool_call_id="call-1")

        result = middleware.wrap_tool_call(ToolRequest("read_file"), lambda _request: expected)

        self.assertIs(result, expected)

    async def test_async_excluded_tool_call_is_rejected(self) -> None:
        middleware = ModelToolVisibilityMiddleware(("write_file",))
        called = False

        async def handler(_request: Any) -> ToolMessage:
            nonlocal called
            called = True
            return ToolMessage(content="wrote", tool_call_id="call-1")

        result = await middleware.awrap_tool_call(ToolRequest("write_file"), handler)

        self.assertFalse(called)
        self.assertEqual(result.status, "error")

    def test_newly_excluded_previously_visible_tool_cannot_execute(self) -> None:
        middleware = ModelToolVisibilityMiddleware(())
        request = ModelRequest([{"name": "read_file"}])
        first = middleware._filter_request(request)
        self.assertEqual([tool_name(tool) for tool in first.tools], ["read_file"])

        middleware.excluded_tools.add("read_file")
        second = middleware._filter_request(request)
        called = False

        def handler(_request: Any) -> ToolMessage:
            nonlocal called
            called = True
            return ToolMessage(content="read", tool_call_id="call-1")

        result = middleware.wrap_tool_call(ToolRequest("read_file"), handler)

        self.assertEqual(second.tools, [])
        self.assertFalse(called)
        self.assertEqual(result.status, "error")

    def test_stale_tool_call_is_rejected_through_native_agent_pipeline(self) -> None:
        calls: list[str] = []

        @tool
        def remembered_tool(value: str) -> str:
            """Record one visible tool call."""
            calls.append(value)
            return value

        middleware = ModelToolVisibilityMiddleware(())
        model = BindableFakeModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "remembered_tool",
                            "args": {"value": "first"},
                            "id": "call-first",
                        }
                    ],
                ),
                AIMessage(content="first complete"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "remembered_tool",
                            "args": {"value": "stale"},
                            "id": "call-stale",
                        }
                    ],
                ),
                AIMessage(content="stale call handled"),
            ]
        )
        graph = create_agent(model, tools=[remembered_tool], middleware=[middleware])

        first = graph.invoke({"messages": [HumanMessage(content="Use the tool.")]})
        middleware.excluded_tools.add("remembered_tool")
        second = graph.invoke({"messages": [HumanMessage(content="Use it again.")]})

        self.assertEqual(calls, ["first"])
        self.assertTrue(
            any(
                isinstance(message, ToolMessage)
                and message.status == "error"
                and "not available" in str(message.content)
                for message in second["messages"]
            )
        )
        self.assertTrue(any(isinstance(message, ToolMessage) for message in first["messages"]))


if __name__ == "__main__":
    unittest.main()
