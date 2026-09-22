"""Production execution-context regression tests."""

from __future__ import annotations

import tempfile
import unittest
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any, TypedDict
from unittest.mock import patch

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command

from agent import factory
from agent.execution import MiraContext, create_mira_context
from agent.execution.registry import executable_tool_registry
from agent.execution.tools import RUNTIME_PAYLOAD_TAG
from agent.resources import build_resources


class StaticModel(BaseChatModel):
    """Local model sufficient for compiling real DeepAgents graphs."""

    @property
    def _llm_type(self) -> str:
        return "mira-execution-context-test"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "StaticModel":
        del tools, kwargs
        return self

    def _generate(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del messages, stop, run_manager, kwargs
        return ChatResult(generations=[ChatGeneration(message=AIMessage("done"))])


class NestedToolModel(StaticModel):
    """Exercise a real subagent tool turn before returning."""

    bound_tool_names: list[str] = []

    def bind_tools(self, tools: Any, **kwargs: Any) -> "NestedToolModel":
        del kwargs
        clone = self.model_copy(deep=True)
        clone.bound_tool_names = [str(getattr(item, "name", "")) for item in tools]
        return clone

    def _generate(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        tool_ids = {
            message.tool_call_id
            for message in messages
            if isinstance(message, ToolMessage)
        }
        if "child-context-call" in tool_ids:
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage("child done"))]
            )
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "child_context_tool",
                                "args": {},
                                "id": "child-context-call",
                                "type": "tool_call",
                            }
                        ],
                    )
                )
            ]
        )


class CaptureToolCalls(AsyncCallbackHandler):
    """Capture callback payloads and markers from nested calls."""

    def __init__(self) -> None:
        self.starts: list[dict[str, Any]] = []

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: Any,
        inputs: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        del input_str, run_id, kwargs
        self.starts.append(
            {
                "name": serialized.get("name"),
                "inputs": inputs,
                "tags": list(tags or []),
                "metadata": dict(metadata or {}),
            }
        )


class WorkflowState(TypedDict, total=False):
    filename: str
    listing: str
    agent_result: str


class ExecutionContextTests(unittest.IsolatedAsyncioTestCase):
    def build_agent(self, workspace: Path) -> Any:
        with patch("agent.factory.get_llm", return_value=StaticModel()):
            return factory.build_agent({}, workspace, None)

    async def test_real_registry_and_native_workflow_context_call_builtins(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            agent = self.build_agent(workspace)
            registry = executable_tool_registry(agent)
            context = agent.mira_context_factory()

            self.assertIs(context.tools["write_file"].tool, registry["write_file"])
            self.assertIn("general-purpose", context.agents)
            with self.assertRaisesRegex(TypeError, "immutable"):
                context.tools["replacement"] = context.tools["write_file"]
            with self.assertRaisesRegex(TypeError, "immutable"):
                context.agents["replacement"] = context.agents["general-purpose"]

            async def save(
                state: WorkflowState,
                runtime: Runtime[MiraContext],
            ) -> WorkflowState:
                await runtime.context.tools["write_file"].ainvoke(
                    {"file_path": f"/{state['filename']}", "content": "hello"}
                )
                listing = runtime.context.tools["ls"].invoke({"path": "/"})
                agent_result = await runtime.context.agents["general-purpose"].ainvoke(
                    "Return a short result."
                )
                return {"listing": str(listing), "agent_result": str(agent_result)}

            graph = StateGraph(WorkflowState, context_schema=MiraContext)
            graph.add_node("save", save)
            graph.add_edge(START, "save")
            graph.add_edge("save", END)
            result = await graph.compile().ainvoke(
                {"filename": "context.txt"},
                context=context,
            )

            self.assertEqual((workspace / "context.txt").read_text(encoding="utf-8"), "hello")
            self.assertIn("context.txt", result["listing"])
            self.assertEqual(result["agent_result"], "done")

            class DelegateState(TypedDict, total=False):
                result: str

            def delegate_sync(
                _state: DelegateState,
                runtime: Runtime[MiraContext],
            ) -> DelegateState:
                return {
                    "result": runtime.context.agents["general-purpose"].invoke(
                        "Return a synchronous result."
                    )
                }

            delegate_graph = StateGraph(
                DelegateState,
                context_schema=MiraContext,
            )
            delegate_graph.add_node("delegate", delegate_sync)
            delegate_graph.add_edge(START, "delegate")
            delegate_graph.add_edge("delegate", END)
            delegated = delegate_graph.compile().invoke({}, context=context)
            self.assertEqual(delegated["result"], "done")

    async def test_project_tool_calls_actual_nested_tool_without_parent_payload(self) -> None:
        captures: dict[str, Any] = {}
        large_value = "MIRA_NESTED_PAYLOAD" * 10_000
        callbacks = CaptureToolCalls()

        @tool
        async def inner_context_tool(value: str, runtime: ToolRuntime) -> str:
            """Return a large internal value and capture the injected runtime."""
            captures["inner_runtime"] = runtime
            captures["inner_context"] = runtime.context
            captures["inner_state"] = runtime.state
            return f"{value}:{large_value}"

        @tool
        async def outer_context_tool(value: str, runtime: ToolRuntime) -> str:
            """Call another MIRA capability through normal ToolRuntime context."""
            captures["outer_runtime"] = runtime
            captures["outer_context"] = runtime.context
            nested = await runtime.context.tools["inner_context_tool"].ainvoke(
                {"value": value}
            )
            captures["nested"] = nested
            return "nested complete"

        @tool
        async def child_context_tool(runtime: ToolRuntime) -> str:
            """Capture context propagated through the real task subagent."""
            captures["child_context"] = runtime.context
            captures["child_state"] = runtime.state
            return "child context captured"

        self.assertNotIn("runtime", outer_context_tool.args)

        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            resources = build_resources(workspace, create_examples=False, config={})
            general_purpose = dict(resources.subagents[0])
            general_purpose["tools"] = ["child_context_tool"]
            resources = replace(
                resources,
                tools=[inner_context_tool, outer_context_tool, child_context_tool],
                subagents=[general_purpose],
                metadata={
                    **resources.metadata,
                    "tools": [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "source": "project",
                        }
                        for tool in (
                            inner_context_tool,
                            outer_context_tool,
                            child_context_tool,
                        )
                    ],
                },
            )
            config = {
                "settings": {
                    "hitl": {
                        "tools": {
                            "child_context_tool": {
                                "enabled": True,
                                "always_allow": True,
                            }
                        }
                    }
                }
            }
            with patch("agent.factory.get_llm", return_value=NestedToolModel()):
                agent = factory.build_agent(config, workspace, None, resources=resources)
            context = agent.mira_context_factory()
            parent_messages = [
                HumanMessage("run nested tool"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "outer_context_tool",
                            "args": {"value": "payload"},
                            "id": "outer-call",
                            "type": "tool_call",
                        }
                    ],
                ),
            ]
            tool_graph = StateGraph(MessagesState, context_schema=MiraContext)
            tool_graph.add_node("tools", agent.nodes["tools"].bound)
            tool_graph.add_edge(START, "tools")
            tool_graph.add_edge("tools", END)
            with warnings.catch_warnings(record=True) as caught:
                result = await tool_graph.compile().ainvoke(
                    {"messages": parent_messages},
                    context=context,
                    config={
                        "callbacks": [callbacks],
                        "metadata": {"source": "execution-context-test"},
                    },
                )

                class DelegateState(TypedDict, total=False):
                    result: str

                async def delegate(
                    _state: DelegateState,
                    runtime: Runtime[MiraContext],
                ) -> DelegateState:
                    delegated = await runtime.context.agents[
                        "general-purpose"
                    ].ainvoke("Check the child runtime context.")
                    return {"result": str(delegated)}

                delegate_graph = StateGraph(
                    DelegateState,
                    context_schema=MiraContext,
                )
                delegate_graph.add_node("delegate", delegate)
                delegate_graph.add_edge(START, "delegate")
                delegate_graph.add_edge("delegate", END)
                delegated = await delegate_graph.compile().ainvoke(
                    {},
                    context=context,
                    config={
                        "callbacks": [callbacks],
                        "metadata": {"source": "execution-context-test"},
                    },
                )

            self.assertIs(captures["outer_context"], context)
            self.assertIs(captures["inner_context"], context)
            self.assertIs(captures.get("child_context"), context)
            self.assertNotEqual(
                captures["outer_runtime"].tool_call_id,
                captures["inner_runtime"].tool_call_id,
            )
            self.assertIs(
                captures["outer_runtime"].stream_writer,
                captures["inner_runtime"].stream_writer,
            )
            self.assertEqual(
                captures["inner_runtime"].config["metadata"]["source"],
                "execution-context-test",
            )
            self.assertIn(
                inner_context_tool,
                captures["inner_runtime"].tools,
            )
            self.assertTrue(captures["nested"].endswith(large_value))
            self.assertTrue(captures["inner_state"]["messages"])
            self.assertTrue(captures["child_state"]["messages"])
            self.assertIn("child done", delegated["result"])
            parent_text = "\n".join(str(message.content) for message in result["messages"])
            self.assertNotIn(large_value, parent_text)
            self.assertEqual(len(parent_messages), 2)
            inner_start = next(
                item for item in callbacks.starts if item["name"] == "inner_context_tool"
            )
            self.assertEqual(inner_start["inputs"]["value"], "payload")
            self.assertIn(RUNTIME_PAYLOAD_TAG, inner_start["tags"])
            self.assertEqual(
                inner_start["metadata"]["source"],
                "execution-context-test",
            )
            task_start = next(
                item for item in callbacks.starts if item["name"] == "task"
            )
            self.assertIn(RUNTIME_PAYLOAD_TAG, task_start["tags"])
            self.assertFalse(
                any("Pydantic serializer warnings" in str(item.message) for item in caught)
            )

    async def test_async_only_bound_tool_rejects_sync_and_supports_async(self) -> None:
        @tool
        async def async_only(value: str) -> str:
            """Return a value asynchronously."""
            return f"async:{value}"

        context = create_mira_context({"async_only": async_only}, ())

        class State(TypedDict, total=False):
            value: str

        async def async_node(state: State, runtime: Runtime[MiraContext]) -> State:
            return {
                "value": await runtime.context.tools["async_only"].ainvoke(
                    {"value": state["value"]}
                )
            }

        graph = StateGraph(State, context_schema=MiraContext)
        graph.add_node("call", async_node)
        graph.add_edge(START, "call")
        graph.add_edge("call", END)
        result = await graph.compile().ainvoke({"value": "ok"}, context=context)
        self.assertEqual(result["value"], "async:ok")

        def sync_node(state: State, runtime: Runtime[MiraContext]) -> State:
            return {
                "value": runtime.context.tools["async_only"].invoke(
                    {"value": state["value"]}
                )
            }

        sync_graph = StateGraph(State, context_schema=MiraContext)
        sync_graph.add_node("call", sync_node)
        sync_graph.add_edge(START, "call")
        sync_graph.add_edge("call", END)
        with self.assertRaisesRegex(NotImplementedError, "does not support synchronous"):
            sync_graph.compile().invoke({"value": "no"}, context=context)

    def test_bound_tools_normalize_native_message_and_command_envelopes(self) -> None:
        @tool
        def message_result() -> ToolMessage:
            """Return a native ToolMessage envelope."""
            return ToolMessage(content="message value", tool_call_id="inner-message")

        @tool
        def command_result() -> Command:
            """Return a native Command envelope."""
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content="command value",
                            tool_call_id="inner-command",
                        )
                    ]
                }
            )

        context = create_mira_context(
            {
                "message_result": message_result,
                "command_result": command_result,
            },
            (),
        )

        class State(TypedDict, total=False):
            message: str
            command: str

        def call_envelopes(_state: State, runtime: Runtime[MiraContext]) -> State:
            return {
                "message": runtime.context.tools["message_result"].invoke({}),
                "command": runtime.context.tools["command_result"].invoke({}),
            }

        graph = StateGraph(State, context_schema=MiraContext)
        graph.add_node("call", call_envelopes)
        graph.add_edge(START, "call")
        graph.add_edge("call", END)
        result = graph.compile().invoke({}, context=context)

        self.assertEqual(result["message"], "message value")
        self.assertEqual(result["command"], "command value")


if __name__ == "__main__":
    unittest.main()
