"""Regression coverage for construction-time AgentMiddleware span policy."""

from __future__ import annotations

import asyncio
import importlib.util
import os
import tempfile
import threading
import unittest
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

from agent import factory as mira_factory
from agent.resources import build_resources
from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware, omit_payload
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool, tool
from runtime.runner import TurnResult
from tracing import bootstrap
from tracing.middleware_spans import middleware_span_policy

_WRAPPER_SUFFIXES = (
    ".wrap_model_call", ".awrap_model_call", ".wrap_tool_call", ".awrap_tool_call",
)
_NODE_HOOK_SUFFIXES = (
    ".before_agent", ".before_model", ".after_model", ".after_agent",
)


def _real_tracing_available() -> bool:
    try:
        return all(
            importlib.util.find_spec(name) is not None
            for name in ("langsmith", "openinference.instrumentation", "opentelemetry.sdk.trace")
        )
    except ModuleNotFoundError:
        return False


class _ToolCallingModel(FakeMessagesListChatModel):
    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[Any, AIMessage]:
        return self


@dataclass
class _TraceCapture:
    spans: list[Any]
    tool_calls: int
    subagent_text: str
    main_text: str
    main_model_provider: str
    subagent_messages: tuple[tuple[str, str], ...]
    main_messages: tuple[tuple[str, str], ...]

    @property
    def names(self) -> Counter[str]:
        return Counter(span.name for span in self.spans)

    @property
    def wrapper_names(self) -> Counter[str]:
        return Counter(
            {name: count for name, count in self.names.items() if name.endswith(_WRAPPER_SUFFIXES)}
        )

    @property
    def node_hook_names(self) -> Counter[str]:
        return Counter(
            {name: count for name, count in self.names.items() if name.endswith(_NODE_HOOK_SUFFIXES)}
        )


@contextmanager
def _raw_tracing() -> Iterator[Any]:
    import langsmith
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from tracing.semantic_processor import LangSmithOpenInferenceProcessor

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(LangSmithOpenInferenceProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    request_patch = patch.object(langsmith.Client, "request_with_retries", autospec=True)
    rest_request = request_patch.start()
    client = langsmith.Client(tracing_mode="otel", otel_tracer_provider=provider)
    try:
        with patch.dict(
            os.environ,
            {"LANGSMITH_TRACING": "true", "LANGSMITH_TRACING_MODE": "otel"},
        ):
            langsmith.configure(client=client, enabled=True)
            bootstrap._active_langsmith = langsmith
            bootstrap._active_client = client
            yield exporter
    finally:
        bootstrap._active_langsmith = None
        bootstrap._active_client = None
        langsmith.configure(client=None, enabled=None)
        client.close(timeout=1.0)
        request_patch.stop()
        provider.force_flush(timeout_millis=1_000)
        provider.shutdown()
    rest_request.assert_not_called()


async def _capture_mira_trace(mode: str) -> _TraceCapture:
    with _raw_tracing() as exporter:
        tool_effect = {"calls": 0}

        @tool
        def double(value: int) -> int:
            """Double one integer and record that execution reached the tool."""
            tool_effect["calls"] += 1
            return value * 2

        model = _ToolCallingModel(
            responses=[
                AIMessage(
                    content="Subagent construction is active.",
                    usage_metadata={"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
                ),
                AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "double", "args": {"value": 2},
                        "id": "call-double", "type": "tool_call",
                    }],
                    usage_metadata={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
                ),
                AIMessage(
                    content="The answer is 4.",
                    usage_metadata={"input_tokens": 6, "output_tokens": 4, "total_tokens": 10},
                ),
            ]
        )
        config = {
            "settings": {
                "tracing": {"middleware_spans": mode},
                "system": {"dynamic_subagents": {"enabled": True, "response_schema": False}},
                "hitl": {"tools": {"double": {"enabled": True, "always_allow": True}}},
            }
        }
        captured_subagents: list[dict[str, Any]] = []
        original_create_deep_agent = mira_factory.create_deep_agent

        def capture_create_deep_agent(*args: Any, **kwargs: Any) -> Any:
            captured_subagents.extend(kwargs.get("subagents") or [])
            return original_create_deep_agent(*args, **kwargs)

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            resources = build_resources(workspace, create_examples=False, config=config)
            metadata = {
                **resources.metadata,
                "tools": [{
                    "name": "double", "description": "Double one integer.", "source": "project",
                }],
            }
            resources = replace(
                resources, skills=[], memory=[], subagents=[], tools=[double], metadata=metadata,
            )
            with (
                patch.object(mira_factory, "get_llm", return_value=model),
                patch.object(mira_factory, "create_deep_agent", side_effect=capture_create_deep_agent),
            ):
                agent = mira_factory.build_agent(
                    config, workspace, checkpointer=None, resources=resources,
                )

            if len(captured_subagents) != 1 or "runnable" not in captured_subagents[0]:
                raise AssertionError("MIRA did not compile the dynamic subagent")
            subagent = captured_subagents[0]["runnable"]
            captured_results: dict[str, dict[str, Any]] = {}

            @bootstrap.trace_user_turn
            async def invoke_main_and_subagent(**kwargs: object) -> TurnResult:
                captured_results["subagent"] = await subagent.ainvoke(
                    {"messages": [HumanMessage("Report that the subagent is active.")]}
                )
                captured_results["main"] = await agent.ainvoke(
                    {"messages": [HumanMessage("Double 2.")]}
                )
                return TurnResult(
                    final_text="The answer is 4.", input_tokens=11, output_tokens=9, total_tokens=20,
                )

            await invoke_main_and_subagent(
                text="internal expanded text",
                display_text="Run a subagent, then double 2.",
            )
            subagent_message = captured_results["subagent"]["messages"][-1]
            main_message = captured_results["main"]["messages"][-1]

    spans = list(exporter.get_finished_spans())
    return _TraceCapture(
        spans=spans,
        tool_calls=tool_effect["calls"],
        subagent_text=str(subagent_message.content),
        main_text=str(main_message.content),
        main_model_provider=str(main_message.response_metadata.get("model_provider", "")),
        subagent_messages=tuple(
            (message.type, str(message.content)) for message in captured_results["subagent"]["messages"]
        ),
        main_messages=tuple(
            (message.type, str(message.content)) for message in captured_results["main"]["messages"]
        ),
    )


class _CoverageMiddleware(AgentMiddleware):
    def __init__(self, counters: Counter[str], sync_child: Callable[[], Any], async_child: Callable[[], Any]):
        self.counters = counters
        self.sync_child = sync_child
        self.async_child = async_child

    def before_agent(self, state: Any, runtime: Any) -> None:
        self.counters["before_agent"] += 1
        self.sync_child()

    async def abefore_agent(self, state: Any, runtime: Any) -> None:
        self.counters["abefore_agent"] += 1
        await self.async_child()

    def before_model(self, state: Any, runtime: Any) -> None:
        self.counters["before_model"] += 1

    async def abefore_model(self, state: Any, runtime: Any) -> None:
        self.counters["abefore_model"] += 1

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        self.counters["wrap_model_call"] += 1
        return handler(request)

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        self.counters["awrap_model_call"] += 1
        return await handler(request)

    def wrap_tool_call(self, request: Any, handler: Any) -> Any:
        self.counters["wrap_tool_call"] += 1
        return handler(request)

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        self.counters["awrap_tool_call"] += 1
        return await handler(request)

    def after_model(self, state: Any, runtime: Any) -> None:
        self.counters["after_model"] += 1

    async def aafter_model(self, state: Any, runtime: Any) -> None:
        self.counters["aafter_model"] += 1

    def after_agent(self, state: Any, runtime: Any) -> None:
        self.counters["after_agent"] += 1

    async def aafter_agent(self, state: Any, runtime: Any) -> None:
        self.counters["aafter_agent"] += 1


@dataclass
class _HookCapture:
    spans: list[Any]
    counters: Counter[str]
    tool_calls: int
    sync_messages: tuple[tuple[str, str], ...]
    async_messages: tuple[tuple[str, str], ...]


async def _capture_all_hooks(mode: str) -> _HookCapture:
    import langsmith

    with _raw_tracing() as exporter:
        counters: Counter[str] = Counter()
        tool_effect = {"calls": 0}

        @langsmith.traceable(name="useful_before_sync", run_type="tool")
        def useful_before_sync() -> str:
            return "sync child"

        @langsmith.traceable(name="useful_before_async", run_type="tool")
        async def useful_before_async() -> str:
            return "async child"

        @tool
        def double(value: int) -> int:
            """Double one integer."""
            tool_effect["calls"] += 1
            return value * 2

        def responses(label: str) -> list[AIMessage]:
            return [
                AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "double", "args": {"value": 2},
                        "id": f"call-{label}", "type": "tool_call",
                    }],
                ),
                AIMessage(content=f"{label} answer"),
            ]

        middleware = _CoverageMiddleware(counters, useful_before_sync, useful_before_async)
        with middleware_span_policy(mode):
            sync_agent = create_agent(
                _ToolCallingModel(responses=responses("sync")), tools=[double], middleware=[middleware],
            )
            async_agent = create_agent(
                _ToolCallingModel(responses=responses("async")), tools=[double], middleware=[middleware],
            )

        results: dict[str, dict[str, Any]] = {}

        @bootstrap.trace_user_turn
        async def invoke_both(**kwargs: object) -> TurnResult:
            results["sync"] = sync_agent.invoke({"messages": [HumanMessage("Run sync.")]})
            results["async"] = await async_agent.ainvoke(
                {"messages": [HumanMessage("Run async.")]}
            )
            return TurnResult(final_text="done")

        await invoke_both(text="internal", display_text="Run both paths.")
    spans = list(exporter.get_finished_spans())

    return _HookCapture(
        spans=spans,
        counters=counters,
        tool_calls=tool_effect["calls"],
        sync_messages=tuple(
            (message.type, str(message.content)) for message in results["sync"]["messages"]
        ),
        async_messages=tuple(
            (message.type, str(message.content)) for message in results["async"]["messages"]
        ),
    )


def _assert_valid_topology(test: unittest.TestCase, spans: list[Any]) -> None:
    test.assertEqual(len({span.context.trace_id for span in spans}), 1)
    roots = [span for span in spans if span.parent is None]
    test.assertEqual([span.name for span in roots], ["MIRA Turn"])
    span_ids = {span.context.span_id for span in spans}
    test.assertEqual(
        [span.name for span in spans if span.parent is not None and span.parent.span_id not in span_ids],
        [],
    )


@unittest.skipUnless(_real_tracing_available(), "requires the mira[tracing] extra")
class MiddlewareSpanConstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.full = asyncio.run(_capture_mira_trace("full"))
        cls.hidden = asyncio.run(_capture_mira_trace("hidden"))
        cls.full_hooks = asyncio.run(_capture_all_hooks("full"))
        cls.hidden_hooks = asyncio.run(_capture_all_hooks("hidden"))

    def test_full_preserves_native_framework_tracing(self) -> None:
        self.assertTrue(self.full.wrapper_names)
        self.assertTrue(self.full.node_hook_names)
        names = self.full.names
        self.assertEqual(names["MIRA Turn"], 1)
        self.assertEqual(names["LangGraph"], 1)
        self.assertEqual(names["general-purpose"], 1)
        self.assertEqual(names["_ToolCallingModel"], 3)
        self.assertEqual(names["double"], 1)
        _assert_valid_topology(self, self.full.spans)

    def test_hidden_removes_every_middleware_hook_span(self) -> None:
        hidden_names = Counter(span.name for span in self.hidden_hooks.spans)
        full_names = Counter(span.name for span in self.full_hooks.spans)
        expected = {
            "_CoverageMiddleware.before_agent", "_CoverageMiddleware.before_model",
            "_CoverageMiddleware.wrap_model_call", "_CoverageMiddleware.awrap_model_call",
            "_CoverageMiddleware.wrap_tool_call", "_CoverageMiddleware.awrap_tool_call",
            "_CoverageMiddleware.after_model", "_CoverageMiddleware.after_agent",
        }
        self.assertTrue(expected.issubset(full_names))
        self.assertFalse(expected.intersection(hidden_names))
        self.assertEqual(self.hidden.wrapper_names, Counter())
        self.assertEqual(self.hidden.node_hook_names, Counter())

    def test_hidden_preserves_main_subagent_tool_and_model_behavior(self) -> None:
        self.assertEqual(self.hidden.tool_calls, self.full.tool_calls)
        self.assertEqual(self.hidden.tool_calls, 1)
        self.assertEqual(self.hidden.subagent_messages, self.full.subagent_messages)
        self.assertEqual(self.hidden.main_messages, self.full.main_messages)
        self.assertEqual(self.hidden.subagent_text, "Subagent construction is active.")
        self.assertEqual(self.hidden.main_text, "The answer is 4.")
        self.assertEqual(self.hidden.main_model_provider, "anyllm")
        names = self.hidden.names
        self.assertEqual(names["MIRA Turn"], 1)
        self.assertEqual(names["LangGraph"], 1)
        self.assertEqual(names["general-purpose"], 1)
        self.assertEqual(names["_ToolCallingModel"], 3)
        self.assertEqual(names["double"], 1)
        self.assertTrue(
            {"CHAIN", "LLM", "TOOL"}.issubset(
                {span.attributes.get("openinference.span.kind") for span in self.hidden.spans}
            )
        )
        _assert_valid_topology(self, self.hidden.spans)

    def test_sync_async_hooks_and_writers_keep_behavior(self) -> None:
        self.assertEqual(self.hidden_hooks.counters, self.full_hooks.counters)
        self.assertEqual(
            self.hidden_hooks.counters,
            Counter({
                "before_model": 2, "wrap_model_call": 2, "after_model": 2,
                "abefore_model": 2, "awrap_model_call": 2, "aafter_model": 2,
                "before_agent": 1, "wrap_tool_call": 1, "after_agent": 1,
                "abefore_agent": 1, "awrap_tool_call": 1, "aafter_agent": 1,
            }),
        )
        self.assertEqual(self.hidden_hooks.tool_calls, 2)
        self.assertEqual(self.hidden_hooks.sync_messages, self.full_hooks.sync_messages)
        self.assertEqual(self.hidden_hooks.async_messages, self.full_hooks.async_messages)
        self.assertEqual(self.hidden_hooks.sync_messages[-1], ("ai", "sync answer"))
        self.assertEqual(self.hidden_hooks.async_messages[-1], ("ai", "async answer"))

    def test_useful_children_survive_hidden_node_without_orphans(self) -> None:
        by_name = {span.name: span for span in self.hidden_hooks.spans}
        by_id = {span.context.span_id: span for span in self.hidden_hooks.spans}
        for name in ("useful_before_sync", "useful_before_async"):
            child = by_name[name]
            self.assertIsNotNone(child.parent)
            parent = by_id[child.parent.span_id]
            self.assertNotIn(parent.name, {
                "_CoverageMiddleware.before_agent", "_CoverageMiddleware.before_model",
                "_CoverageMiddleware.after_model", "_CoverageMiddleware.after_agent",
            })
        _assert_valid_topology(self, self.hidden_hooks.spans)


class MiddlewareSpanPolicyLifecycleTests(unittest.TestCase):
    def test_traceable_patch_restores_after_success_and_failure(self) -> None:
        import langchain.agents.factory as langchain_factory
        from langgraph.graph.state import CompiledStateGraph

        original = langchain_factory.traceable
        original_attach_node = CompiledStateGraph.attach_node
        with middleware_span_policy("hidden"):
            self.assertIsNot(langchain_factory.traceable, original)
            self.assertIsNot(CompiledStateGraph.attach_node, original_attach_node)
        self.assertIs(langchain_factory.traceable, original)
        self.assertIs(CompiledStateGraph.attach_node, original_attach_node)

        with self.assertRaisesRegex(RuntimeError, "construction failed"):
            with middleware_span_policy("hidden"):
                self.assertIsNot(langchain_factory.traceable, original)
                raise RuntimeError("construction failed")
        self.assertIs(langchain_factory.traceable, original)
        self.assertIs(CompiledStateGraph.attach_node, original_attach_node)

    def test_full_is_a_true_no_op(self) -> None:
        import langchain.agents.factory as langchain_factory
        from langgraph.graph.state import CompiledStateGraph

        traceable = langchain_factory.traceable
        attach_node = CompiledStateGraph.attach_node
        resolved_transform = langchain_factory._resolved_transform
        node_trace_policy = langchain_factory._node_trace_policy
        with patch("tracing.middleware_spans.TRACE_MIDDLEWARE_INPUTS", False):
            with middleware_span_policy("full"):
                self.assertIs(langchain_factory.traceable, traceable)
                self.assertIs(CompiledStateGraph.attach_node, attach_node)
                self.assertIs(langchain_factory._resolved_transform, resolved_transform)
                self.assertIs(langchain_factory._node_trace_policy, node_trace_policy)

    def test_middleware_inputs_are_restored_by_default(self) -> None:
        import langchain.agents.factory as langchain_factory
        from deepagents.middleware.rubric import RubricMiddleware

        upstream = RubricMiddleware.trace_policy
        self.assertIs(upstream.process_inputs, omit_payload)
        payload = {"messages": ["visible"]}
        with middleware_span_policy("full"):
            wrap_processor = langchain_factory._resolved_transform(upstream, "process_inputs")
            node_policy = langchain_factory._node_trace_policy(upstream)
        self.assertEqual(wrap_processor(payload), payload)
        self.assertEqual(node_policy.process_inputs(payload), payload)

    def test_disabling_input_toggle_preserves_upstream_omission(self) -> None:
        import langchain.agents.factory as langchain_factory
        from deepagents.middleware.rubric import RubricMiddleware

        upstream = RubricMiddleware.trace_policy
        self.assertIs(upstream.process_inputs, omit_payload)
        payload = {"messages": ["hidden"]}
        with patch("tracing.middleware_spans.TRACE_MIDDLEWARE_INPUTS", False):
            with middleware_span_policy("full"):
                wrap_processor = langchain_factory._resolved_transform(upstream, "process_inputs")
                node_policy = langchain_factory._node_trace_policy(upstream)
        self.assertEqual(wrap_processor(payload), {})
        self.assertEqual(node_policy.process_inputs(payload), {})

    def test_span_visibility_and_input_payload_policies_are_independent(self) -> None:
        import langchain.agents.factory as langchain_factory
        from deepagents.middleware.rubric import RubricMiddleware
        from langgraph.graph.state import CompiledStateGraph

        traceable = langchain_factory.traceable
        attach_node = CompiledStateGraph.attach_node
        upstream = RubricMiddleware.trace_policy
        with middleware_span_policy("hidden"):
            self.assertIsNot(langchain_factory.traceable, traceable)
            self.assertIsNot(CompiledStateGraph.attach_node, attach_node)
            processor = langchain_factory._resolved_transform(upstream, "process_inputs")
        self.assertEqual(processor({"input": "visible"}), {"input": "visible"})

    def test_hidden_construction_scopes_are_serialized(self) -> None:
        first_entered = threading.Event()
        release_first = threading.Event()
        second_entered = threading.Event()

        def first() -> None:
            with middleware_span_policy("hidden"):
                first_entered.set()
                release_first.wait(timeout=2)

        def second() -> None:
            first_entered.wait(timeout=2)
            with middleware_span_policy("hidden"):
                second_entered.set()

        with ThreadPoolExecutor(max_workers=2) as pool:
            first_future = pool.submit(first)
            second_future = pool.submit(second)
            self.assertTrue(first_entered.wait(timeout=2))
            self.assertFalse(second_entered.wait(timeout=0.1))
            release_first.set()
            first_future.result(timeout=2)
            second_future.result(timeout=2)
        self.assertTrue(second_entered.is_set())


if __name__ == "__main__":
    unittest.main()
