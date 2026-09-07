"""Tests for schema-free compiled subagent construction."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from deepagents.backends import StateBackend
from deepagents.middleware.filesystem import FilesystemPermission
from deepagents.middleware.subagents import SUBAGENT_RESPONSE_FORMAT_CONFIG_KEY, SubAgentMiddleware
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from agent.subagents.compilation import compile_dynamic_subagents


class SubagentCompilationTests(unittest.TestCase):
    """Compiled mode should preserve subagent capabilities and block schemas."""

    def test_injects_and_compiles_full_general_purpose_subagent(self) -> None:
        permissions = [FilesystemPermission(operations=["read", "write"], paths=["/**"])]
        interrupts = {"write_file": {"allowed_decisions": ["approve", "reject"]}}

        with (
            patch("agent.subagents.compilation.resolve_subagent_model", side_effect=lambda value: value),
            patch("agent.subagents.compilation.create_summarization_middleware", return_value="summary"),
            patch("agent.subagents.compilation.create_sub_agent", side_effect=lambda spec: spec) as create,
        ):
            compiled = compile_dynamic_subagents(
                [],
                model="parent-model",
                tools=["parent-tool"],
                backend=StateBackend(),
                skills=["/skills/project/"],
                permissions=permissions,
                interrupt_on=interrupts,
                enable_todos=True,
            )

        self.assertEqual([item["name"] for item in compiled], ["general-purpose"])
        materialized = create.call_args.args[0]
        self.assertEqual(materialized["model"], "parent-model")
        self.assertEqual(materialized["tools"], ["parent-tool"])
        self.assertEqual(materialized["skills"], ["/skills/project/"])
        self.assertEqual(materialized["interrupt_on"], interrupts)
        self.assertIsInstance(materialized["middleware"][0], TodoListMiddleware)
        self.assertEqual(materialized["middleware"][1]._permissions, permissions)
        self.assertEqual(materialized["middleware"][2], "summary")
        self.assertEqual(compiled[0]["runnable"], materialized)

    def test_compiles_raw_specs_and_preserves_compiled_and_async_entries(self) -> None:
        custom_middleware = object()
        static_schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        raw = {
            "name": "researcher",
            "description": "Researches",
            "system_prompt": "Research carefully.",
            "model": "child-model",
            "tools": ["child-tool"],
            "middleware": [custom_middleware],
            "skills": ["/skills/research/"],
            "permissions": [],
            "interrupt_on": {},
            "response_format": static_schema,
        }
        existing_compiled = {
            "name": "compiled",
            "description": "Already compiled",
            "runnable": object(),
        }
        asynchronous = {"name": "remote", "description": "Remote", "graph_id": "graph"}

        with (
            patch(
                "agent.subagents.compilation.resolve_subagent_model",
                side_effect=lambda value: f"resolved:{value}",
            ),
            patch("agent.subagents.compilation.create_summarization_middleware", return_value="summary"),
            patch("agent.subagents.compilation.create_sub_agent", side_effect=lambda spec: spec) as create,
        ):
            compiled = compile_dynamic_subagents(
                [raw, existing_compiled, asynchronous],
                model="parent-model",
                tools=["parent-tool"],
                backend=StateBackend(),
                skills=None,
                permissions=None,
                interrupt_on={"task": True},
            )

        self.assertEqual([item["name"] for item in compiled], ["general-purpose", "researcher", "compiled", "remote"])
        self.assertIs(compiled[2], existing_compiled)
        self.assertIs(compiled[3], asynchronous)
        researcher = next(call.args[0] for call in create.call_args_list if call.args[0]["name"] == "researcher")
        self.assertEqual(researcher["model"], "resolved:child-model")
        self.assertEqual(researcher["tools"], ["child-tool"])
        self.assertIs(researcher["middleware"][-1], custom_middleware)
        self.assertEqual(researcher["response_format"], static_schema)
        self.assertEqual(researcher["interrupt_on"], {})
        self.assertFalse(any(isinstance(item, TodoListMiddleware) for item in researcher["middleware"]))

    def test_fork_specs_remain_declarative_for_native_deepagents_construction(self) -> None:
        fork = {
            "name": "reviewer",
            "description": "Reviews with parent context",
            "system_prompt": "Review carefully.",
            "mode": "fork",
            "model": "child-model",
            "tools": ["child-tool"],
        }
        isolated = {
            "name": "researcher",
            "description": "Researches in isolation",
            "system_prompt": "Research carefully.",
        }

        with (
            patch("agent.subagents.compilation.resolve_subagent_model", side_effect=lambda value: value),
            patch("agent.subagents.compilation.create_summarization_middleware", return_value="summary"),
            patch("agent.subagents.compilation.create_sub_agent", side_effect=lambda spec: spec) as create,
        ):
            compiled = compile_dynamic_subagents(
                [fork, isolated],
                model="parent-model",
                tools=["parent-tool"],
                backend=StateBackend(),
                skills=None,
                permissions=None,
                interrupt_on=None,
            )

        reviewer = next(item for item in compiled if item["name"] == "reviewer")
        researcher = next(item for item in compiled if item["name"] == "researcher")
        self.assertIs(reviewer, fork)
        self.assertEqual(reviewer["mode"], "fork")
        self.assertEqual(reviewer["model"], "child-model")
        self.assertEqual(reviewer["tools"], ["child-tool"])
        self.assertIn("runnable", researcher)
        self.assertNotIn("reviewer", [call.args[0]["name"] for call in create.call_args_list])

    def test_deepagents_native_fork_inherits_messages_while_default_isolated_does_not(self) -> None:
        invocations: list[list[object]] = []

        def compiled_subagent(_spec: dict[str, object], **_kwargs: object) -> RunnableLambda:
            return RunnableLambda(
                lambda state: invocations.append(list(state["messages"]))
                or {"messages": [AIMessage("done")]}
            )

        parent_messages = [HumanMessage("original request"), AIMessage("parent context")]
        runtime = SimpleNamespace(
            config={},
            state={"messages": parent_messages},
            tool_call_id="task-call",
        )

        with patch("deepagents.middleware.subagents.create_sub_agent", side_effect=compiled_subagent):
            forked = SubAgentMiddleware(
                backend=StateBackend(),
                subagents=[
                    {
                        "name": "reviewer",
                        "description": "Reviews",
                        "system_prompt": "Review carefully.",
                        "mode": "fork",
                    }
                ],
            )
            forked.tools[0].func(
                description="Check the answer",
                subagent_type="reviewer",
                runtime=runtime,
            )
            isolated = SubAgentMiddleware(
                backend=StateBackend(),
                subagents=[
                    {
                        "name": "reviewer",
                        "description": "Reviews",
                        "system_prompt": "Review carefully.",
                    }
                ],
            )
            isolated.tools[0].func(
                description="Check the answer",
                subagent_type="reviewer",
                runtime=runtime,
            )

        fork_contents = [message.content for message in invocations[0]]
        self.assertEqual(fork_contents[:2], ["original request", "parent context"])
        self.assertIn("Check the answer", fork_contents[-1])
        self.assertEqual([message.content for message in invocations[1]], ["Check the answer"])

    def test_native_raw_fork_accepts_a_dynamic_response_schema(self) -> None:
        response_schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        response_formats: list[object] = []

        def compiled_subagent(
            _spec: dict[str, object],
            *,
            response_format: object = None,
            **_kwargs: object,
        ) -> RunnableLambda:
            response_formats.append(response_format)
            return RunnableLambda(lambda _state: {"messages": [AIMessage("done")]})

        runtime = SimpleNamespace(
            config={"configurable": {SUBAGENT_RESPONSE_FORMAT_CONFIG_KEY: response_schema}},
            state={"messages": [HumanMessage("parent context")]},
            tool_call_id="task-call",
        )
        with patch("deepagents.middleware.subagents.create_sub_agent", side_effect=compiled_subagent):
            middleware = SubAgentMiddleware(
                backend=StateBackend(),
                subagents=[
                    {
                        "name": "reviewer",
                        "description": "Reviews",
                        "system_prompt": "Review carefully.",
                        "mode": "fork",
                    }
                ],
            )
            middleware.tools[0].func(
                description="Return structured feedback",
                subagent_type="reviewer",
                runtime=runtime,
            )

        self.assertEqual(response_formats, [None, response_schema])

    def test_existing_general_purpose_is_not_duplicated(self) -> None:
        existing = {"name": "general-purpose", "description": "Custom", "runnable": object()}

        compiled = compile_dynamic_subagents(
            [existing],
            model="parent-model",
            tools=[],
            backend=StateBackend(),
            skills=[],
            permissions=[],
            interrupt_on=None,
        )

        self.assertEqual(compiled, [existing])
        self.assertIs(compiled[0], existing)

    def test_existing_raw_general_purpose_is_compiled_without_duplicate(self) -> None:
        existing = {
            "name": "general-purpose",
            "description": "Custom",
            "system_prompt": "Custom prompt",
        }

        with (
            patch("agent.subagents.compilation.resolve_subagent_model", side_effect=lambda value: value),
            patch("agent.subagents.compilation.create_summarization_middleware", return_value="summary"),
            patch("agent.subagents.compilation.create_sub_agent", side_effect=lambda spec: spec),
        ):
            compiled = compile_dynamic_subagents(
                [existing],
                model="parent-model",
                tools=[],
                backend=StateBackend(),
                skills=[],
                permissions=[],
                interrupt_on=None,
            )

        self.assertEqual([item["name"] for item in compiled], ["general-purpose"])
        self.assertEqual(compiled[0]["runnable"]["system_prompt"], "Custom prompt")

    def test_deepagents_rejects_schema_before_invoking_compiled_worker(self) -> None:
        calls = []
        runnable = RunnableLambda(lambda state: calls.append(state) or {"messages": [AIMessage("done")]})
        middleware = SubAgentMiddleware(
            backend=StateBackend(),
            subagents=[{"name": "general-purpose", "description": "Full worker", "runnable": runnable}],
        )
        task = middleware.tools[0]
        runtime = SimpleNamespace(
            config={"configurable": {SUBAGENT_RESPONSE_FORMAT_CONFIG_KEY: {"type": "object"}}},
            state={"messages": []},
            tool_call_id="task-call",
        )

        with self.assertRaisesRegex(ValueError, "response_schema cannot be used with compiled subagent"):
            task.func(description="Judge the haiku", subagent_type="general-purpose", runtime=runtime)

        self.assertEqual(calls, [])

    def test_deepagents_invokes_compiled_worker_without_schema(self) -> None:
        calls = []
        runnable = RunnableLambda(lambda state: calls.append(state) or {"messages": [AIMessage("done")]})
        middleware = SubAgentMiddleware(
            backend=StateBackend(),
            subagents=[{"name": "general-purpose", "description": "Full worker", "runnable": runnable}],
        )
        task = middleware.tools[0]
        runtime = SimpleNamespace(config={}, state={"messages": []}, tool_call_id="task-call")

        result = task.func(description="Judge the haiku", subagent_type="general-purpose", runtime=runtime)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["messages"][0].content, "Judge the haiku")
        self.assertEqual(result.update["messages"][0].content, "done")


if __name__ == "__main__":
    unittest.main()
