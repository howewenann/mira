"""Tests for schema-free compiled subagent construction."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from deepagents.backends import StateBackend
from deepagents.middleware.filesystem import FilesystemPermission
from deepagents.middleware.subagents import SUBAGENT_RESPONSE_FORMAT_CONFIG_KEY, SubAgentMiddleware
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from agent.subagent_compilation import compile_dynamic_subagents


class SubagentCompilationTests(unittest.TestCase):
    """Compiled mode should preserve subagent capabilities and block schemas."""

    def test_injects_and_compiles_full_general_purpose_subagent(self) -> None:
        permissions = [FilesystemPermission(operations=["read", "write"], paths=["/**"])]
        interrupts = {"write_file": {"allowed_decisions": ["approve", "reject"]}}

        with (
            patch("agent.subagent_compilation.resolve_model", side_effect=lambda value: value),
            patch("agent.subagent_compilation.create_summarization_middleware", return_value="summary"),
            patch("agent.subagent_compilation.create_sub_agent", side_effect=lambda spec: spec) as create,
        ):
            compiled = compile_dynamic_subagents(
                [],
                model="parent-model",
                tools=["parent-tool"],
                backend=StateBackend(),
                skills=["/skills/project/"],
                permissions=permissions,
                interrupt_on=interrupts,
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
            patch("agent.subagent_compilation.resolve_model", side_effect=lambda value: f"resolved:{value}"),
            patch("agent.subagent_compilation.create_summarization_middleware", return_value="summary"),
            patch("agent.subagent_compilation.create_sub_agent", side_effect=lambda spec: spec) as create,
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
            patch("agent.subagent_compilation.resolve_model", side_effect=lambda value: value),
            patch("agent.subagent_compilation.create_summarization_middleware", return_value="summary"),
            patch("agent.subagent_compilation.create_sub_agent", side_effect=lambda spec: spec),
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
