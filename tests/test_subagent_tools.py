"""Tests for declarative and hybrid raw-subagent tool allowlists."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from deepagents.backends import StateBackend
from langchain_core.tools import tool

from agent.middleware.model_tool_visibility import ModelToolVisibilityMiddleware
from agent import factory
from agent.resources import ResourceBundle
from agent.subagents.discovery import (
    DiscoveredSubagent,
    SubagentDiscovery,
    resolve_subagent_tool_allowlists,
)
from agent.subagents.compilation import compile_dynamic_subagents


@tool
def calculator(value: int) -> int:
    """Return a test calculation."""
    return value


@tool("mcp__server__search")
def mcp_search(query: str) -> str:
    """Search a test MCP server."""
    return query


@tool
def custom_tool(value: str) -> str:
    """Return a custom value."""
    return value


def raw(name: str, **values: Any) -> dict[str, Any]:
    return {
        "name": name,
        "description": f"{name} description",
        "system_prompt": f"{name} prompt",
        **values,
    }


def discovery_for(*specs: dict[str, Any]) -> SubagentDiscovery:
    return SubagentDiscovery(
        tuple(
            DiscoveredSubagent(
                name=spec["name"],
                description=spec["description"],
                spec=spec,
                path=f"/.mira/subagents/{spec['name']}.py",
                source="project",
            )
            for spec in specs
        )
    )


def visibility(spec: dict[str, Any]) -> ModelToolVisibilityMiddleware:
    return next(
        item
        for item in spec.get("middleware", [])
        if isinstance(item, ModelToolVisibilityMiddleware)
    )


def resource_bundle(*specs: dict[str, Any]) -> ResourceBundle:
    discovery = discovery_for(*specs)
    backend = StateBackend()
    return ResourceBundle(
        backend=backend,
        project_backend=backend,
        skills=[],
        memory=[],
        subagents=list(specs),
        tools=[calculator],
        metadata={
            "memories": [],
            "skills": [],
            "subagents": [item.display_item() for item in discovery.items],
            "tools": [
                {
                    "name": "calculator",
                    "description": "Calculate",
                    "source": "project",
                }
            ],
        },
        tool_failures=[],
        subagent_discovery=discovery,
        issues=[],
    )


def factory_patches(agent: Any) -> tuple[Any, ...]:
    return (
        patch("agent.factory.get_llm", return_value="model"),
        patch("agent.middleware.builder.CodeInterpreterMiddleware", return_value="code"),
        patch(
            "agent.middleware.builder.create_mira_summarization_middleware",
            return_value="auto-summary",
        ),
        patch(
            "agent.middleware.builder.create_mira_summarization_tool_middleware",
            return_value="summary",
        ),
        patch("agent.factory.create_deep_agent", return_value=agent),
    )


class SubagentToolAllowlistTests(unittest.TestCase):
    builtins = {"ls", "read_file", "write_file", "edit_file", "glob", "grep"}

    def test_omitted_tools_preserves_inheritance_without_filtering(self) -> None:
        spec = raw("researcher")

        resolved, issues = resolve_subagent_tool_allowlists(
            [spec], [calculator], self.builtins, discovery_for(spec)
        )

        self.assertEqual(resolved, [spec])
        self.assertIs(resolved[0], spec)
        self.assertNotIn("middleware", resolved[0])
        self.assertEqual(issues, [])

    def test_empty_tools_creates_an_explicit_empty_surface(self) -> None:
        spec = raw("quiet", tools=[])

        resolved, issues = resolve_subagent_tool_allowlists(
            [spec], [calculator], self.builtins, discovery_for(spec)
        )

        self.assertEqual(resolved[0]["tools"], [])
        self.assertEqual(visibility(resolved[0]).allowed_tools, set())
        self.assertEqual(issues, [])

    def test_builtin_only_declaration_registers_no_duplicate_objects(self) -> None:
        spec = raw("searcher", tools=["grep", "ls"])

        resolved, issues = resolve_subagent_tool_allowlists(
            [spec], [calculator], self.builtins, discovery_for(spec)
        )

        self.assertEqual(resolved[0]["tools"], [])
        self.assertEqual(visibility(resolved[0]).allowed_tools, {"grep", "ls"})
        self.assertEqual(issues, [])

    def test_declarative_project_and_mcp_tools_resolve_to_live_objects(self) -> None:
        spec = raw("researcher", tools=["calculator", "mcp__server__search"])

        resolved, issues = resolve_subagent_tool_allowlists(
            [spec], [calculator, mcp_search], self.builtins, discovery_for(spec)
        )

        self.assertEqual(resolved[0]["tools"], [calculator, mcp_search])
        self.assertEqual(
            visibility(resolved[0]).allowed_tools,
            {"calculator", "mcp__server__search"},
        )
        self.assertEqual(issues, [])

    def test_hybrid_declaration_preserves_concrete_tools(self) -> None:
        spec = raw(
            "hybrid",
            tools=["grep", "mcp__server__search", custom_tool],
        )

        resolved, issues = resolve_subagent_tool_allowlists(
            [spec], [mcp_search], self.builtins, discovery_for(spec)
        )

        self.assertEqual(resolved[0]["tools"], [mcp_search, custom_tool])
        self.assertEqual(
            visibility(resolved[0]).allowed_tools,
            {"grep", "mcp__server__search", "custom_tool"},
        )
        self.assertEqual(issues, [])

    def test_concrete_only_declaration_preserves_objects(self) -> None:
        spec = raw("concrete", tools=[custom_tool])

        resolved, issues = resolve_subagent_tool_allowlists(
            [spec], [calculator], self.builtins, discovery_for(spec)
        )

        self.assertEqual(resolved[0]["tools"], [custom_tool])
        self.assertEqual(visibility(resolved[0]).allowed_tools, {"custom_tool"})
        self.assertEqual(issues, [])

    def test_concrete_provider_tool_dicts_are_preserved(self) -> None:
        provider_tool = {"type": "web_search_preview"}
        function_tool = {
            "type": "function",
            "function": {"name": "lookup", "description": "Look up a value"},
        }
        spec = raw("provider", tools=[provider_tool, function_tool])

        resolved, issues = resolve_subagent_tool_allowlists(
            [spec], [calculator], self.builtins, discovery_for(spec)
        )

        self.assertEqual(resolved[0]["tools"], [provider_tool, function_tool])
        self.assertEqual(
            visibility(resolved[0]).allowed_tools,
            {"web_search_preview", "lookup"},
        )
        self.assertEqual(issues, [])

    def test_unknown_reference_disables_only_affected_subagent(self) -> None:
        broken = raw("broken", tools=["mcp__missing__search"])
        healthy = raw("healthy", tools=["grep"])

        resolved, issues = resolve_subagent_tool_allowlists(
            [broken, healthy], [calculator], self.builtins, discovery_for(broken, healthy)
        )

        self.assertEqual([item["name"] for item in resolved], ["healthy"])
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].category, "TOOL")
        self.assertIn("mcp__missing__search", issues[0].details)
        self.assertIn("broken.py", issues[0].location)

    def test_disabled_subagent_does_not_generate_dependency_issue(self) -> None:
        disabled = raw("disabled", tools=["missing"])
        enabled = raw("enabled", tools=["grep"])
        discovery = discovery_for(disabled, enabled)

        resolved, issues = resolve_subagent_tool_allowlists(
            [enabled], [calculator], self.builtins, discovery
        )

        self.assertEqual([item["name"] for item in resolved], ["enabled"])
        self.assertEqual(issues, [])

    def test_globally_excluded_tool_cannot_be_restored_by_allowlist(self) -> None:
        declarative = raw("declarative", tools=["write_file"])
        concrete = raw("concrete", tools=[custom_tool])

        declarative_resolved, declarative_issues = resolve_subagent_tool_allowlists(
            [declarative],
            [calculator],
            self.builtins,
            discovery_for(declarative),
            {"write_file"},
        )
        concrete_resolved, concrete_issues = resolve_subagent_tool_allowlists(
            [concrete],
            [calculator],
            self.builtins,
            discovery_for(concrete),
            {"custom_tool"},
        )

        self.assertEqual(declarative_resolved, [])
        self.assertIn("write_file", declarative_issues[0].details)
        self.assertEqual(concrete_resolved, [])
        self.assertIn("custom_tool", concrete_issues[0].details)

    def test_duplicate_and_ambiguous_names_disable_subagent(self) -> None:
        duplicate = raw("duplicate", tools=["calculator", calculator])
        ambiguous = raw("ambiguous", tools=["grep"])
        shadow_grep = {"name": "grep", "description": "shadow"}

        duplicate_resolved, duplicate_issues = resolve_subagent_tool_allowlists(
            [duplicate], [calculator], self.builtins, discovery_for(duplicate)
        )
        ambiguous_resolved, ambiguous_issues = resolve_subagent_tool_allowlists(
            [ambiguous], [shadow_grep], self.builtins, discovery_for(ambiguous)
        )

        self.assertEqual(duplicate_resolved, [])
        self.assertIn("Duplicate resolved names: calculator", duplicate_issues[0].details)
        self.assertEqual(ambiguous_resolved, [])
        self.assertIn("Ambiguous references: grep", ambiguous_issues[0].details)

    def test_resolved_allowlist_is_preserved_by_fallback_compilation(self) -> None:
        spec = raw("searcher", tools=["grep", custom_tool])
        resolved, issues = resolve_subagent_tool_allowlists(
            [spec], [calculator], self.builtins, discovery_for(spec)
        )

        with (
            patch("agent.subagents.compilation.resolve_subagent_model", side_effect=lambda value: value),
            patch("agent.subagents.compilation.create_summarization_middleware", return_value="summary"),
            patch("agent.subagents.compilation.create_sub_agent", side_effect=lambda value: value) as create,
        ):
            compile_dynamic_subagents(
                resolved,
                model="parent-model",
                tools=[calculator],
                backend=StateBackend(),
                skills=[],
                permissions=[],
                interrupt_on=None,
            )

        materialized = next(
            call.args[0]
            for call in create.call_args_list
            if call.args[0]["name"] == "searcher"
        )
        self.assertEqual(materialized["tools"], [custom_tool])
        self.assertEqual(visibility(materialized).allowed_tools, {"grep", "custom_tool"})
        self.assertEqual(issues, [])

    def test_factory_resolves_live_project_and_mcp_tools_for_normal_path(self) -> None:
        spec = raw(
            "searcher",
            tools=["grep", "calculator", "mcp__server__search"],
        )
        resources = resource_bundle(spec)
        manager = SimpleNamespace(
            tools_for_mode=lambda _settings, planning: (
                [mcp_search],
                [
                    {
                        "name": "mcp__server__search",
                        "description": "Search",
                        "source": "mcp",
                        "server": "server",
                        "original_name": "search",
                    }
                ],
            )
        )
        agent = type("Agent", (), {})()
        patches = factory_patches(agent)

        with patches[0], patches[1], patches[2], patches[3], patches[4] as create:
            built = factory.build_agent(
                {}, ".", "checkpointer", resources=resources, mcp_manager=manager
            )

        passed = create.call_args.kwargs["subagents"][0]
        self.assertIs(built, agent)
        self.assertEqual(passed["tools"], [calculator, mcp_search])
        self.assertEqual(
            visibility(passed).allowed_tools,
            {"grep", "calculator", "mcp__server__search"},
        )

    def test_factory_keeps_running_and_reports_broken_optional_subagent(self) -> None:
        broken = raw("broken", tools=["mcp__missing__search"])
        healthy = raw("healthy", tools=["grep"])
        resources = resource_bundle(broken, healthy)
        agent = type("Agent", (), {})()
        patches = factory_patches(agent)

        with patches[0], patches[1], patches[2], patches[3], patches[4] as create:
            built = factory.build_agent({}, ".", "checkpointer", resources=resources)

        self.assertIs(built, agent)
        self.assertEqual(
            [item["name"] for item in create.call_args.kwargs["subagents"]],
            ["healthy"],
        )
        self.assertEqual(len(agent.mira_resource_issues), 1)
        self.assertIn("mcp__missing__search", agent.mira_resource_issues[0].details)

    def test_factory_passes_identically_resolved_specs_to_fallback_compiler(self) -> None:
        spec = raw("searcher", tools=["grep", "calculator"])
        resources = resource_bundle(spec)
        compiled = [{"name": "searcher", "description": "compiled", "runnable": object()}]
        agent = type("Agent", (), {})()
        patches = factory_patches(agent)
        config = {
            "settings": {
                "system": {
                    "dynamic_subagents": {"enabled": True, "response_schema": False}
                }
            }
        }

        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patch("agent.factory.compile_dynamic_subagents", return_value=compiled) as compile_subagents,
        ):
            factory.build_agent(config, ".", "checkpointer", resources=resources)

        passed = compile_subagents.call_args.args[0]
        self.assertEqual(passed[0]["tools"], [calculator])
        self.assertEqual(visibility(passed[0]).allowed_tools, {"grep", "calculator"})


if __name__ == "__main__":
    unittest.main()
