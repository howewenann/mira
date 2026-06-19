"""Tests for MIRA resource discovery and layering."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.context_overflow import ProviderContextOverflowMiddleware
from agent import factory
from agent.resources import build_resources
from ui import repl


class ResourceDiscoveryTests(unittest.TestCase):
    """Tests for default and project resource layering."""

    def test_launch_creates_project_examples_without_overwriting(self) -> None:
        """Missing project resource examples should be created once."""
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            memory = workspace / ".mira" / "memories" / "AGENTS.md"
            memory.parent.mkdir(parents=True)
            memory.write_text("custom memory", encoding="utf-8")

            build_resources(workspace)

            self.assertEqual(memory.read_text(encoding="utf-8"), "custom memory")
            self.assertTrue((workspace / ".mira" / "README.md").exists())
            self.assertTrue((workspace / ".mira" / "skills" / "example-skill" / "SKILL.md").exists())
            self.assertTrue((workspace / ".mira" / "subagents" / "example_subagent.py").exists())
            self.assertTrue((workspace / ".mira" / "tools" / "example_tool.py").exists())

    def test_default_memory_loads_without_project_memory(self) -> None:
        """The bundled AGENTS.md should load when project examples are skipped."""
        with tempfile.TemporaryDirectory() as directory:
            resources = build_resources(Path(directory), create_examples=False)

            self.assertEqual(resources.memory, ["/mira-defaults/memories/AGENTS.md"])
            self.assertEqual(resources.metadata["memories"][0]["source"], "default")

    def test_project_memory_replaces_default_by_filename(self) -> None:
        """A project memory with the same filename should replace the default."""
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            memory_dir = workspace / ".mira" / "memories"
            memory_dir.mkdir(parents=True)
            (memory_dir / "AGENTS.md").write_text("project memory", encoding="utf-8")
            (memory_dir / "soul.md").write_text("project-only memory", encoding="utf-8")

            resources = build_resources(workspace, create_examples=False)

            self.assertEqual(resources.memory, ["/.mira/memories/AGENTS.md", "/.mira/memories/soul.md"])
            self.assertEqual(
                resources.metadata["memories"],
                [
                    {
                        "name": "AGENTS.md",
                        "path": "/.mira/memories/AGENTS.md",
                        "source": "project",
                        "replaces": "default",
                    },
                    {
                        "name": "soul.md",
                        "path": "/.mira/memories/soul.md",
                        "source": "project",
                        "replaces": "",
                    },
                ],
            )

    def test_project_skill_replaces_default_by_name(self) -> None:
        """A project skill with the same frontmatter name should replace display metadata."""
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            skill_dir = workspace / ".mira" / "skills" / "custom-folder"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                """---
name: codebase-orientation
description: Project-specific orientation.
---

# Project Orientation
""",
                encoding="utf-8",
            )

            resources = build_resources(workspace, create_examples=False)

            self.assertEqual(resources.skills, ["/mira-defaults/skills", "/.mira/skills"])
            self.assertEqual(
                [item for item in resources.metadata["skills"] if item["name"] == "codebase-orientation"],
                [
                    {
                        "name": "codebase-orientation",
                        "path": "/.mira/skills/custom-folder/SKILL.md",
                        "source": "project",
                        "replaces": "default",
                    }
                ],
            )

    def test_project_subagent_replaces_default_by_name(self) -> None:
        """A project subagent with the same name should replace the default."""
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            subagent_dir = workspace / ".mira" / "subagents"
            subagent_dir.mkdir(parents=True)
            (subagent_dir / "reviewer.py").write_text(
                """SUBAGENTS = [
    {
        "name": "code-reviewer",
        "description": "Project reviewer.",
        "system_prompt": "Review this project.",
    }
]
""",
                encoding="utf-8",
            )

            resources = build_resources(workspace, create_examples=False)

            self.assertEqual(len(resources.subagents), 1)
            self.assertEqual(resources.subagents[0]["description"], "Project reviewer.")
            self.assertEqual(
                resources.metadata["subagents"],
                [
                    {
                        "name": "code-reviewer",
                        "path": "/.mira/subagents/reviewer.py",
                        "source": "project",
                        "replaces": "default",
                    }
                ],
            )

    def test_default_tools_include_ask_user_and_regex_grep(self) -> None:
        """Default tools should include ask_user and the built-in grep replacement."""
        with tempfile.TemporaryDirectory() as directory:
            resources = build_resources(Path(directory), create_examples=False)

            self.assertTrue(any(tool.name == "ask_user" for tool in resources.tools))
            self.assertTrue(any(tool.name == "grep" for tool in resources.tools))
            self.assertEqual(
                resources.metadata["tools"],
                [
                    {
                        "name": "ask_user",
                        "path": "/mira-defaults/tools/ask_user.py",
                        "source": "default",
                        "replaces": "",
                    },
                    {
                        "name": "grep",
                        "path": "/mira-defaults/tools/regex_grep.py",
                        "source": "default",
                        "replaces": "built-in",
                    }
                ],
            )

    def test_regex_grep_matches_regex_patterns(self) -> None:
        """The default grep should treat the pattern as regex."""
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "sample.py").write_text("def sample_function():\n    return 1\n", encoding="utf-8")
            resources = build_resources(workspace, create_examples=False)
            grep = next(tool for tool in resources.tools if tool.name == "grep")

            result = grep.invoke({"pattern": r"def\s+\w+_function", "glob": "*.py", "output_mode": "content"})

            self.assertIn("/sample.py:", result)
            self.assertIn("def sample_function()", result)

    def test_regex_grep_stays_inside_project_backend(self) -> None:
        """Regex grep should reject traversal and default-resource searches."""
        with tempfile.TemporaryDirectory() as directory:
            resources = build_resources(Path(directory), create_examples=False)
            grep = next(tool for tool in resources.tools if tool.name == "grep")

            traversal = grep.invoke({"pattern": "anything", "path": "../"})
            defaults = grep.invoke({"pattern": "anything", "path": "/mira-defaults"})

            self.assertIn("Path traversal not allowed", traversal)
            self.assertIn("not /mira-defaults", defaults)

    def test_project_tools_replace_defaults_and_add_unique_tools(self) -> None:
        """Project tools should replace by tool name and add unique tools."""
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            tools_dir = workspace / ".mira" / "tools"
            tools_dir.mkdir(parents=True)
            (tools_dir / "custom_tools.py").write_text(
                '''from langchain.tools import tool


@tool("grep")
def project_grep(pattern: str) -> str:
    """Project grep override."""
    return f"project grep: {pattern}"


@tool
def project_status() -> str:
    """Return project status."""
    return "ready"
''',
                encoding="utf-8",
            )

            resources = build_resources(workspace, create_examples=False)

            names = [tool.name for tool in resources.tools]
            self.assertEqual(names, ["ask_user", "grep", "project_status"])
            self.assertEqual(resources.tools[1].invoke({"pattern": "needle"}), "project grep: needle")
            self.assertEqual(
                resources.metadata["tools"],
                [
                    {
                        "name": "ask_user",
                        "path": "/mira-defaults/tools/ask_user.py",
                        "source": "default",
                        "replaces": "",
                    },
                    {
                        "name": "grep",
                        "path": "/.mira/tools/custom_tools.py",
                        "source": "project",
                        "replaces": "default",
                    },
                    {
                        "name": "project_status",
                        "path": "/.mira/tools/custom_tools.py",
                        "source": "project",
                        "replaces": "",
                    },
                ],
            )

    def test_project_decorated_tool_loads_without_tools_export(self) -> None:
        """A module-level @tool object should load without TOOLS."""
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            tools_dir = workspace / ".mira" / "tools"
            tools_dir.mkdir(parents=True)
            (tools_dir / "web_search.py").write_text(
                '''from langchain_core.tools import tool


@tool("web_search")
def web_search(query: str) -> str:
    """Search the web."""
    return f"result: {query}"
''',
                encoding="utf-8",
            )

            resources = build_resources(workspace, create_examples=False)

            tool = next(tool for tool in resources.tools if tool.name == "web_search")
            self.assertEqual(tool.invoke({"query": "mira"}), "result: mira")
            self.assertIn(
                {
                    "name": "web_search",
                    "path": "/.mira/tools/web_search.py",
                    "source": "project",
                    "replaces": "",
                },
                resources.metadata["tools"],
            )

    def test_multiple_decorated_tools_load_from_one_file(self) -> None:
        """All module-level @tool objects in a file should load."""
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            tools_dir = workspace / ".mira" / "tools"
            tools_dir.mkdir(parents=True)
            (tools_dir / "multi.py").write_text(
                '''from langchain.tools import tool


@tool
def first_tool() -> str:
    """Return first."""
    return "first"


@tool
def second_tool() -> str:
    """Return second."""
    return "second"
''',
                encoding="utf-8",
            )

            resources = build_resources(workspace, create_examples=False)

            names = [tool.name for tool in resources.tools]
            self.assertIn("first_tool", names)
            self.assertIn("second_tool", names)

    def test_explicit_tools_export_still_works_and_deduplicates(self) -> None:
        """TOOLS remains supported without duplicating module-level tools."""
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            tools_dir = workspace / ".mira" / "tools"
            tools_dir.mkdir(parents=True)
            (tools_dir / "explicit.py").write_text(
                '''from langchain.tools import tool


@tool
def exported_tool() -> str:
    """Return exported."""
    return "exported"


TOOLS = [exported_tool]
''',
                encoding="utf-8",
            )

            resources = build_resources(workspace, create_examples=False)

            names = [tool.name for tool in resources.tools]
            self.assertEqual(names.count("exported_tool"), 1)
            self.assertEqual(next(tool for tool in resources.tools if tool.name == "exported_tool").invoke({}), "exported")

    def test_get_tools_export_still_loads_backend_bound_tools(self) -> None:
        """get_tools(project_backend) should remain supported."""
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            tools_dir = workspace / ".mira" / "tools"
            tools_dir.mkdir(parents=True)
            (tools_dir / "dynamic.py").write_text(
                '''from langchain.tools import tool


def get_tools(project_backend):
    """Return tools bound to the workspace backend."""

    @tool
    def workspace_root() -> str:
        """List the workspace root."""
        result = project_backend.ls("/")
        return ",".join(item["path"] for item in result.entries or [])

    return [workspace_root]
''',
                encoding="utf-8",
            )
            (workspace / "sample.txt").write_text("sample", encoding="utf-8")

            resources = build_resources(workspace, create_examples=False)

            tool = next(tool for tool in resources.tools if tool.name == "workspace_root")
            self.assertIn("/sample.txt", tool.invoke({}))

    def test_factory_passes_resources_to_deepagents_and_attaches_metadata(self) -> None:
        """Agent construction should pass discovered resources into DeepAgents."""
        with tempfile.TemporaryDirectory() as directory:
            agent = type("Agent", (), {})()
            with (
                patch("agent.factory.get_llm", return_value="model"),
                patch("agent.factory.CodeInterpreterMiddleware", return_value="code") as code_middleware,
                patch("agent.factory.create_summarization_tool_middleware", return_value="summary"),
                patch("agent.factory.create_deep_agent", return_value=agent) as create_deep_agent,
            ):
                built = factory.build_agent({}, Path(directory), "checkpointer")

        self.assertIs(built, agent)
        code_middleware.assert_called_once()
        self.assertEqual(code_middleware.call_args.kwargs["ptc"], ["task"])
        self.assertIsNotNone(code_middleware.call_args.kwargs["skills_backend"])
        kwargs = create_deep_agent.call_args.kwargs
        self.assertTrue(any(isinstance(middleware, ProviderContextOverflowMiddleware) for middleware in kwargs["middleware"]))
        self.assertIn("/mira-defaults/skills", kwargs["skills"])
        self.assertIn("/.mira/skills", kwargs["skills"])
        self.assertEqual(kwargs["memory"][0], "/.mira/memories/AGENTS.md")
        self.assertTrue(any(subagent["name"] == "code-reviewer" for subagent in kwargs["subagents"]))
        self.assertTrue(any(subagent["name"] == "project-guide" for subagent in kwargs["subagents"]))
        self.assertTrue(any(tool.name == "grep" for tool in kwargs["tools"]))
        self.assertTrue(any(tool.name == "project_note" for tool in kwargs["tools"]))
        self.assertIn("memories", agent.mira_resources)
        self.assertIn("tools", agent.mira_resources)

    def test_default_tool_specs_use_current_eval_name(self) -> None:
        """Fallback UI metadata should use the current interpreter tool name."""
        names = [tool["name"] for tool in repl.DEFAULT_TOOL_SPECS]

        self.assertIn("eval", names)
        self.assertNotIn("execute", names)

    def test_resource_specs_normalize_agent_metadata(self) -> None:
        """UI resource specs should come from attached agent metadata."""
        agent = type(
            "Agent",
            (),
            {
                "mira_resources": {
                    "memories": [
                        {
                            "name": "AGENTS.md",
                            "path": "/.mira/memories/AGENTS.md",
                            "source": "project",
                            "replaces": "default",
                        }
                    ],
                }
            },
        )()

        self.assertEqual(
            repl.resource_specs(agent)["memories"],
            [
                {
                    "name": "AGENTS.md",
                    "path": "/.mira/memories/AGENTS.md",
                    "source": "project",
                    "replaces": "default",
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
