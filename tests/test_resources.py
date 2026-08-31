"""Tests for MIRA resource discovery and layering."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from deepagents.backends import FilesystemBackend, LocalShellBackend

from agent import factory
from agent.middleware.context_overflow import ProviderContextOverflowMiddleware
from agent.middleware import (
    ExecuteToolDescriptionRewriteMiddleware,
    FileReferenceMiddleware,
    ProjectToolErrorMiddleware,
    QUICKJS_MEMORY_LIMIT,
    QUICKJS_PERSISTENCE_MODE,
    QUICKJS_PTC_TOOLS,
    QUICKJS_TIMEOUT_SECONDS,
)
from agent.resources import (
    EXECUTE_ENV_KEYS,
    ProjectShellBackend,
    build_resources,
    execute_env,
    wrap_execute_command,
)
from agent.resources.project_setup import (
    EMPTY_MCP_CONFIGURATION,
    EXAMPLE_MCP_CONFIGURATION,
    MCP_CONFIGURATION_SCHEMA,
    ensure_project_examples,
)
from config.settings import READ_ONLY_BUILTIN_TOOLS, load_settings, set_subagent_enabled
from core.application import DEFAULT_TOOL_SPECS, resource_specs


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
            mcp_dir = workspace / ".mira" / "mcp"
            expected_files = {
                "mcp.json": EMPTY_MCP_CONFIGURATION,
                "example.json": EXAMPLE_MCP_CONFIGURATION,
                "schema.json": MCP_CONFIGURATION_SCHEMA,
            }
            for name, expected in expected_files.items():
                with self.subTest(name=name):
                    generated = (mcp_dir / name).read_text(encoding="utf-8")
                    self.assertEqual(generated, expected)
                    self.assertTrue(generated.endswith("\n"))
                    self.assertIsInstance(json.loads(generated), dict)
            self.assertFalse((workspace / ".mira" / "mcp.json").exists())
            project_readme = (workspace / ".mira" / "README.md").read_text(encoding="utf-8")
            self.assertIn("`mcp/mcp.json`: active MCP configuration", project_readme)
            self.assertIn("Run `/reload-runtime` after changes", project_readme)
            self.assertTrue((workspace / ".mira" / "skills" / "example-skill" / "SKILL.md").exists())
            self.assertTrue((workspace / ".mira" / "subagents" / "example_subagent.py").exists())
            self.assertEqual(list((workspace / ".mira" / "tools").glob("*.py")), [])
            self.assertTrue((workspace / ".mira" / "examples" / "tools" / "mira_runtime_tool.py").exists())
            self.assertTrue((workspace / ".mira" / "examples" / "tools" / "project_runtime_tool.py").exists())
            self.assertIn(
                "Example Skill",
                (workspace / ".mira" / "skills" / "example-skill" / "SKILL.md").read_text(encoding="utf-8"),
            )
            self.assertIn(
                "example-project-guide",
                (workspace / ".mira" / "subagents" / "example_subagent.py").read_text(encoding="utf-8"),
            )
            self.assertIn(
                "project_tool",
                (workspace / ".mira" / "examples" / "tools" / "project_runtime_tool.py").read_text(
                    encoding="utf-8"
                ),
            )

    def test_mcp_bootstrap_preserves_each_existing_file_and_fills_missing_companions(self) -> None:
        """Each MCP file should be preserved independently while missing files appear."""
        filenames = ("mcp.json", "example.json", "schema.json")
        for existing_name in filenames:
            with self.subTest(existing_name=existing_name), tempfile.TemporaryDirectory() as directory:
                workspace = Path(directory)
                mcp_dir = workspace / ".mira" / "mcp"
                mcp_dir.mkdir(parents=True)
                existing_path = mcp_dir / existing_name
                existing = f"custom {existing_name}\n"
                existing_path.write_text(existing, encoding="utf-8")

                ensure_project_examples(workspace)

                self.assertEqual(existing_path.read_text(encoding="utf-8"), existing)
                self.assertTrue(all((mcp_dir / name).exists() for name in filenames))

    def test_repeated_bootstrap_does_not_modify_existing_mcp_files(self) -> None:
        """Re-running bootstrap should leave all three MCP files byte-for-byte unchanged."""
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            ensure_project_examples(workspace)
            mcp_dir = workspace / ".mira" / "mcp"
            before = {path.name: path.read_bytes() for path in mcp_dir.iterdir()}

            ensure_project_examples(workspace)

            self.assertEqual({path.name: path.read_bytes() for path in mcp_dir.iterdir()}, before)

    def test_default_memories_load_without_project_memory(self) -> None:
        """Both bundled memories should load when project examples are skipped."""
        with tempfile.TemporaryDirectory() as directory:
            resources = build_resources(Path(directory), create_examples=False)

            self.assertEqual(
                resources.memory,
                [
                    "/mira-defaults/memories/AGENTS.md",
                    "/mira-defaults/memories/software-development.md",
                ],
            )
            self.assertEqual(
                [item["source"] for item in resources.metadata["memories"]],
                ["default", "default"],
            )

    def test_execute_disabled_uses_filesystem_backend(self) -> None:
        """Disabled execute should keep the normal filesystem backend."""
        with tempfile.TemporaryDirectory() as directory:
            resources = build_resources(Path(directory), create_examples=False)

            self.assertIsInstance(resources.backend.default, FilesystemBackend)
            self.assertNotIsInstance(resources.backend.default, LocalShellBackend)
            self.assertIs(resources.project_backend, resources.backend.default)

    def test_execute_enabled_uses_local_shell_backend(self) -> None:
        """Enabled execute should switch the project backend to LocalShellBackend."""
        with tempfile.TemporaryDirectory() as directory:
            resources = build_resources(
                Path(directory),
                create_examples=False,
                settings={"hitl": {"tools": {"execute": {"enabled": True, "always_allow": False}}}},
            )

            self.assertIsInstance(resources.backend.default, LocalShellBackend)
            self.assertIsInstance(resources.backend.default, ProjectShellBackend)
            self.assertEqual(resources.backend.default._env, execute_env(settings=resources.backend.default._execute_env_settings))
            self.assertLessEqual(set(resources.backend.default._env), set(EXECUTE_ENV_KEYS))
            if os.environ.get("PATH"):
                self.assertEqual(resources.backend.default._env["PATH"], os.environ["PATH"])
            for secret_name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GITHUB_TOKEN", "PASSWORD", "SECRET"):
                self.assertNotIn(secret_name, resources.backend.default._env)

    def test_execute_env_includes_safe_windows_path_vars_when_present(self) -> None:
        """Execute env should include safe OS paths without inheriting secrets."""
        with patch.dict(
            os.environ,
            {
                "PATH": "C:\\Tools",
                "SYSTEMDRIVE": "C:",
                "PROGRAMDATA": "C:\\ProgramData",
                "APPDATA": "C:\\Users\\me\\AppData\\Roaming",
                "LOCALAPPDATA": "C:\\Users\\me\\AppData\\Local",
                "OPENAI_API_KEY": "secret",
            },
            clear=True,
        ):
            env = execute_env()

        self.assertEqual(env["SYSTEMDRIVE"], "C:")
        self.assertEqual(env["PROGRAMDATA"], "C:\\ProgramData")
        self.assertEqual(env["APPDATA"], "C:\\Users\\me\\AppData\\Roaming")
        self.assertEqual(env["LOCALAPPDATA"], "C:\\Users\\me\\AppData\\Local")
        self.assertNotIn("OPENAI_API_KEY", env)

    def test_execute_env_additional_allowlist_reads_current_host_value_only(self) -> None:
        """User allowlists should include present names and ignore missing names."""
        settings = {"hitl": {"execute_env": {"allow": ["CUDA_HOME", "MISSING_LOCAL_VAR"]}}}
        with patch.dict(os.environ, {"CUDA_HOME": "C:\\CUDA"}, clear=True):
            env = execute_env(settings=settings)

        self.assertEqual(env["CUDA_HOME"], "C:\\CUDA")
        self.assertNotIn("MISSING_LOCAL_VAR", env)

    def test_execute_env_venv_mode_sets_virtual_env_and_path(self) -> None:
        """Venv mode should prepare PATH from either a venv folder or executable path."""
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            settings = {"hitl": {"execute_env": {"mode": "venv", "path": ".venv"}}}
            with patch.dict(os.environ, {"PATH": "C:\\Tools"}, clear=True):
                env = execute_env(settings=settings, workspace=workspace)

        self.assertEqual(env["VIRTUAL_ENV"], str((workspace / ".venv").resolve()))
        self.assertTrue(env["PATH"].startswith(str((workspace / ".venv" / "Scripts").resolve())))
        self.assertIn(os.pathsep + "C:\\Tools", env["PATH"])

    def test_execute_env_conda_modes_wrap_commands(self) -> None:
        """Conda modes should run the full shell command through conda run."""
        by_name = wrap_execute_command("python -V && echo ok", {"mode": "conda_name", "name": "project_env"})
        by_prefix = wrap_execute_command("python -V", {"mode": "conda_prefix", "prefix": r"C:\envs\project env"})

        self.assertTrue(by_name.startswith("conda run -n project_env "))
        self.assertIn("python -V", by_name)
        self.assertIn("echo ok", by_name)
        self.assertTrue(by_prefix.startswith("conda run -p "))
        self.assertIn("project env", by_prefix)
        self.assertIn("python -V", by_prefix)

    def test_project_memory_replaces_default_by_filename(self) -> None:
        """A project memory with the same filename should replace the default."""
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            memory_dir = workspace / ".mira" / "memories"
            memory_dir.mkdir(parents=True)
            (memory_dir / "AGENTS.md").write_text("project memory", encoding="utf-8")
            (memory_dir / "soul.md").write_text("project-only memory", encoding="utf-8")

            resources = build_resources(workspace, create_examples=False)

            self.assertEqual(
                resources.memory,
                [
                    "/.mira/memories/AGENTS.md",
                    "/mira-defaults/memories/software-development.md",
                    "/.mira/memories/soul.md",
                ],
            )
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
                        "name": "software-development.md",
                        "path": "/mira-defaults/memories/software-development.md",
                        "source": "default",
                        "replaces": "",
                    },
                    {
                        "name": "soul.md",
                        "path": "/.mira/memories/soul.md",
                        "source": "project",
                        "replaces": "",
                    },
                ],
            )

    def test_project_software_development_memory_replaces_only_bundled_match(self) -> None:
        """Projects may replace the software guide without losing other defaults."""
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            memory_dir = workspace / ".mira" / "memories"
            memory_dir.mkdir(parents=True)
            (memory_dir / "software-development.md").write_text(
                "project methodology",
                encoding="utf-8",
            )

            resources = build_resources(workspace, create_examples=False)

            self.assertEqual(
                resources.memory,
                [
                    "/mira-defaults/memories/AGENTS.md",
                    "/.mira/memories/software-development.md",
                ],
            )
            self.assertEqual(
                resources.metadata["memories"],
                [
                    {
                        "name": "AGENTS.md",
                        "path": "/mira-defaults/memories/AGENTS.md",
                        "source": "default",
                        "replaces": "",
                    },
                    {
                        "name": "software-development.md",
                        "path": "/.mira/memories/software-development.md",
                        "source": "project",
                        "replaces": "default",
                    },
                ],
            )

    def test_project_skill_loads_by_name(self) -> None:
        """A project skill should load by frontmatter name without bundled skill defaults."""
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            skill_dir = workspace / ".mira" / "skills" / "custom-folder"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                """---
name: project-skill
description: Project-specific workflow.
---

# Project Skill
""",
                encoding="utf-8",
            )

            resources = build_resources(workspace, create_examples=False)

            self.assertEqual(resources.skills, ["/.mira/skills"])
            self.assertEqual(
                resources.metadata["skills"],
                [
                    {
                        "name": "project-skill",
                        "path": "/.mira/skills/custom-folder/SKILL.md",
                        "source": "project",
                        "replaces": "",
                    }
                ],
            )

    def test_project_subagent_is_discovered_disabled_then_can_be_enabled(self) -> None:
        """New project subagents are visible but excluded until enabled."""
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            subagent_dir = workspace / ".mira" / "subagents"
            subagent_dir.mkdir(parents=True)
            (subagent_dir / "reviewer.py").write_text(
                """SUBAGENTS = [
    {
        "name": "project-guide",
        "description": "Project guide.",
        "system_prompt": "Guide this project.",
    }
]
""",
                encoding="utf-8",
            )

            discovered = build_resources(workspace, create_examples=False)
            self.assertEqual(discovered.subagents, [])
            project = next(item for item in discovered.metadata["subagents"] if item["name"] == "project-guide")
            self.assertEqual(project["description"], "Project guide.")
            self.assertEqual(project["path"], "/.mira/subagents/reviewer.py")

            configured = set_subagent_enabled(load_settings(workspace), "project-guide", True)
            resources = build_resources(
                workspace,
                create_examples=False,
                settings=configured,
                config={"settings": configured},
            )
            self.assertEqual(
                [subagent["name"] for subagent in resources.subagents],
                ["general-purpose", "project-guide"],
            )

    def test_default_resources_discover_permanent_general_purpose_subagent(self) -> None:
        """General-purpose is first in metadata and enabled during configured builds."""
        with tempfile.TemporaryDirectory() as directory:
            resources = build_resources(Path(directory), create_examples=False)

            self.assertEqual(resources.skills, [])
            self.assertEqual(resources.metadata["skills"], [])
            self.assertEqual(resources.subagents, [])
            self.assertEqual(resources.metadata["subagents"][0]["name"], "general-purpose")

    def test_default_tools_include_planning_controls_and_regex_grep(self) -> None:
        """Default tools should include planning controls and the built-in grep replacement."""
        with tempfile.TemporaryDirectory() as directory:
            resources = build_resources(Path(directory), create_examples=False)

            self.assertTrue(any(tool.name == "ask_user" for tool in resources.tools))
            self.assertTrue(any(tool.name == "show_plan" for tool in resources.tools))
            self.assertTrue(any(tool.name == "show_goal" for tool in resources.tools))
            self.assertTrue(any(tool.name == "prepare_goal" for tool in resources.tools))
            self.assertTrue(any(tool.name == "prepare_plan" for tool in resources.tools))
            self.assertTrue(any(tool.name == "finalize_plan" for tool in resources.tools))
            self.assertTrue(any(tool.name == "finalize_goal" for tool in resources.tools))
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
                        "name": "finalize_goal",
                        "path": "/mira-defaults/tools/finalize_goal.py",
                        "source": "default",
                        "replaces": "",
                    },
                    {
                        "name": "finalize_plan",
                        "path": "/mira-defaults/tools/finalize_plan.py",
                        "source": "default",
                        "replaces": "",
                    },
                    {
                        "name": "prepare_goal",
                        "path": "/mira-defaults/tools/prepare_goal.py",
                        "source": "default",
                        "replaces": "",
                    },
                    {
                        "name": "prepare_plan",
                        "path": "/mira-defaults/tools/prepare_plan.py",
                        "source": "default",
                        "replaces": "",
                    },
                    {
                        "name": "grep",
                        "path": "/mira-defaults/tools/regex_grep.py",
                        "source": "default",
                        "replaces": "built-in",
                    },
                    {
                        "name": "show_goal",
                        "path": "/mira-defaults/tools/show_goal.py",
                        "source": "default",
                        "replaces": "",
                    },
                    {
                        "name": "show_plan",
                        "path": "/mira-defaults/tools/show_plan.py",
                        "source": "default",
                        "replaces": "",
                    }
                ],
            )

    def test_ask_user_description_honors_explicit_many_option_requests(self) -> None:
        """The default ask_user prompt should distinguish explicit ask_user requests."""
        with tempfile.TemporaryDirectory() as directory:
            resources = build_resources(Path(directory), create_examples=False)
            ask_user = next(tool for tool in resources.tools if tool.name == "ask_user")

            description = str(ask_user.description)

            self.assertIn("either Plan or Act mode", description)
            self.assertIn("Prefer this tool over asking any user-facing question in prose", description)
            self.assertIn("without asking for ask_user, answer normally in chat", description)
            self.assertIn("explicitly asks you to use ask_user with many options", description)
            self.assertIn("include every requested option", description)
            self.assertIn("MIRA numbers choices in the UI", description)
            self.assertIn("good options: ['test_checkpoint.py', 'test_config.py']", description)
            self.assertIn("bad options: ['1. test_checkpoint.py', '2. test_config.py']", description)

    def test_finalize_plan_description_is_for_forced_finalisation(self) -> None:
        """finalize_plan should be reserved for criteria-first finalisation."""
        with tempfile.TemporaryDirectory() as directory:
            resources = build_resources(Path(directory), create_examples=False)
            finalize_plan = next(tool for tool in resources.tools if tool.name == "finalize_plan")

            description = str(finalize_plan.description)

            self.assertIn("after MIRA has generated Success Criteria", description)
            self.assertIn("required in the formal finalisation stage", description)
            self.assertIn("Do not add a Summary section", description)

    def test_show_controls_require_immediate_exact_retained_artifact_display(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            resources = build_resources(Path(directory), create_examples=False)

            for name, artifact in (("show_plan", "Plan"), ("show_goal", "Goal")):
                with self.subTest(name=name):
                    control = next(tool for tool in resources.tools if tool.name == name)
                    description = str(control.description)
                    self.assertIn("Immediately render the exact retained", description)
                    self.assertIn("show, reopen, review, or return", description)
                    self.assertIn("Do not research", description)
                    self.assertIn(f"reproduce the {artifact} in prose", description)
                    self.assertIn("prepare a replacement", description)
                    self.assertIn("finalize it first", description)

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
            self.assertEqual(
                names,
                [
                    "ask_user",
                    "finalize_goal",
                    "finalize_plan",
                    "prepare_goal",
                    "prepare_plan",
                    "grep",
                    "show_goal",
                    "show_plan",
                    "project_status",
                ],
            )
            self.assertEqual(resources.tools[5].invoke({"pattern": "needle"}), "project grep: needle")
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
                        "name": "finalize_goal",
                        "path": "/mira-defaults/tools/finalize_goal.py",
                        "source": "default",
                        "replaces": "",
                    },
                    {
                        "name": "finalize_plan",
                        "path": "/mira-defaults/tools/finalize_plan.py",
                        "source": "default",
                        "replaces": "",
                    },
                    {
                        "name": "prepare_goal",
                        "path": "/mira-defaults/tools/prepare_goal.py",
                        "source": "default",
                        "replaces": "",
                    },
                    {
                        "name": "prepare_plan",
                        "path": "/mira-defaults/tools/prepare_plan.py",
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
                        "name": "show_goal",
                        "path": "/mira-defaults/tools/show_goal.py",
                        "source": "default",
                        "replaces": "",
                    },
                    {
                        "name": "show_plan",
                        "path": "/mira-defaults/tools/show_plan.py",
                        "source": "default",
                        "replaces": "",
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

    def test_disabled_project_tool_stays_in_metadata_but_not_agent_tools(self) -> None:
        """Disabled project tools should be hidden from the agent while remaining configurable."""
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

            resources = build_resources(
                workspace,
                create_examples=False,
                settings={"hitl": {"tools": {"web_search": {"enabled": False, "always_allow": False}}}},
            )

            self.assertNotIn("web_search", [tool.name for tool in resources.tools])
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
                patch("agent.middleware.builder.CodeInterpreterMiddleware", return_value="code") as code_middleware,
                patch("agent.middleware.builder.create_mira_summarization_middleware", return_value="auto-summary"),
                patch("agent.middleware.builder.create_mira_summarization_tool_middleware", return_value="summary"),
                patch("agent.factory.create_deep_agent", return_value=agent) as create_deep_agent,
            ):
                built = factory.build_agent({}, Path(directory), "checkpointer")

        self.assertIs(built, agent)
        code_middleware.assert_called_once()
        self.assertEqual(code_middleware.call_args.kwargs["ptc"], list(QUICKJS_PTC_TOOLS))
        self.assertEqual(code_middleware.call_args.kwargs["memory_limit"], QUICKJS_MEMORY_LIMIT)
        self.assertEqual(code_middleware.call_args.kwargs["timeout"], QUICKJS_TIMEOUT_SECONDS)
        self.assertEqual(code_middleware.call_args.kwargs["mode"], QUICKJS_PERSISTENCE_MODE)
        self.assertNotIn("skills_backend", code_middleware.call_args.kwargs)
        kwargs = create_deep_agent.call_args.kwargs
        self.assertIn("auto-summary", kwargs["middleware"])
        self.assertIn("summary", kwargs["middleware"])
        self.assertTrue(any(isinstance(middleware, ProviderContextOverflowMiddleware) for middleware in kwargs["middleware"]))
        self.assertTrue(
            any(
                isinstance(middleware, ExecuteToolDescriptionRewriteMiddleware)
                for middleware in kwargs["middleware"]
            )
        )
        self.assertTrue(any(isinstance(middleware, FileReferenceMiddleware) for middleware in kwargs["middleware"]))
        self.assertIn("/.mira/skills", kwargs["skills"])
        self.assertEqual(kwargs["memory"][0], "/.mira/memories/AGENTS.md")
        self.assertEqual([subagent["name"] for subagent in kwargs["subagents"]], ["general-purpose"])
        self.assertTrue(any(tool.name == "grep" for tool in kwargs["tools"]))
        self.assertFalse(any(tool.name == "example_project_note" for tool in kwargs["tools"]))
        self.assertIn("memories", agent.mira_resources)
        self.assertIn("tools", agent.mira_resources)
        self.assertNotIn("execute", [tool["name"] for tool in agent.mira_tool_specs])
        self.assertNotIn("finalize_plan", [tool["name"] for tool in agent.mira_tool_specs])
        self.assertIs(agent.mira_project_backend, kwargs["backend"].default)

    def test_factory_scopes_project_tool_error_middleware_to_enabled_workspace_tools(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            tools_dir = workspace / ".mira" / "tools"
            tools_dir.mkdir(parents=True)
            (tools_dir / "reader.py").write_text(
                '''from langchain.tools import tool


@tool
def read_file_as_bytes(path: str) -> str:
    """Read a file as bytes."""
    raise FileNotFoundError(path)
''',
                encoding="utf-8",
            )
            config = {
                "settings": {
                    "hitl": {
                        "tools": {
                            "read_file_as_bytes": {"enabled": True, "always_allow": True}
                        }
                    }
                }
            }
            agent = type("Agent", (), {})()
            with (
                patch("agent.factory.get_llm", return_value="model"),
                patch("agent.middleware.builder.CodeInterpreterMiddleware", return_value="code"),
                patch("agent.middleware.builder.create_mira_summarization_middleware", return_value="auto-summary"),
                patch("agent.middleware.builder.create_mira_summarization_tool_middleware", return_value="summary"),
                patch("agent.factory.create_deep_agent", return_value=agent) as create,
            ):
                factory.build_agent(config, workspace, "checkpointer")

        middleware = [
            item
            for item in create.call_args.kwargs["middleware"]
            if isinstance(item, ProjectToolErrorMiddleware)
        ]
        self.assertEqual(len(middleware), 1)
        self.assertEqual(middleware[0].tool_names, frozenset({"read_file_as_bytes"}))

    def test_action_and_plan_agents_share_ordered_opaque_memory_resources(self) -> None:
        """Memory filenames should only determine stable ordering, not agent roles."""
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            memory_dir = workspace / ".mira" / "memories"
            memory_dir.mkdir(parents=True)
            (memory_dir / "zebra-notes.md").write_text("z", encoding="utf-8")
            (memory_dir / "01-context.md").write_text("a", encoding="utf-8")
            built_agents = [type("Agent", (), {})(), type("Agent", (), {})()]
            with (
                patch("agent.factory.get_llm", return_value="model"),
                patch("agent.middleware.builder.CodeInterpreterMiddleware", return_value="code"),
                patch("agent.middleware.builder.create_mira_summarization_middleware", return_value="auto-summary"),
                patch("agent.middleware.builder.create_mira_summarization_tool_middleware", return_value="summary"),
                patch("agent.factory.create_deep_agent", side_effect=built_agents) as create,
            ):
                factory.build_agent({}, workspace, "checkpointer")
                factory.build_plan_agent({}, workspace, "checkpointer")

        action_memory = create.call_args_list[0].kwargs["memory"]
        plan_memory = create.call_args_list[1].kwargs["memory"]
        for call in create.call_args_list:
            self.assertTrue(
                any(isinstance(item, FileReferenceMiddleware) for item in call.kwargs["middleware"])
            )
        self.assertEqual(action_memory, plan_memory)
        self.assertEqual(
            action_memory,
            [
                "/.mira/memories/AGENTS.md",
                "/mira-defaults/memories/software-development.md",
                "/.mira/memories/01-context.md",
                "/.mira/memories/zebra-notes.md",
            ],
        )

    def test_factory_enables_execute_with_local_shell_backend_without_permissions(self) -> None:
        """Execute mode should expose execute and avoid incompatible filesystem permissions."""
        with tempfile.TemporaryDirectory() as directory:
            agent = type("Agent", (), {})()
            config = {"settings": {"hitl": {"tools": {"execute": {"enabled": True, "always_allow": False}}}}}
            with (
                patch("agent.factory.get_llm", return_value="model"),
                patch("agent.middleware.builder.CodeInterpreterMiddleware", return_value="code"),
                patch("agent.middleware.builder.create_mira_summarization_middleware", return_value="auto-summary"),
                patch("agent.middleware.builder.create_mira_summarization_tool_middleware", return_value="summary"),
                patch("agent.factory.create_deep_agent", return_value=agent) as create_deep_agent,
            ):
                factory.build_agent(config, Path(directory), "checkpointer")

        kwargs = create_deep_agent.call_args.kwargs
        self.assertEqual(kwargs["permissions"], [])
        self.assertIsInstance(kwargs["backend"].default, LocalShellBackend)
        self.assertIn("execute", kwargs["interrupt_on"])
        self.assertIn("execute", [tool["name"] for tool in agent.mira_tool_specs])
        self.assertNotIn("finalize_plan", [tool["name"] for tool in agent.mira_tool_specs])

    def test_quickjs_ptc_tools_include_only_safe_project_exploration(self) -> None:
        """QuickJS PTC should expose read-only exploration tools, not writes or interrupts."""
        ptc_tools = set(QUICKJS_PTC_TOOLS)

        self.assertEqual(ptc_tools, {"ls", "read_file", "glob", "grep"})
        self.assertEqual(QUICKJS_PTC_TOOLS, READ_ONLY_BUILTIN_TOOLS)
        self.assertFalse({"task", "write_file", "edit_file", "execute", "ask_user", "finalize_plan"} & ptc_tools)

    def test_effective_ptc_tools_exclude_globally_disabled_reads(self) -> None:
        """PTC should derive its safe reads from the same global exclusions as the model."""
        config = {
            "settings": {
                "hitl": {
                    "tools": {
                        name: {"enabled": name != "grep"}
                        for name in READ_ONLY_BUILTIN_TOOLS
                    }
                }
            }
        }
        excluded = factory.effective_excluded_tools(config, (), True)

        resolved = factory.effective_ptc_tool_names(config, [], [], excluded)

        self.assertEqual(resolved, ["ls", "read_file", "glob"])

    def test_effective_ptc_tools_use_only_resolved_tools_and_runtime_names(self) -> None:
        """Saved PTC flags should not bypass effective local or MCP availability."""
        config = {
            "settings": {
                "hitl": {
                    "tools": {
                        "write_file": {"enabled": True, "ptc": True},
                        "edit_file": {"enabled": False, "ptc": True},
                        "eval": {"enabled": True, "ptc": True},
                        "task": {"enabled": True, "ptc": True},
                        "available_custom": {"enabled": True, "ptc": True},
                        "missing_custom": {"enabled": True, "ptc": True},
                    }
                },
                "mcp": {
                    "servers": {
                        "docs": {
                            "tools": {
                                "search": {"enabled": True, "ptc": True},
                                "missing": {"enabled": True, "ptc": True},
                            }
                        }
                    }
                },
            }
        }
        tools = [
            SimpleNamespace(name="available_custom"),
            SimpleNamespace(name="mcp__docs__search"),
        ]
        metadata = [
            {"name": "available_custom", "source": "project"},
            {"name": "missing_custom", "source": "project"},
            {
                "name": "mcp__docs__search",
                "original_name": "search",
                "server": "docs",
                "source": "mcp",
            },
            {
                "name": "mcp__docs__missing",
                "original_name": "missing",
                "server": "docs",
                "source": "mcp",
            },
        ]

        resolved = factory.effective_ptc_tool_names(
            config,
            tools,
            metadata,
            excluded_tools=("edit_file",),
        )

        self.assertEqual(
            resolved,
            [*QUICKJS_PTC_TOOLS, "write_file", "available_custom", "mcp__docs__search"],
        )
        self.assertNotIn("search", resolved)
        self.assertNotIn("missing_custom", resolved)
        self.assertNotIn("mcp__docs__missing", resolved)
        self.assertNotIn("eval", resolved)
        self.assertNotIn("task", resolved)

    def test_factory_passes_action_only_custom_ptc_access_to_quickjs(self) -> None:
        """Plan filtering should happen before the effective PTC list is built."""
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            tools_dir = workspace / ".mira" / "tools"
            tools_dir.mkdir(parents=True)
            (tools_dir / "lookup.py").write_text(
                '''from langchain.tools import tool


@tool
def project_lookup(query: str) -> str:
    """Look up project information."""
    return query
''',
                encoding="utf-8",
            )
            config = {
                "settings": {
                    "hitl": {
                        "tools": {
                            "project_lookup": {
                                "enabled": True,
                                "plan_access": False,
                                "ptc": True,
                            }
                        }
                    }
                }
            }
            with (
                patch("agent.factory.get_llm", return_value="model"),
                patch("agent.middleware.builder.CodeInterpreterMiddleware", return_value="code") as code,
                patch("agent.middleware.builder.create_mira_summarization_middleware", return_value="auto-summary"),
                patch("agent.middleware.builder.create_mira_summarization_tool_middleware", return_value="summary"),
                patch(
                    "agent.factory.create_deep_agent",
                    side_effect=[type("Agent", (), {})(), type("Agent", (), {})()],
                ),
            ):
                factory.build_agent(config, workspace, "checkpointer")
                factory.build_plan_agent(config, workspace, "checkpointer")

        action_ptc = code.call_args_list[0].kwargs["ptc"]
        plan_ptc = code.call_args_list[1].kwargs["ptc"]
        self.assertIn("project_lookup", action_ptc)
        self.assertNotIn("project_lookup", plan_ptc)

    def test_factory_registers_specific_and_provider_summarization_exclusions(self) -> None:
        """DeepAgents should exclude its hidden default summarization for resolved models."""
        model = type(
            "Model",
            (),
            {
                "model_name": "google/gemma",
                "_get_ls_params": lambda self: {"ls_provider": "anyllm"},
            },
        )()
        with (
            patch("agent.factory.register_harness_profile") as register,
            patch.object(factory, "_REGISTERED_SUMMARIZATION_PROFILE_KEYS", set()),
        ):
            factory._register_summarization_exclusion({"llm_provider": "openai", "llm_model": "gpt-test"}, model)

        keys = [call.args[0] for call in register.call_args_list]
        self.assertEqual(keys, ["anyllm:google/gemma", "anyllm"])

    def test_factory_skips_invalid_ollama_summarization_profile_key(self) -> None:
        """Ollama model tags should not create a double-colon registry key."""
        model = type(
            "Model",
            (),
            {
                "model_name": "qwen3.6:27b",
                "_get_ls_params": lambda self: {"ls_provider": "ollama"},
            },
        )()
        with (
            patch("agent.factory.register_harness_profile") as register,
            patch.object(factory, "_REGISTERED_SUMMARIZATION_PROFILE_KEYS", set()),
        ):
            factory._register_summarization_exclusion(
                {"llm_provider": "ollama", "llm_model": "qwen3.6:27b"},
                model,
            )

        keys = [call.args[0] for call in register.call_args_list]
        self.assertEqual(keys, ["qwen3.6:27b", "ollama"])

    def test_default_tool_specs_use_current_eval_name(self) -> None:
        """Fallback UI metadata should use the current interpreter tool name."""
        names = [tool["name"] for tool in DEFAULT_TOOL_SPECS]

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
            resource_specs(agent)["memories"],
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
