"""Tests for MIRA resource discovery and layering."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from deepagents.backends import FilesystemBackend, LocalShellBackend

from agent import factory
from agent.context_overflow import ProviderContextOverflowMiddleware
from agent.middleware import ExecuteToolPromptMiddleware, QUICKJS_PTC_TOOLS
from agent.resources import (
    EXECUTE_ENV_KEYS,
    ProjectShellBackend,
    build_resources,
    execute_env,
    wrap_execute_command,
)
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
            self.assertIn(
                "Example Skill",
                (workspace / ".mira" / "skills" / "example-skill" / "SKILL.md").read_text(encoding="utf-8"),
            )
            self.assertIn(
                "example-project-guide",
                (workspace / ".mira" / "subagents" / "example_subagent.py").read_text(encoding="utf-8"),
            )
            self.assertIn(
                "example_project_note",
                (workspace / ".mira" / "tools" / "example_tool.py").read_text(encoding="utf-8"),
            )

    def test_default_memory_loads_without_project_memory(self) -> None:
        """The bundled AGENTS.md should load when project examples are skipped."""
        with tempfile.TemporaryDirectory() as directory:
            resources = build_resources(Path(directory), create_examples=False)

            self.assertEqual(resources.memory, ["/mira-defaults/memories/AGENTS.md"])
            self.assertEqual(resources.metadata["memories"][0]["source"], "default")

    def test_execute_disabled_uses_filesystem_backend(self) -> None:
        """Disabled execute should keep the normal filesystem backend."""
        with tempfile.TemporaryDirectory() as directory:
            resources = build_resources(Path(directory), create_examples=False)

            self.assertIsInstance(resources.backend.default, FilesystemBackend)
            self.assertNotIsInstance(resources.backend.default, LocalShellBackend)

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

    def test_project_subagent_loads_by_name(self) -> None:
        """A project subagent should load without bundled subagent defaults."""
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

            resources = build_resources(workspace, create_examples=False)

            self.assertEqual(len(resources.subagents), 1)
            self.assertEqual(resources.subagents[0]["description"], "Project guide.")
            self.assertEqual(
                resources.metadata["subagents"],
                [
                    {
                        "name": "project-guide",
                        "path": "/.mira/subagents/reviewer.py",
                        "source": "project",
                        "replaces": "",
                    }
                ],
            )

    def test_default_resources_include_no_skills_or_subagents(self) -> None:
        """Bundled defaults should stay minimal: memory plus tools only."""
        with tempfile.TemporaryDirectory() as directory:
            resources = build_resources(Path(directory), create_examples=False)

            self.assertEqual(resources.skills, [])
            self.assertEqual(resources.metadata["skills"], [])
            self.assertEqual(resources.subagents, [])
            self.assertEqual(resources.metadata["subagents"], [])

    def test_default_tools_include_ask_user_and_regex_grep(self) -> None:
        """Default tools should include ask_user and the built-in grep replacement."""
        with tempfile.TemporaryDirectory() as directory:
            resources = build_resources(Path(directory), create_examples=False)

            self.assertTrue(any(tool.name == "ask_user" for tool in resources.tools))
            self.assertTrue(any(tool.name == "present_plan" for tool in resources.tools))
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
                        "name": "present_plan",
                        "path": "/mira-defaults/tools/present_plan.py",
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

    def test_ask_user_description_honors_explicit_many_option_requests(self) -> None:
        """The default ask_user prompt should distinguish explicit ask_user requests."""
        with tempfile.TemporaryDirectory() as directory:
            resources = build_resources(Path(directory), create_examples=False)
            ask_user = next(tool for tool in resources.tools if tool.name == "ask_user")

            description = str(ask_user.description)

            self.assertIn("without asking for ask_user, answer normally in chat", description)
            self.assertIn("explicitly asks you to use ask_user with many options", description)
            self.assertIn("include every requested option", description)
            self.assertIn("MIRA numbers choices in the UI", description)
            self.assertIn("good options: ['test_checkpoint.py', 'test_config.py']", description)
            self.assertIn("bad options: ['1. test_checkpoint.py', '2. test_config.py']", description)

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
            self.assertEqual(names, ["ask_user", "present_plan", "grep", "project_status"])
            self.assertEqual(resources.tools[2].invoke({"pattern": "needle"}), "project grep: needle")
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
                        "name": "present_plan",
                        "path": "/mira-defaults/tools/present_plan.py",
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
                patch("agent.middleware.CodeInterpreterMiddleware", return_value="code") as code_middleware,
                patch("agent.middleware.create_mira_summarization_middleware", return_value="auto-summary"),
                patch("agent.middleware.create_mira_summarization_tool_middleware", return_value="summary"),
                patch("agent.factory.create_deep_agent", return_value=agent) as create_deep_agent,
            ):
                built = factory.build_agent({}, Path(directory), "checkpointer")

        self.assertIs(built, agent)
        code_middleware.assert_called_once()
        self.assertEqual(code_middleware.call_args.kwargs["ptc"], list(QUICKJS_PTC_TOOLS))
        self.assertNotIn("skills_backend", code_middleware.call_args.kwargs)
        kwargs = create_deep_agent.call_args.kwargs
        self.assertIn("auto-summary", kwargs["middleware"])
        self.assertIn("summary", kwargs["middleware"])
        self.assertTrue(any(isinstance(middleware, ProviderContextOverflowMiddleware) for middleware in kwargs["middleware"]))
        self.assertTrue(any(isinstance(middleware, ExecuteToolPromptMiddleware) for middleware in kwargs["middleware"]))
        self.assertIn("/.mira/skills", kwargs["skills"])
        self.assertEqual(kwargs["memory"][0], "/.mira/memories/AGENTS.md")
        self.assertTrue(any(subagent["name"] == "example-project-guide" for subagent in kwargs["subagents"]))
        self.assertTrue(any(tool.name == "grep" for tool in kwargs["tools"]))
        self.assertTrue(any(tool.name == "example_project_note" for tool in kwargs["tools"]))
        self.assertIn("memories", agent.mira_resources)
        self.assertIn("tools", agent.mira_resources)
        self.assertNotIn("execute", [tool["name"] for tool in agent.mira_tool_specs])
        self.assertNotIn("present_plan", [tool["name"] for tool in agent.mira_tool_specs])

    def test_factory_enables_execute_with_local_shell_backend_without_permissions(self) -> None:
        """Execute mode should expose execute and avoid incompatible filesystem permissions."""
        with tempfile.TemporaryDirectory() as directory:
            agent = type("Agent", (), {})()
            config = {"settings": {"hitl": {"tools": {"execute": {"enabled": True, "always_allow": False}}}}}
            with (
                patch("agent.factory.get_llm", return_value="model"),
                patch("agent.middleware.CodeInterpreterMiddleware", return_value="code"),
                patch("agent.middleware.create_mira_summarization_middleware", return_value="auto-summary"),
                patch("agent.middleware.create_mira_summarization_tool_middleware", return_value="summary"),
                patch("agent.factory.create_deep_agent", return_value=agent) as create_deep_agent,
            ):
                factory.build_agent(config, Path(directory), "checkpointer")

        kwargs = create_deep_agent.call_args.kwargs
        self.assertEqual(kwargs["permissions"], [])
        self.assertIsInstance(kwargs["backend"].default, LocalShellBackend)
        self.assertIn("execute", kwargs["interrupt_on"])
        self.assertIn("execute", [tool["name"] for tool in agent.mira_tool_specs])
        self.assertNotIn("present_plan", [tool["name"] for tool in agent.mira_tool_specs])

    def test_quickjs_ptc_tools_include_only_safe_project_exploration(self) -> None:
        """QuickJS PTC should expose read-only exploration tools, not writes or interrupts."""
        ptc_tools = set(QUICKJS_PTC_TOOLS)

        self.assertEqual(ptc_tools, {"ls", "read_file", "glob", "grep"})
        self.assertFalse({"task", "write_file", "edit_file", "execute", "ask_user", "present_plan"} & ptc_tools)

    def test_factory_registers_specific_and_provider_summarization_exclusions(self) -> None:
        """DeepAgents should exclude its hidden default summarization for resolved models."""
        model = type("Model", (), {})()
        with (
            patch("agent.factory.register_harness_profile") as register,
            patch("deepagents._models.get_model_provider", return_value="anyllm"),
            patch("deepagents._models.get_model_identifier", return_value="google/gemma"),
            patch.object(factory, "_REGISTERED_SUMMARIZATION_PROFILE_KEYS", set()),
        ):
            factory._register_summarization_exclusion({"llm_provider": "openai", "llm_model": "gpt-test"}, model)

        keys = [call.args[0] for call in register.call_args_list]
        self.assertEqual(keys, ["openai:gpt-test", "openai", "anyllm:google/gemma", "anyllm"])

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
