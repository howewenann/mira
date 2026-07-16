"""Focused tests for resilient custom tools and project-runtime proxies."""

from __future__ import annotations

import sys
import subprocess
import tempfile
import unittest
from hashlib import sha1
from pathlib import Path
from unittest.mock import patch

from deepagents.backends import FilesystemBackend
from langchain_core.tools import ToolException

from agent import factory
from agent.resources import build_resources, project_python_command
from agent.resources.project_setup import ensure_project_examples
from agent.resources.python_files import import_python_file
from agent.resources.tool_failures import missing_requirements, one_shot_warning
from agent.resources.tools import load_tools
from mira_tool_api import PROJECT_TOOL_METADATA_VERSION, project_tool


class ProjectToolApiTests(unittest.TestCase):
    def test_decorator_preserves_callable_and_metadata(self) -> None:
        @project_tool(name="public_name", description="Public description.")
        def implementation(value: int = 3) -> int:
            return value + 1

        self.assertEqual(implementation(), 4)
        metadata = implementation.__mira_project_tool__
        self.assertEqual(metadata["version"], PROJECT_TOOL_METADATA_VERSION)
        self.assertEqual(metadata["name"], "public_name")
        self.assertEqual(metadata["description"], "Public description.")

    def test_decorator_rejects_empty_name_and_invalid_usage(self) -> None:
        with self.assertRaises(ValueError):
            project_tool(name="  ")
        with self.assertRaises(TypeError):
            project_tool("not callable")  # type: ignore[arg-type]

    def test_api_imports_without_langchain_or_mira_dependencies(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-S",
                "-c",
                "import mira_tool_api, sys; print('langchain_core' in sys.modules)",
            ],
            cwd=Path(__file__).parents[1],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "False")


class ResilientToolLoadingTests(unittest.TestCase):
    def test_project_files_are_isolated_and_failures_are_structured(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            tools = workspace / ".mira" / "tools"
            tools.mkdir(parents=True)
            (tools / "good.py").write_text(
                "from langchain_core.tools import tool\n\n@tool\ndef available() -> str:\n"
                '    """Return success."""\n    return "ok"\n',
                encoding="utf-8",
            )
            (tools / "alpha.py").write_text("import mira_missing_alpha\n", encoding="utf-8")
            (tools / "beta.py").write_text("import mira_missing_beta\n", encoding="utf-8")
            (tools / "broken.py").write_text("def broken(\n    return 1\n", encoding="utf-8")

            resources = build_resources(workspace, create_examples=False)

            self.assertIn("available", [tool.name for tool in resources.tools])
            self.assertEqual(len(resources.tool_failures), 3)
            self.assertEqual(
                {failure.exception_type for failure in resources.tool_failures},
                {"ModuleNotFoundError", "SyntaxError"},
            )
            self.assertNotIn("mira_missing_alpha", [item["name"] for item in resources.metadata["tools"]])
            alpha = next(failure for failure in resources.tool_failures if failure.missing_module == "mira_missing_alpha")
            self.assertEqual(alpha.display_path, ".mira/tools/alpha.py")
            self.assertEqual(alpha.line_number, 1)
            self.assertEqual(alpha.source_line, "import mira_missing_alpha")
            self.assertIn("ModuleNotFoundError", alpha.traceback_text)

    def test_shared_dependency_is_listed_per_file_but_suggested_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            tools = workspace / ".mira" / "tools"
            tools.mkdir(parents=True)
            for name in ("one.py", "two.py"):
                (tools / name).write_text("import mira_missing_shared\n", encoding="utf-8")
            failures = build_resources(workspace, create_examples=False).tool_failures
            self.assertEqual(len(failures), 2)
            self.assertEqual(missing_requirements(failures), ["mira_missing_shared"])

    def test_import_error_is_not_treated_as_missing_package(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            tools = workspace / ".mira" / "tools"
            tools.mkdir(parents=True)
            (tools / "wrong_export.py").write_text("from pathlib import definitely_missing\n", encoding="utf-8")
            failure = build_resources(workspace, create_examples=False).tool_failures[0]
            self.assertEqual(failure.exception_type, "ImportError")
            self.assertEqual(failure.missing_module, "")
            self.assertEqual(missing_requirements([failure]), [])

    def test_error_location_prefers_deep_workspace_frame(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            tools = workspace / ".mira" / "tools"
            tools.mkdir(parents=True)
            (workspace / "workspace_helper.py").write_text(
                "VALUE = 1\nraise RuntimeError('helper failed')\n", encoding="utf-8"
            )
            (tools / "main.py").write_text("import workspace_helper\n", encoding="utf-8")
            sys.path.insert(0, str(workspace))
            try:
                failure = build_resources(workspace, create_examples=False).tool_failures[0]
            finally:
                sys.path.remove(str(workspace))
                sys.modules.pop("workspace_helper", None)
            self.assertEqual(failure.display_path, "workspace_helper.py")
            self.assertEqual(failure.line_number, 2)
            self.assertEqual(failure.source_line, "raise RuntimeError('helper failed')")
            self.assertEqual(failure.source_path.name, "main.py")

    def test_default_tool_failure_remains_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "defaults"
            workspace = Path(directory) / "workspace"
            root.mkdir()
            workspace.mkdir()
            (root / "bad.py").write_text("import mira_missing_builtin\n", encoding="utf-8")
            with patch("agent.resources.tools.default_dir", return_value=root):
                with self.assertRaises(ModuleNotFoundError):
                    load_tools(workspace, FilesystemBackend(root_dir=workspace, virtual_mode=True))

    def test_failed_import_is_removed_and_retry_is_clean(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resource.py"
            path.write_text("raise RuntimeError('first')\n", encoding="utf-8")
            module_id = sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
            module_name = f"test_resource_{module_id}"
            with self.assertRaises(RuntimeError):
                import_python_file(path, "test_resource")
            self.assertNotIn(module_name, sys.modules)
            path.write_text("VALUE = 42\n", encoding="utf-8")
            self.assertEqual(import_python_file(path, "test_resource").VALUE, 42)

    def test_grouped_one_shot_warning_keeps_execution_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            tools = workspace / ".mira" / "tools"
            tools.mkdir(parents=True)
            (tools / "bad.py").write_text("import mira_missing_warning\n", encoding="utf-8")
            warning = one_shot_warning(build_resources(workspace, create_examples=False).tool_failures)
            self.assertIn("Warning: 1 project tool file", warning)
            self.assertIn("mira_missing_warning", warning)
            self.assertIn("Normal @tool dependencies run in MIRA's environment", warning)
            self.assertIn(".mira/examples/tools/project_runtime_tool.py", warning)

    def test_factory_exposes_only_tools_from_successful_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            tools = workspace / ".mira" / "tools"
            tools.mkdir(parents=True)
            (tools / "good.py").write_text(
                "from langchain_core.tools import tool\n@tool\ndef safe_tool() -> str:\n"
                '    """A safe loaded tool."""\n    return "safe"\n',
                encoding="utf-8",
            )
            (tools / "bad.py").write_text(
                "import mira_missing_factory\nfrom langchain_core.tools import tool\n"
                "@tool\ndef broken_tool() -> str:\n"
                '    """Must never be exposed."""\n    return "broken"\n',
                encoding="utf-8",
            )
            agent = type("Agent", (), {})()
            with (
                patch("agent.factory.get_llm", return_value="model"),
                patch("agent.middleware.CodeInterpreterMiddleware", return_value="code"),
                patch("agent.middleware.create_mira_summarization_middleware", return_value="auto-summary"),
                patch("agent.middleware.create_mira_summarization_tool_middleware", return_value="summary"),
                patch("agent.factory.create_deep_agent", return_value=agent) as create,
            ):
                factory.build_agent({}, workspace, "checkpointer")
            passed_names = [tool.name for tool in create.call_args.kwargs["tools"]]
            spec_names = [item["name"] for item in agent.mira_tool_specs]
            self.assertIn("safe_tool", passed_names)
            self.assertNotIn("broken_tool", passed_names)
            self.assertNotIn("broken_tool", spec_names)
            self.assertEqual(len(agent.mira_tool_failures), 1)


class ProjectToolProxyTests(unittest.TestCase):
    def test_python_command_reuses_execute_environment_selection(self) -> None:
        workspace = Path("workspace").resolve()
        self.assertEqual(
            project_python_command(
                {"hitl": {"execute_env": {"mode": "conda_name", "name": "analytics"}}}, workspace
            ),
            ["conda", "run", "-n", "analytics", "python"],
        )
        venv = project_python_command(
            {"hitl": {"execute_env": {"mode": "venv", "path": ".venv"}}}, workspace
        )
        self.assertEqual(Path(venv[0]).name, "python.exe" if sys.platform == "win32" else "python")

    def test_public_project_name_keeps_existing_project_approval_policy(self) -> None:
        interrupts = factory._write_interrupts(
            {
                "settings": {
                    "hitl": {
                        "tools": {"public_project": {"enabled": True, "always_allow": False}}
                    }
                }
            },
            [{"name": "public_project", "source": "project", "path": "/.mira/tools/project.py"}],
        )
        self.assertIn("public_project", interrupts)
        self.assertNotIn("_invoke_project_tool", interrupts)

    def test_proxy_uses_public_identity_schema_and_child_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            tools = workspace / ".mira" / "tools"
            tools.mkdir(parents=True)
            (workspace / "project_only_module.py").write_text(
                'def decorate(value):\n    return f"project:{value}"\n', encoding="utf-8"
            )
            (tools / "project.py").write_text(
                "from mira_tool_api import project_tool\n\n"
                '@project_tool(name="public_project", description="Run in project Python.")\n'
                "def implementation(value: str, repeat: int = 1) -> str:\n"
                "    from project_only_module import decorate\n"
                "    return decorate(value) * repeat\n",
                encoding="utf-8",
            )

            resources = build_resources(workspace, create_examples=False)
            tool = next(tool for tool in resources.tools if tool.name == "public_project")
            metadata = next(item for item in resources.metadata["tools"] if item["name"] == "public_project")

            self.assertEqual(tool.name, "public_project")
            self.assertNotEqual(tool.name, "_invoke_project_tool")
            self.assertEqual(tool.args_schema.model_fields["repeat"].default, 1)
            self.assertEqual(metadata["runtime"], "Project")
            self.assertEqual(metadata["environment"], "System")
            self.assertEqual(tool.invoke({"value": "x", "repeat": 2}), "project:xproject:x")

    def test_marked_original_in_tools_export_does_not_create_second_tool(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            tools = workspace / ".mira" / "tools"
            tools.mkdir(parents=True)
            (tools / "declared.py").write_text(
                "from mira_tool_api import project_tool\n"
                "@project_tool(name='public_declared')\n"
                "def implementation(value: str) -> str:\n"
                '    """Return a value."""\n    return value\n'
                "TOOLS = [implementation]\n",
                encoding="utf-8",
            )
            names = [
                tool.name
                for tool in build_resources(workspace, create_examples=False).tools
                if tool.name in {"public_declared", "implementation"}
            ]
            self.assertEqual(names, ["public_declared"])

    def test_non_json_result_falls_back_and_child_error_is_a_tool_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            tools = workspace / ".mira" / "tools"
            tools.mkdir(parents=True)
            (tools / "project.py").write_text(
                "from mira_tool_api import project_tool\n\n"
                "@project_tool\ndef make_set() -> object:\n"
                '    """Return a non-JSON value."""\n    return {2, 1}\n\n'
                "@project_tool\ndef explode() -> str:\n"
                '    """Raise a project error."""\n    raise RuntimeError("child boom")\n',
                encoding="utf-8",
            )
            resources = build_resources(workspace, create_examples=False)
            by_name = {tool.name: tool for tool in resources.tools}
            self.assertEqual(by_name["make_set"].invoke({}), "{1, 2}")
            with self.assertRaises(ToolException) as caught:
                by_name["explode"].invoke({})
            self.assertIn("explode failed in the project environment", str(caught.exception))
            self.assertIn("Runtime: Project", str(caught.exception))
            self.assertIn("RuntimeError: child boom", str(caught.exception))

    def test_async_project_function_is_awaited_in_child(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            tools = workspace / ".mira" / "tools"
            tools.mkdir(parents=True)
            (tools / "async_tool.py").write_text(
                "from mira_tool_api import project_tool\n"
                "@project_tool\nasync def async_project(value: int) -> int:\n"
                '    """Double a value asynchronously."""\n    return value * 2\n',
                encoding="utf-8",
            )
            tool = next(
                tool
                for tool in build_resources(workspace, create_examples=False).tools
                if tool.name == "async_project"
            )
            self.assertEqual(tool.invoke({"value": 4}), 8)

    def test_imported_marked_function_is_not_rediscovered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            tools = workspace / ".mira" / "tools"
            tools.mkdir(parents=True)
            helper = workspace / "helper_marked.py"
            helper.write_text(
                "from mira_tool_api import project_tool\n@project_tool\ndef imported() -> str:\n"
                '    """Imported helper."""\n    return "no"\n',
                encoding="utf-8",
            )
            (tools / "main.py").write_text(
                "from helper_marked import imported\nfrom mira_tool_api import project_tool\n"
                "@project_tool\ndef local() -> str:\n"
                '    """Local helper."""\n    return "yes"\n',
                encoding="utf-8",
            )
            sys.path.insert(0, str(workspace))
            try:
                resources = build_resources(workspace, create_examples=False)
            finally:
                sys.path.remove(str(workspace))
                sys.modules.pop("helper_marked", None)
            names = [tool.name for tool in resources.tools]
            self.assertIn("local", names)
            self.assertNotIn("imported", names)


class ProjectExamplesTests(unittest.TestCase):
    def test_examples_are_inert_and_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            ensure_project_examples(workspace)
            project_example = workspace / ".mira" / "examples" / "tools" / "project_runtime_tool.py"
            project_example.write_text("custom", encoding="utf-8")
            ensure_project_examples(workspace)
            self.assertEqual(project_example.read_text(encoding="utf-8"), "custom")
            self.assertEqual(list((workspace / ".mira" / "tools").glob("*.py")), [])
            self.assertFalse(build_resources(workspace, create_examples=False).tool_failures)


if __name__ == "__main__":
    unittest.main()
