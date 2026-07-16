"""LangChain proxies for metadata-only project-runtime tools."""

from __future__ import annotations

import inspect
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool, ToolException, create_schema_from_function

import mira_tool_api
from mira_tool_api import PROJECT_TOOL_METADATA_ATTRIBUTE, PROJECT_TOOL_METADATA_VERSION


def project_tools_from_module(
    module: Any,
    source_path: Path,
    workspace: Path,
    settings: dict[str, Any] | None,
) -> list[Any]:
    """Discover module-defined markers and wrap them as explicit StructuredTools."""
    tools: list[Any] = []
    for value in vars(module).values():
        if not inspect.isfunction(value) or value.__module__ != module.__name__:
            continue
        metadata = getattr(value, PROJECT_TOOL_METADATA_ATTRIBUTE, None)
        if not isinstance(metadata, dict):
            continue
        if metadata.get("version") != PROJECT_TOOL_METADATA_VERSION:
            raise ValueError(f"Unsupported @project_tool metadata version in {source_path}")
        public_name = str(metadata.get("name") or value.__name__)
        description = str(metadata.get("description") or inspect.getdoc(value) or "").strip()
        if not description:
            raise ValueError(f"@project_tool {public_name!r} must have a description or docstring")
        schema = create_schema_from_function(public_name, value)

        def _invoke_project_tool(
            _source_path: Path = source_path,
            _function_name: str = value.__name__,
            _public_name: str = public_name,
            **arguments: Any,
        ) -> Any:
            return invoke_project_tool(
                public_name=_public_name,
                source_path=_source_path,
                function_name=_function_name,
                arguments=arguments,
                workspace=workspace,
                settings=settings,
            )

        tools.append(
            StructuredTool.from_function(
                func=_invoke_project_tool,
                name=public_name,
                description=description,
                args_schema=schema,
            )
        )
        setattr(tools[-1], "__mira_project_runtime__", True)
        from agent.resources import project_environment_label

        setattr(tools[-1], "__mira_project_environment__", project_environment_label(settings, workspace))
    return tools


def invoke_project_tool(
    *,
    public_name: str,
    source_path: Path,
    function_name: str,
    arguments: dict[str, Any],
    workspace: Path,
    settings: dict[str, Any] | None,
) -> Any:
    """Run one function through the configured project Python process."""
    from agent.resources import execute_env, project_environment_label, project_python_command

    bridge_path = Path(mira_tool_api.__file__ or "").resolve()
    runner_path = Path(__file__).with_name("project_tool_runner.py").resolve()
    environment = project_environment_label(settings, workspace)
    request = {
        "source_path": str(source_path.resolve()),
        "function_name": function_name,
        "arguments": arguments,
        "workspace": str(workspace.resolve()),
        "bridge_path": str(bridge_path),
    }
    try:
        with tempfile.TemporaryDirectory(prefix="mira-project-tool-") as directory:
            request_path = Path(directory) / "request.json"
            response_path = Path(directory) / "response.json"
            request_path.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
            command = [
                *project_python_command(settings, workspace),
                str(runner_path),
                str(request_path),
                str(response_path),
            ]
            completed = subprocess.run(
                command,
                cwd=workspace,
                env=execute_env(settings=settings, workspace=workspace),
                capture_output=True,
                text=True,
                shell=False,
                check=False,
            )
            if not response_path.exists():
                details = (completed.stderr or completed.stdout or "child runner produced no response").strip()
                raise RuntimeError(details)
            response = json.loads(response_path.read_text(encoding="utf-8"))
    except ToolException:
        raise
    except BaseException as error:
        raise ToolException(project_tool_error(public_name, environment, type(error).__name__, str(error), "")) from error

    if response.get("ok") is True:
        return response.get("result")
    raise ToolException(
        project_tool_error(
            public_name,
            environment,
            str(response.get("exception_type") or "ProjectToolError"),
            str(response.get("message") or "project tool failed"),
            str(response.get("traceback") or ""),
        )
    )


def project_tool_error(name: str, environment: str, error_type: str, message: str, details: str) -> str:
    lines = [
        f"{name} failed in the project environment.",
        "Runtime: Project",
        f"Environment: {environment}",
        "",
        f"{error_type}: {message}",
    ]
    if details:
        lines.extend(("", "Diagnostic details:", details.rstrip()))
    return "\n".join(lines)
