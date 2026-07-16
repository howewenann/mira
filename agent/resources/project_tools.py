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
    project_backend: Any,
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
        path_parameters = workspace_path_parameters(inspect.signature(value))

        def _invoke_project_tool(
            _source_path: Path = source_path,
            _function_name: str = value.__name__,
            _public_name: str = public_name,
            _path_parameters: frozenset[str] = path_parameters,
            _project_backend: Any = project_backend,
            **arguments: Any,
        ) -> Any:
            return invoke_project_tool(
                public_name=_public_name,
                source_path=_source_path,
                function_name=_function_name,
                arguments=arguments,
                workspace=workspace,
                settings=settings,
                project_backend=_project_backend,
                path_parameters=_path_parameters,
            )

        tools.append(
            StructuredTool.from_function(
                func=_invoke_project_tool,
                name=public_name,
                description=description,
                args_schema=schema,
                handle_tool_error=True,
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
    project_backend: Any,
    path_parameters: frozenset[str],
) -> Any:
    """Run one function through the configured project Python process."""
    from agent.resources import execute_env, project_environment_label, project_python_command

    bridge_path = Path(mira_tool_api.__file__ or "").resolve()
    runner_path = Path(__file__).with_name("project_tool_runner.py").resolve()
    environment = project_environment_label(settings, workspace)
    request = {
        "source_path": str(source_path.resolve()),
        "function_name": function_name,
        "arguments": resolve_workspace_path_arguments(arguments, path_parameters, project_backend),
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


def workspace_path_parameters(signature: inspect.Signature) -> frozenset[str]:
    """Return conventional structured arguments that carry workspace paths."""
    return frozenset(
        name
        for name in signature.parameters
        if name == "path" or name == "paths" or name.endswith("_path") or name.endswith("_paths")
    )


def resolve_workspace_path_arguments(
    arguments: dict[str, Any],
    path_parameters: frozenset[str],
    project_backend: Any,
) -> dict[str, Any]:
    """Translate MIRA virtual paths to host paths before crossing the child boundary."""
    return {
        name: resolve_workspace_path_value(value, project_backend) if name in path_parameters else value
        for name, value in arguments.items()
    }


def resolve_workspace_path_value(value: Any, project_backend: Any) -> Any:
    """Resolve leading-slash virtual paths while preserving other argument values."""
    if isinstance(value, str):
        if not value.startswith("/"):
            return value
        resolver = getattr(project_backend, "_resolve_path", None)
        return str(resolver(value)) if callable(resolver) else value
    if isinstance(value, list):
        return [resolve_workspace_path_value(item, project_backend) for item in value]
    return value
