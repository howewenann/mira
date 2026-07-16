"""Top-level assembly for default and project resources."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepagents.backends import CompositeBackend, FilesystemBackend, LocalShellBackend

from agent.resources.memories import load_memories
from agent.resources.paths import DEFAULT_ROUTE, DEFAULTS_ROOT
from agent.resources.project_setup import ensure_project_examples
from agent.resources.skills import load_skills
from agent.resources.subagents import load_subagents
from agent.resources.tools import load_tools, tool_name
from agent.resources.tool_failures import ToolLoadFailure
from config.settings import EXECUTE_TOOL, execute_env_settings, tool_enabled

DEFAULT_EXECUTE_ENV_KEYS = (
    "PATH",
    "PATHEXT",
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "SYSTEMDRIVE",
    "PROGRAMDATA",
    "APPDATA",
    "LOCALAPPDATA",
    "USERPROFILE",
    "HOMEDRIVE",
    "HOMEPATH",
    "TEMP",
    "TMP",
)
EXECUTE_ENV_KEYS = DEFAULT_EXECUTE_ENV_KEYS


@dataclass(frozen=True)
class ResourceBundle:
    """Everything DeepAgents needs, plus metadata for REPL inspection."""

    backend: Any
    skills: list[str]
    memory: list[str]
    subagents: list[Any]
    tools: list[Any]
    metadata: dict[str, list[dict[str, str]]]
    tool_failures: list[ToolLoadFailure]


@dataclass(frozen=True)
class ResourceBackends:
    """Project files are writable; bundled defaults are mounted read-only."""

    project: Any
    combined: Any


def build_resources(
    workspace: Path,
    *,
    create_examples: bool = True,
    settings: dict[str, Any] | None = None,
    enable_execute: bool | None = None,
) -> ResourceBundle:
    """Build the resources passed into create_deep_agent()."""
    workspace = Path(workspace).expanduser().resolve()
    if create_examples:
        ensure_project_examples(workspace)

    backends = build_backends(workspace, settings=settings, enable_execute=enable_execute)

    memories = load_memories(workspace)
    skill_sources, skills = load_skills(workspace)
    subagents, subagent_info = load_subagents(workspace)
    tools, tool_info, tool_failures = load_tools(workspace, backends.project, settings)
    active_tools = enabled_tools(tools, tool_info, settings)

    return ResourceBundle(
        backend=backends.combined,
        skills=skill_sources,
        memory=[item["path"] for item in memories],
        subagents=subagents,
        tools=active_tools,
        metadata={
            "memories": memories,
            "skills": skills,
            "subagents": subagent_info,
            "tools": tool_info,
        },
        tool_failures=tool_failures,
    )


def enabled_tools(
    tools: list[Any],
    metadata: list[dict[str, str]],
    settings: dict[str, Any] | None,
) -> list[Any]:
    """Return tools that should be exposed to the agent."""
    metadata_by_name = {item["name"]: item for item in metadata}
    active = []
    for tool in tools:
        name = tool_name(tool)
        info = metadata_by_name.get(name, {})
        if info.get("source") == "project" and not tool_enabled(settings, name):
            continue
        active.append(tool)
    return active


def build_backends(
    workspace: Path,
    *,
    settings: dict[str, Any] | None = None,
    enable_execute: bool | None = None,
) -> ResourceBackends:
    execute_enabled = tool_enabled(settings, EXECUTE_TOOL) if enable_execute is None else bool(enable_execute)
    project_backend = (
        ProjectShellBackend(
            root_dir=workspace,
            virtual_mode=True,
            inherit_env=False,
            env=execute_env(settings=settings, workspace=workspace),
            execute_env_settings=execute_env_settings(settings),
            workspace=workspace,
        )
        if execute_enabled
        else FilesystemBackend(root_dir=workspace, virtual_mode=True)
    )
    defaults_backend = FilesystemBackend(root_dir=DEFAULTS_ROOT, virtual_mode=True)
    combined_backend = CompositeBackend(
        default=project_backend,
        routes={f"{DEFAULT_ROUTE}/": defaults_backend},
        artifacts_root="/.mira",
    )

    return ResourceBackends(project=project_backend, combined=combined_backend)


def execute_env(
    *,
    settings: dict[str, Any] | None = None,
    workspace: Path | None = None,
) -> dict[str, str]:
    """Return the small host environment exposed to execute commands."""
    env_settings = execute_env_settings(settings)
    names = [*DEFAULT_EXECUTE_ENV_KEYS, *env_settings.get("allow", [])]
    env = {key: os.environ[key] for key in names if os.environ.get(key)}
    if env_settings.get("mode") == "venv":
        apply_venv_env(env, env_settings, workspace)
    return env


class ProjectShellBackend(LocalShellBackend):
    """Local shell backend with project execute environment selection."""

    def __init__(
        self,
        *args: Any,
        execute_env_settings: dict[str, Any] | None = None,
        workspace: Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._execute_env_settings = execute_env_settings or {}
        self._workspace = Path(workspace or ".").expanduser().resolve()

    def execute(self, command: str, *, timeout: int | None = None) -> Any:
        """Execute a command through the configured project environment."""
        return super().execute(wrap_execute_command(command, self._execute_env_settings), timeout=timeout)


def wrap_execute_command(command: str, settings: dict[str, Any] | None) -> str:
    """Return a shell command wrapped for the configured execute environment."""
    prefix = conda_command_prefix({"hitl": {"execute_env": settings or {}}})
    if prefix:
        wrapped_prefix = " ".join(shell_arg(part) for part in prefix)
        return f"{wrapped_prefix} {shell_command(command)}"
    return command


def apply_venv_env(env: dict[str, str], settings: dict[str, Any], workspace: Path | None) -> None:
    """Apply venv PATH and VIRTUAL_ENV entries to an execute environment."""
    raw_path = str(settings.get("path") or "").strip()
    if not raw_path:
        return
    workspace = Path(workspace or ".").expanduser().resolve()
    venv_root, bin_dir = resolve_venv_paths(workspace, raw_path)
    if not bin_dir:
        return

    env["VIRTUAL_ENV"] = str(venv_root)
    current_path = env.get("PATH", "")
    separator = os.pathsep
    env["PATH"] = str(bin_dir) if not current_path else f"{bin_dir}{separator}{current_path}"


def resolve_venv_paths(workspace: Path, value: str) -> tuple[Path, Path | None]:
    """Return the venv root and executable directory for a path or python executable."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = workspace / path
    path = path.resolve()

    lowered = path.name.lower()
    if lowered in {"python", "python.exe"} and path.parent.name.lower() in {"scripts", "bin"}:
        return path.parent.parent, path.parent

    scripts = path / ("Scripts" if os.name == "nt" else "bin")
    return path, scripts


def project_python_command(settings: dict[str, Any] | None, workspace: Path) -> list[str]:
    """Return an argument-list prefix for Python in the configured execute environment."""
    selected = execute_env_settings(settings)
    prefix = conda_command_prefix(settings)
    if prefix:
        return [*prefix, "python"]
    mode = selected.get("mode")
    if mode == "venv" and selected.get("path"):
        root, bin_dir = resolve_venv_paths(workspace, str(selected["path"]))
        if bin_dir is not None:
            return [str(bin_dir / ("python.exe" if os.name == "nt" else "python"))]
        return [str(root)]
    return [sys.executable]


def conda_command_prefix(settings: dict[str, Any] | None) -> list[str]:
    """Return the shared Conda argument prefix for execute and project tools."""
    selected = execute_env_settings(settings)
    mode = selected.get("mode")
    if mode == "conda_name" and selected.get("name"):
        return ["conda", "run", "-n", str(selected["name"])]
    if mode == "conda_prefix" and selected.get("prefix"):
        return ["conda", "run", "-p", str(selected["prefix"])]
    return []


def project_environment_label(settings: dict[str, Any] | None, workspace: Path) -> str:
    """Return a concise display label for the configured execute environment."""
    selected = execute_env_settings(settings)
    mode = selected.get("mode")
    if mode == "conda_name":
        return str(selected.get("name") or "Conda (name not configured)")
    if mode == "conda_prefix":
        return str(selected.get("prefix") or "Conda (path not configured)")
    if mode == "venv":
        return str(selected.get("path") or "Venv (path not configured)")
    return "System"


def shell_arg(value: str) -> str:
    """Quote one argument for the current platform shell."""
    if os.name == "nt":
        return subprocess.list2cmdline([value])
    import shlex

    return shlex.quote(value)


def shell_command(command: str) -> str:
    """Return a shell invocation that keeps compound commands inside wrappers."""
    if os.name == "nt":
        return f"cmd /d /s /c {shell_arg(command)}"
    return f"sh -lc {shell_arg(command)}"
