"""Top-level assembly for default and project resources."""

from __future__ import annotations

import os
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
from config.settings import EXECUTE_TOOL, tool_enabled

EXECUTE_ENV_KEYS = (
    "PATH",
    "PATHEXT",
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "USERPROFILE",
    "HOMEDRIVE",
    "HOMEPATH",
    "TEMP",
    "TMP",
)


@dataclass(frozen=True)
class ResourceBundle:
    """Everything DeepAgents needs, plus metadata for REPL inspection."""

    backend: Any
    skills: list[str]
    memory: list[str]
    subagents: list[Any]
    tools: list[Any]
    metadata: dict[str, list[dict[str, str]]]


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
    tools, tool_info = load_tools(workspace, backends.project)
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
        LocalShellBackend(
            root_dir=workspace,
            virtual_mode=True,
            inherit_env=False,
            env=execute_env(),
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


def execute_env() -> dict[str, str]:
    """Return the small host environment exposed to execute commands."""
    return {key: os.environ[key] for key in EXECUTE_ENV_KEYS if os.environ.get(key)}
