"""Top-level assembly for default and project resources."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepagents.backends import CompositeBackend, FilesystemBackend

from agent.resources.memories import load_memories
from agent.resources.paths import DEFAULT_ROUTE, DEFAULTS_ROOT
from agent.resources.project_setup import ensure_project_examples
from agent.resources.skills import load_skills
from agent.resources.subagents import load_subagents
from agent.resources.tools import load_tools


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


def build_resources(workspace: Path, *, create_examples: bool = True) -> ResourceBundle:
    """Build the resources passed into create_deep_agent()."""
    workspace = Path(workspace).expanduser().resolve()
    if create_examples:
        ensure_project_examples(workspace)

    backends = build_backends(workspace)

    memories = load_memories(workspace)
    skill_sources, skills = load_skills(workspace)
    subagents, subagent_info = load_subagents(workspace)
    tools, tool_info = load_tools(workspace, backends.project)

    return ResourceBundle(
        backend=backends.combined,
        skills=skill_sources,
        memory=[item["path"] for item in memories],
        subagents=subagents,
        tools=tools,
        metadata={
            "memories": memories,
            "skills": skills,
            "subagents": subagent_info,
            "tools": tool_info,
        },
    )


def build_backends(workspace: Path) -> ResourceBackends:
    project_backend = FilesystemBackend(root_dir=workspace, virtual_mode=True)
    defaults_backend = FilesystemBackend(root_dir=DEFAULTS_ROOT, virtual_mode=True)
    combined_backend = CompositeBackend(
        default=project_backend,
        routes={f"{DEFAULT_ROUTE}/": defaults_backend},
        artifacts_root="/.mira",
    )

    return ResourceBackends(project=project_backend, combined=combined_backend)
