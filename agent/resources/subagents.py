"""Discovery for DeepAgents subagent files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.resources.items import display_item, merge_project_overrides
from agent.resources.paths import (
    SUBAGENTS_DIR,
    default_dir,
    default_virtual_dir,
    project_dir,
    project_virtual_dir,
)
from agent.resources.python_files import import_python_file


def load_subagents(workspace: Path) -> tuple[list[Any], list[dict[str, str]]]:
    defaults = subagent_files(default_dir(SUBAGENTS_DIR), default_virtual_dir(SUBAGENTS_DIR), "default")
    projects = subagent_files(project_dir(workspace, SUBAGENTS_DIR), project_virtual_dir(SUBAGENTS_DIR), "project")

    merged = merge_project_overrides(defaults, projects)
    return [item["subagent"] for item in merged], [display_item(item) for item in merged]


def subagent_files(root: Path, virtual_root: str, source: str) -> list[dict[str, Any]]:
    if not root.exists():
        return []

    items = []
    for path in sorted(root.glob("*.py")):
        for subagent in subagents_from_file(path):
            items.append(
                {
                    "name": subagent_name(subagent),
                    "path": f"{virtual_root}/{path.name}",
                    "source": source,
                    "replaces": "",
                    "subagent": subagent,
                }
            )

    return items


def subagents_from_file(path: Path) -> list[Any]:
    module = import_python_file(path, "mira_resource_subagents")
    subagents = getattr(module, "SUBAGENTS", [])
    if not isinstance(subagents, list | tuple):
        raise TypeError(f"{path} must define SUBAGENTS as a list")

    return list(subagents)


def subagent_name(subagent: Any) -> str:
    if isinstance(subagent, dict):
        name = subagent.get("name")
    else:
        name = getattr(subagent, "name", None)

    if not name:
        raise ValueError(f"Subagent is missing a name: {subagent!r}")
    return str(name)
