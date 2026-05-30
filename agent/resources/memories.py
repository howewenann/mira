"""Discovery for MIRA memory files."""

from __future__ import annotations

from pathlib import Path

from agent.resources.items import merge_project_overrides
from agent.resources.paths import (
    MEMORIES_DIR,
    default_dir,
    default_virtual_dir,
    project_dir,
    project_virtual_dir,
)


def load_memories(workspace: Path) -> list[dict[str, str]]:
    defaults = memory_files(default_dir(MEMORIES_DIR), default_virtual_dir(MEMORIES_DIR), "default")
    projects = memory_files(project_dir(workspace, MEMORIES_DIR), project_virtual_dir(MEMORIES_DIR), "project")
    return merge_project_overrides(defaults, projects)


def memory_files(root: Path, virtual_root: str, source: str) -> list[dict[str, str]]:
    if not root.exists():
        return []

    return [
        {
            "name": path.name,
            "path": f"{virtual_root}/{path.name}",
            "source": source,
            "replaces": "",
        }
        for path in sorted(root.glob("*.md"))
    ]
