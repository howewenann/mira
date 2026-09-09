"""Discovery for DeepAgents skill folders."""

from __future__ import annotations

from typing import Any

from deepagents.middleware.skills import SkillMetadata, _list_skills

from agent.resources.items import merge_project_overrides
from agent.resources.paths import (
    SKILLS_DIR,
    default_virtual_dir,
    project_virtual_dir,
)


def load_skills(backend: Any) -> tuple[list[str], list[dict[str, str]]]:
    """Discover MIRA skill roots with DeepAgents' canonical parser."""
    default_source = default_virtual_dir(SKILLS_DIR)
    project_source = project_virtual_dir(SKILLS_DIR)

    defaults = skill_files(backend, default_source, "default")
    projects = skill_files(backend, project_source, "project")

    sources = []
    if defaults:
        sources.append(default_source)
    if projects:
        sources.append(project_source)

    return sources, merge_project_overrides(defaults, projects)


def skill_files(backend: Any, virtual_root: str, source: str) -> list[dict[str, str]]:
    """Project DeepAgents metadata into MIRA's display shape."""
    return [skill_item(skill, source) for skill in _list_skills(backend, virtual_root)]


def skill_item(skill: SkillMetadata, source: str) -> dict[str, str]:
    return {
        "name": skill["name"],
        "description": skill["description"],
        "path": skill["path"],
        "source": source,
        "replaces": "",
    }
