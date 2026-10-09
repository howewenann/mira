"""Discovery for DeepAgents skill folders."""

from __future__ import annotations

from typing import Any

from deepagents.middleware.skills import SkillMetadata, SkillsMiddleware, _list_skills

from agent.resources.paths import (
    SKILLS_DIR,
    default_virtual_dir,
    project_virtual_dir,
)


def load_skills(backend: Any) -> tuple[list[str], list[dict[str, str]], set[str]]:
    """Project DeepAgents' effective skills into MIRA's resource display."""
    default_source = default_virtual_dir(SKILLS_DIR)
    project_source = project_virtual_dir(SKILLS_DIR)
    middleware = SkillsMiddleware(backend=backend, sources=[default_source, project_source])
    loaded = middleware.before_agent({}, None, {})
    effective = loaded["skills_metadata"] if loaded is not None else []
    default_names = {skill["name"] for skill in _list_skills(backend, default_source)}
    sources = [
        root for root in (default_source, project_source)
        if any(skill["path"].startswith(f"{root}/") for skill in effective)
    ]
    included_tools = {
        name
        for skill in effective
        for name in (skill.get("metadata") or {}).get("include_tools", "").split()
    }
    return sources, [
        skill_item(
            skill,
            "project" if skill["path"].startswith(f"{project_source}/") else "default",
            replaces="default" if skill["name"] in default_names and skill["path"].startswith(f"{project_source}/") else "",
        )
        for skill in effective
    ], included_tools


def skill_item(skill: SkillMetadata, source: str, *, replaces: str = "") -> dict[str, str]:
    return {
        "name": skill["name"],
        "description": skill["description"],
        "path": skill["path"],
        "source": source,
        "replaces": replaces,
    }
