"""Discovery for DeepAgents skill folders."""

from __future__ import annotations

from pathlib import Path

from agent.resources.items import merge_project_overrides
from agent.resources.paths import (
    SKILLS_DIR,
    default_dir,
    default_virtual_dir,
    project_dir,
    project_virtual_dir,
)


def load_skills(workspace: Path) -> tuple[list[str], list[dict[str, str]]]:
    default_source = default_virtual_dir(SKILLS_DIR)
    project_source = project_virtual_dir(SKILLS_DIR)

    defaults = skill_files(default_dir(SKILLS_DIR), default_source, "default")
    projects = skill_files(project_dir(workspace, SKILLS_DIR), project_source, "project")

    sources = []
    if defaults:
        sources.append(default_source)
    if projects:
        sources.append(project_source)

    return sources, merge_project_overrides(defaults, projects)


def skill_files(root: Path, virtual_root: str, source: str) -> list[dict[str, str]]:
    if not root.exists():
        return []

    items = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        skill_file = directory / "SKILL.md"
        if not skill_file.exists():
            continue

        items.append(
            {
                "name": frontmatter_name(skill_file) or directory.name,
                "path": f"{virtual_root}/{directory.name}/SKILL.md",
                "source": source,
                "replaces": "",
            }
        )

    return items


def frontmatter_name(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return ""

    parts = text.split("---", 2)
    if len(parts) < 3:
        return ""

    for line in parts[1].splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() == "name":
            return value.strip().strip("\"'")

    return ""
