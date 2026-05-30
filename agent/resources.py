"""Discovery for MIRA default and project resources."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any

from deepagents.backends import CompositeBackend, FilesystemBackend

from agent.tool_resources import discover_tools

DEFAULT_ROUTE = "/mira-defaults/"
PROJECT_DIR = ".mira"
MEMORIES_DIR = "memories"
SKILLS_DIR = "skills"
SUBAGENTS_DIR = "subagents"
TOOLS_DIR = "tools"

DEFAULTS_ROOT = Path(__file__).parent / "defaults"


@dataclass(frozen=True)
class ResourceBundle:
    """Resources passed to DeepAgents plus display metadata for the REPL."""

    backend: Any
    skills: list[str]
    memory: list[str]
    subagents: list[Any]
    tools: list[Any]
    metadata: dict[str, list[dict[str, str]]]


def build_resources(workspace: Path, *, create_examples: bool = True) -> ResourceBundle:
    """Create project examples, discover resources, and build the backend."""
    workspace = Path(workspace).expanduser().resolve()
    if create_examples:
        ensure_project_examples(workspace)

    project_backend = FilesystemBackend(root_dir=workspace, virtual_mode=True)
    defaults_backend = FilesystemBackend(root_dir=DEFAULTS_ROOT, virtual_mode=True)
    backend = CompositeBackend(default=project_backend, routes={DEFAULT_ROUTE: defaults_backend})

    memory_items = discover_memories(workspace)
    skill_sources, skill_items = discover_skills(workspace)
    subagents, subagent_items = discover_subagents(workspace)
    tools, tool_items = discover_tools(
        DEFAULTS_ROOT / TOOLS_DIR,
        workspace / PROJECT_DIR / TOOLS_DIR,
        project_backend,
    )

    return ResourceBundle(
        backend=backend,
        skills=skill_sources,
        memory=[item["path"] for item in memory_items],
        subagents=subagents,
        tools=tools,
        metadata={
            "memories": memory_items,
            "skills": skill_items,
            "subagents": subagent_items,
            "tools": tool_items,
        },
    )


def ensure_project_examples(workspace: Path) -> None:
    """Create the editable `.mira` examples without overwriting user files."""
    mira_dir = workspace / PROJECT_DIR
    memories_dir = mira_dir / MEMORIES_DIR
    skills_dir = mira_dir / SKILLS_DIR / "example-skill"
    subagents_dir = mira_dir / SUBAGENTS_DIR
    tools_dir = mira_dir / TOOLS_DIR

    memories_dir.mkdir(parents=True, exist_ok=True)
    skills_dir.mkdir(parents=True, exist_ok=True)
    subagents_dir.mkdir(parents=True, exist_ok=True)
    tools_dir.mkdir(parents=True, exist_ok=True)

    write_example(
        mira_dir / "README.md",
        """# MIRA Project Resources

MIRA loads project resources from this folder on top of its defaults.

- `memories/*.md`: always-on project context. A file with the same name as a
  default memory replaces the default.
- `skills/<skill>/SKILL.md`: DeepAgents skills. Project skills override default
  skills when the frontmatter `name` is the same.
- `subagents/*.py`: Python files that export `SUBAGENTS = [...]`. Project
  subagents override default subagents when the `name` is the same.
- `tools/*.py`: Python files that export `TOOLS = [...]` or
  `get_tools(project_backend)`. Project tools override defaults when the tool
  `name` is the same.

Use `/memories`, `/skills`, `/subagents`, and `/tools` in the REPL to inspect
what MIRA loaded.
""",
    )
    write_example(
        memories_dir / "AGENTS.md",
        """# Project Memory

Describe this project's commands, style, architecture, and preferences here.
MIRA loads this file instead of its default `AGENTS.md`.
""",
    )
    write_example(
        skills_dir / "SKILL.md",
        """---
name: example-skill
description: Example project skill showing where local workflows belong.
---

# Example Skill

Use this file for a project-specific workflow. Rename the folder and `name`
when you turn it into a real skill.
""",
    )
    write_example(
        subagents_dir / "example_subagent.py",
        '''"""Example project subagent.

Edit or delete this file when you know which project helpers you want.
"""

SUBAGENTS = [
    {
        "name": "project-guide",
        "description": "Answer questions about this project's local conventions.",
        "system_prompt": (
            "You are a project guide. Inspect relevant files before answering "
            "and keep guidance specific to this workspace."
        ),
    }
]
''',
    )
    write_example(
        tools_dir / "example_tool.py",
        '''"""Example project tool.

Edit or delete this file when you know which project helpers you want.
"""

from langchain.tools import tool


@tool
def project_note() -> str:
    """Return a short note proving project tools are loaded."""
    return "Project tool loaded."


TOOLS = [project_note]
''',
    )


def write_example(path: Path, content: str) -> None:
    """Write an example file only when it does not already exist."""
    if path.exists():
        return
    path.write_text(content, encoding="utf-8")


def discover_memories(workspace: Path) -> list[dict[str, str]]:
    """Return final memory files after project files replace defaults by name."""
    defaults = memory_items(DEFAULTS_ROOT / MEMORIES_DIR, f"{DEFAULT_ROUTE}{MEMORIES_DIR}", "default")
    projects = memory_items(workspace / PROJECT_DIR / MEMORIES_DIR, f"/{PROJECT_DIR}/{MEMORIES_DIR}", "project")
    return merge_items(defaults, projects)


def memory_items(root: Path, virtual_root: str, source: str) -> list[dict[str, str]]:
    """List Markdown memories in one resource folder."""
    if not root.exists():
        return []

    items = []
    for path in sorted(root.glob("*.md")):
        items.append(
            {
                "name": path.name,
                "path": f"{virtual_root}/{path.name}",
                "source": source,
                "replaces": "",
            }
        )
    return items


def discover_skills(workspace: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Return skill source paths and final display metadata."""
    default_root = DEFAULTS_ROOT / SKILLS_DIR
    project_root = workspace / PROJECT_DIR / SKILLS_DIR
    default_source = f"{DEFAULT_ROUTE}{SKILLS_DIR}"
    project_source = f"/{PROJECT_DIR}/{SKILLS_DIR}"

    defaults = skill_items(default_root, default_source, "default")
    projects = skill_items(project_root, project_source, "project")

    sources = []
    if defaults:
        sources.append(default_source)
    if projects:
        sources.append(project_source)

    return sources, merge_items(defaults, projects)


def skill_items(root: Path, virtual_root: str, source: str) -> list[dict[str, str]]:
    """List skills from directories that contain `SKILL.md`."""
    if not root.exists():
        return []

    items = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        skill_file = directory / "SKILL.md"
        if not skill_file.exists():
            continue
        name = skill_name(skill_file) or directory.name
        items.append(
            {
                "name": name,
                "path": f"{virtual_root}/{directory.name}/SKILL.md",
                "source": source,
                "replaces": "",
            }
        )
    return items


def skill_name(path: Path) -> str:
    """Read the simple `name:` field from skill frontmatter."""
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


def discover_subagents(workspace: Path) -> tuple[list[Any], list[dict[str, str]]]:
    """Load default and project subagents, merging by subagent name."""
    default_root = DEFAULTS_ROOT / SUBAGENTS_DIR
    project_root = workspace / PROJECT_DIR / SUBAGENTS_DIR
    defaults = subagent_items(default_root, f"{DEFAULT_ROUTE}{SUBAGENTS_DIR}", "default")
    projects = subagent_items(project_root, f"/{PROJECT_DIR}/{SUBAGENTS_DIR}", "project")

    merged = merge_subagent_items(defaults, projects)
    return [item["subagent"] for item in merged], [display_item(item) for item in merged]


def subagent_items(root: Path, virtual_root: str, source: str) -> list[dict[str, Any]]:
    """Import subagent modules from one folder."""
    if not root.exists():
        return []

    items = []
    for path in sorted(root.glob("*.py")):
        for subagent in load_subagents(path):
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


def load_subagents(path: Path) -> list[Any]:
    """Import one Python file and return its `SUBAGENTS` list."""
    module_id = sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    module_name = f"mira_resource_subagents_{module_id}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import subagents from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    subagents = getattr(module, "SUBAGENTS", [])
    if not isinstance(subagents, list | tuple):
        raise TypeError(f"{path} must define SUBAGENTS as a list")

    return list(subagents)


def subagent_name(subagent: Any) -> str:
    """Return a DeepAgents subagent name from a dict-like or object-like value."""
    if isinstance(subagent, dict):
        name = subagent.get("name")
    else:
        name = getattr(subagent, "name", None)

    if not name:
        raise ValueError(f"Subagent is missing a name: {subagent!r}")
    return str(name)


def merge_items(defaults: list[dict[str, str]], projects: list[dict[str, str]]) -> list[dict[str, str]]:
    """Merge display items by name with project entries replacing defaults."""
    merged = {item["name"]: dict(item) for item in defaults}
    default_names = set(merged)
    for item in projects:
        merged[item["name"]] = {
            **item,
            "replaces": "default" if item["name"] in default_names else "",
        }
    return list(merged.values())


def merge_subagent_items(defaults: list[dict[str, Any]], projects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge subagent items by name with project entries replacing defaults."""
    merged = {item["name"]: dict(item) for item in defaults}
    default_names = set(merged)
    for item in projects:
        merged[item["name"]] = {
            **item,
            "replaces": "default" if item["name"] in default_names else "",
        }
    return list(merged.values())


def display_item(item: dict[str, Any]) -> dict[str, str]:
    """Drop runtime-only values from a resource display item."""
    return {
        "name": str(item["name"]),
        "path": str(item["path"]),
        "source": str(item["source"]),
        "replaces": str(item.get("replaces") or ""),
    }
