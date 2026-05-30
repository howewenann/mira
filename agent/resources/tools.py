"""Discovery for LangChain custom tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.resources.items import display_item
from agent.resources.paths import (
    TOOLS_DIR,
    default_dir,
    default_virtual_dir,
    project_dir,
    project_virtual_dir,
)
from agent.resources.python_files import import_python_file

BUILT_IN_TOOL_NAMES = {
    "write_todos",
    "ls",
    "read_file",
    "write_file",
    "edit_file",
    "glob",
    "grep",
    "execute",
    "task",
}


def load_tools(workspace: Path, project_backend: Any) -> tuple[list[Any], list[dict[str, str]]]:
    """Load default tools first, then project tools."""
    defaults = tool_files(default_dir(TOOLS_DIR), default_virtual_dir(TOOLS_DIR), "default", project_backend)
    projects = tool_files(project_dir(workspace, TOOLS_DIR), project_virtual_dir(TOOLS_DIR), "project", project_backend)
    merged = merge_tool_items(defaults, projects)
    return [item["tool"] for item in merged], [display_item(item) for item in merged]


def tool_files(root: Path, virtual_root: str, source: str, project_backend: Any) -> list[dict[str, Any]]:
    """Return all tools exported by Python files in one folder."""
    if not root.exists():
        return []

    items = []
    for path in sorted(root.glob("*.py")):
        for tool in tools_from_file(path, project_backend):
            items.append(
                {
                    "name": tool_name(tool),
                    "path": f"{virtual_root}/{path.name}",
                    "source": source,
                    "replaces": "",
                    "tool": tool,
                }
            )
    return items


def tools_from_file(path: Path, project_backend: Any) -> list[Any]:
    """Return tools from TOOLS and get_tools(project_backend)."""
    module = import_python_file(path, "mira_resource_tools")
    tools = []
    declared = getattr(module, "TOOLS", [])
    if declared:
        if not isinstance(declared, list | tuple):
            raise TypeError(f"{path} must define TOOLS as a list")
        tools.extend(declared)

    get_tools = getattr(module, "get_tools", None)
    if callable(get_tools):
        created = get_tools(project_backend)
        if not isinstance(created, list | tuple):
            raise TypeError(f"{path} get_tools() must return a list")
        tools.extend(created)

    return tools


def tool_name(tool: Any) -> str:
    """Return a LangChain tool name from a supported tool shape."""
    if isinstance(tool, dict):
        name = tool.get("name")
    else:
        name = getattr(tool, "name", None) or getattr(tool, "__name__", None)

    if not name:
        raise ValueError(f"Tool is missing a name: {tool!r}")
    return str(name)


def merge_tool_items(defaults: list[dict[str, Any]], projects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge tool items by name with project entries replacing defaults."""
    merged = {}
    for item in defaults:
        merged[item["name"]] = {
            **item,
            "replaces": "built-in" if item["name"] in BUILT_IN_TOOL_NAMES else "",
        }

    for item in projects:
        current = merged.get(item["name"])
        if current:
            replaces = current["source"]
        elif item["name"] in BUILT_IN_TOOL_NAMES:
            replaces = "built-in"
        else:
            replaces = ""
        merged[item["name"]] = {**item, "replaces": replaces}

    return list(merged.values())
