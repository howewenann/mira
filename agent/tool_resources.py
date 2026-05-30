"""Discovery for MIRA custom tools."""

from __future__ import annotations

import importlib.util
import sys
from hashlib import sha1
from pathlib import Path
from typing import Any

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


def discover_tools(
    default_root: Path,
    project_root: Path,
    project_backend: Any,
) -> tuple[list[Any], list[dict[str, str]]]:
    """Load default and project tools, merging by tool name."""
    defaults = tool_items(default_root, "/mira-defaults/tools", "default", project_backend)
    projects = tool_items(project_root, "/.mira/tools", "project", project_backend)
    merged = merge_tool_items(defaults, projects)
    return [item["tool"] for item in merged], [display_item(item) for item in merged]


def tool_items(root: Path, virtual_root: str, source: str, project_backend: Any) -> list[dict[str, Any]]:
    """Import tool modules from one folder."""
    if not root.exists():
        return []

    items = []
    for path in sorted(root.glob("*.py")):
        for tool in load_tools(path, project_backend):
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


def load_tools(path: Path, project_backend: Any) -> list[Any]:
    """Import one Python file and return its tools."""
    module_id = sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    module_name = f"mira_resource_tools_{module_id}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import tools from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

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


def display_item(item: dict[str, Any]) -> dict[str, str]:
    """Drop runtime-only values from a tool display item."""
    return {
        "name": str(item["name"]),
        "path": str(item["path"]),
        "source": str(item["source"]),
        "replaces": str(item.get("replaces") or ""),
    }
