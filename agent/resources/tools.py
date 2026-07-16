"""Discovery for LangChain custom tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool

from agent.resources.items import display_item
from agent.resources.paths import (
    TOOLS_DIR,
    default_dir,
    default_virtual_dir,
    project_dir,
    project_virtual_dir,
)
from agent.resources.python_files import import_python_file
from agent.resources.project_tools import project_tools_from_module
from agent.resources.tool_failures import ToolLoadFailure, tool_load_failure
from mira_tool_api import PROJECT_TOOL_METADATA_ATTRIBUTE

BUILT_IN_TOOL_NAMES = {
    "write_todos",
    "ls",
    "read_file",
    "write_file",
    "edit_file",
    "glob",
    "grep",
    "eval",
    "task",
}


def load_tools(
    workspace: Path,
    project_backend: Any,
    settings: dict[str, Any] | None = None,
) -> tuple[list[Any], list[dict[str, str]], list[ToolLoadFailure]]:
    """Load default tools first, then project tools."""
    defaults, _ = tool_files(
        default_dir(TOOLS_DIR),
        default_virtual_dir(TOOLS_DIR),
        "default",
        project_backend,
        workspace=workspace,
        settings=settings,
    )
    projects, failures = tool_files(
        project_dir(workspace, TOOLS_DIR),
        project_virtual_dir(TOOLS_DIR),
        "project",
        project_backend,
        workspace=workspace,
        settings=settings,
    )
    merged = merge_tool_items(defaults, projects)
    return [item["tool"] for item in merged], [display_item(item) for item in merged], failures


def tool_files(
    root: Path,
    virtual_root: str,
    source: str,
    project_backend: Any,
    *,
    workspace: Path,
    settings: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[ToolLoadFailure]]:
    """Return all tools exported by Python files in one folder."""
    if not root.exists():
        return [], []

    items = []
    failures: list[ToolLoadFailure] = []
    for path in sorted(root.glob("*.py")):
        try:
            tools = tools_from_file(
                path,
                project_backend,
                workspace=workspace,
                settings=settings,
                discover_project_tools=source == "project",
            )
        except BaseException as error:
            if source != "project":
                raise
            failures.append(tool_load_failure(path, workspace, error))
            continue
        for tool in tools:
            item = {
                "name": tool_name(tool),
                "path": f"{virtual_root}/{path.name}",
                "source": source,
                "replaces": "",
                "tool": tool,
            }
            if getattr(tool, "__mira_project_runtime__", False):
                item["runtime"] = "Project"
                item["environment"] = getattr(tool, "__mira_project_environment__", "")
            items.append(item)
    return items, failures


def tools_from_file(
    path: Path,
    project_backend: Any,
    *,
    workspace: Path | None = None,
    settings: dict[str, Any] | None = None,
    discover_project_tools: bool = False,
) -> list[Any]:
    """Return decorated module tools, TOOLS, and get_tools(project_backend)."""
    module = import_python_file(path, "mira_resource_tools")
    tools = module_tools(module)
    if discover_project_tools:
        project_tools = project_tools_from_module(
            module,
            path,
            workspace or path.parent,
            settings,
            project_backend,
        )
        tools.extend(project_tools)
    declared = getattr(module, "TOOLS", [])
    if declared:
        if not isinstance(declared, list | tuple):
            raise TypeError(f"{path} must define TOOLS as a list")
        tools.extend(tool for tool in declared if not is_project_tool_callable(tool))

    get_tools = getattr(module, "get_tools", None)
    if callable(get_tools):
        created = get_tools(project_backend)
        if not isinstance(created, list | tuple):
            raise TypeError(f"{path} get_tools() must return a list")
        tools.extend(tool for tool in created if not is_project_tool_callable(tool))

    return deduplicate_tools(tools)


def module_tools(module: Any) -> list[Any]:
    """Return module-level LangChain tools in definition order."""
    return [
        value
        for value in vars(module).values()
        if isinstance(value, BaseTool)
    ]


def is_project_tool_callable(value: Any) -> bool:
    """Keep marked originals out of direct LangChain registration paths."""
    return callable(value) and isinstance(getattr(value, PROJECT_TOOL_METADATA_ATTRIBUTE, None), dict)


def deduplicate_tools(tools: list[Any]) -> list[Any]:
    """Keep the first tool for each name."""
    unique = {}
    for tool in tools:
        name = tool_name(tool)
        if name not in unique:
            unique[name] = tool
    return list(unique.values())


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
