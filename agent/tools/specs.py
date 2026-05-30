"""Tool metadata used by the REPL inspection commands."""

from __future__ import annotations

from typing import Any

from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.subagents import TASK_TOOL_DESCRIPTION
from langchain.agents.middleware.todo import TodoListMiddleware


def collect_tool_specs(
    backend: Any,
    middleware: list[Any],
    custom_tools: list[Any],
    custom_metadata: list[dict[str, str]],
    excluded_tools: tuple[str, ...],
) -> list[dict[str, str]]:
    blocked = set(excluded_tools)
    specs: list[dict[str, str]] = []

    builtin_providers = [
        TodoListMiddleware(),
        FilesystemMiddleware(backend=backend),
        *middleware,
    ]

    for provider in builtin_providers:
        for tool in getattr(provider, "tools", []):
            add_tool_spec(specs, tool, blocked)

    if "task" not in blocked:
        specs.append({"name": "task", "description": TASK_TOOL_DESCRIPTION.strip()})

    metadata_by_name = {item["name"]: item for item in custom_metadata}
    for tool in custom_tools:
        name = tool_name(tool)
        add_tool_spec(specs, tool, blocked, metadata_by_name.get(name or ""))

    return specs


def add_tool_spec(
    specs: list[dict[str, str]],
    tool: Any,
    blocked: set[str],
    metadata: dict[str, str] | None = None,
) -> None:
    name = tool_name(tool)
    if not name or name in blocked:
        return

    spec = {"name": name, "description": tool_description(tool)}
    if metadata:
        spec.update(metadata)
        spec["description"] = spec["description"] or tool_description(tool)

    for index, existing in enumerate(specs):
        if existing["name"] == name:
            if not spec["description"]:
                spec["description"] = existing.get("description", "")
            specs[index] = spec
            return

    specs.append(spec)


def tool_name(tool: Any) -> str | None:
    if isinstance(tool, dict):
        name = tool.get("name")
        return name if isinstance(name, str) else None

    name = getattr(tool, "name", None) or getattr(tool, "__name__", None)
    return name if isinstance(name, str) else None


def tool_description(tool: Any) -> str:
    if isinstance(tool, dict):
        description = tool.get("description")
        return str(description).strip() if description else ""

    description = getattr(tool, "description", None)
    if description:
        return str(description).strip()

    doc = getattr(tool, "__doc__", None)
    return doc.strip().splitlines()[0] if isinstance(doc, str) and doc.strip() else ""
