"""Tool metadata used by the REPL inspection commands."""

from __future__ import annotations

import os
import sys
from pathlib import Path
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
    environment = mira_environment_label()

    builtin_providers = [
        TodoListMiddleware(),
        FilesystemMiddleware(backend=backend),
        *middleware,
    ]

    for provider in builtin_providers:
        for tool in getattr(provider, "tools", []):
            add_tool_spec(specs, tool, blocked, mira_environment=environment)

    if "task" not in blocked:
        add_tool_spec(
            specs,
            {"name": "task", "description": TASK_TOOL_DESCRIPTION.strip()},
            blocked,
            mira_environment=environment,
        )

    metadata_by_name = {item["name"]: item for item in custom_metadata}
    for tool in custom_tools:
        name = tool_name(tool)
        add_tool_spec(
            specs,
            tool,
            blocked,
            metadata_by_name.get(name or ""),
            mira_environment=environment,
        )

    return specs


def add_tool_spec(
    specs: list[dict[str, str]],
    tool: Any,
    blocked: set[str],
    metadata: dict[str, str] | None = None,
    *,
    mira_environment: str | None = None,
) -> None:
    name = tool_name(tool)
    if not name or name in blocked:
        return

    spec = {
        "name": name,
        "description": tool_description(tool),
        "source": "built-in",
        "runtime": "MIRA",
        "environment": mira_environment or mira_environment_label(),
    }
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


def mira_environment_label() -> str:
    """Return a concise label for the Python environment running MIRA."""
    conda_name = str(os.environ.get("CONDA_DEFAULT_ENV") or "").strip()
    if conda_name:
        return conda_name

    conda_prefix = str(os.environ.get("CONDA_PREFIX") or "").strip()
    if conda_prefix:
        return Path(conda_prefix).name or conda_prefix

    virtual_env = str(os.environ.get("VIRTUAL_ENV") or "").strip()
    if virtual_env:
        return Path(virtual_env).name or virtual_env

    if sys.prefix != sys.base_prefix:
        return Path(sys.prefix).name or sys.prefix
    return "System"


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
