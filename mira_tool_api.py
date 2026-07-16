"""Dependency-free declarations for tools that run in a project environment."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar, overload

PROJECT_TOOL_METADATA_ATTRIBUTE = "__mira_project_tool__"
PROJECT_TOOL_METADATA_VERSION = 1

Function = TypeVar("Function", bound=Callable[..., Any])


@overload
def project_tool(function: Function, /) -> Function: ...


@overload
def project_tool(
    function: None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Callable[[Function], Function]: ...


def project_tool(
    function: Function | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Function | Callable[[Function], Function]:
    """Mark a function for execution in MIRA's configured project environment."""
    if name is not None and (not isinstance(name, str) or not name.strip()):
        raise ValueError("project_tool name must be a non-empty string")
    if description is not None and not isinstance(description, str):
        raise TypeError("project_tool description must be a string")

    def decorate(target: Function) -> Function:
        if not callable(target):
            raise TypeError("project_tool must decorate a callable")
        setattr(
            target,
            PROJECT_TOOL_METADATA_ATTRIBUTE,
            {
                "version": PROJECT_TOOL_METADATA_VERSION,
                "name": name.strip() if name is not None else None,
                "description": description.strip() if description is not None else None,
            },
        )
        return target

    if function is None:
        return decorate
    if not callable(function):
        raise TypeError("project_tool must be used as @project_tool or @project_tool(...)")
    return decorate(function)


__all__ = ["project_tool"]
