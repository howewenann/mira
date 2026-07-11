"""Compile DeepAgents subagents when dynamic response schemas are disabled."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import Any

from deepagents._models import resolve_model
from deepagents.middleware._fs_interrupt import _build_interrupt_on_from_permissions
from deepagents.middleware.filesystem import FilesystemMiddleware, FilesystemPermission
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from deepagents.middleware.skills import SkillsMiddleware
from deepagents.middleware.subagents import GENERAL_PURPOSE_SUBAGENT, create_sub_agent
from deepagents.middleware.summarization import create_summarization_middleware
from langchain.agents.middleware import TodoListMiddleware


def compile_dynamic_subagents(
    subagents: Sequence[dict[str, Any]],
    *,
    model: Any,
    tools: Sequence[Any],
    backend: Any,
    skills: list[str] | None,
    permissions: list[FilesystemPermission] | None,
    interrupt_on: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Return synchronous subagents as compiled runnables.

    DeepAgents rejects per-call response schemas for compiled subagents. Raw
    specs are materialized here with the same inherited capabilities that
    ``create_deep_agent`` normally supplies before compiling them.
    """
    specs = list(subagents)
    if not any(_is_synchronous(spec) and spec.get("name") == "general-purpose" for spec in specs):
        general_purpose = deepcopy(GENERAL_PURPOSE_SUBAGENT)
        if skills is not None:
            general_purpose["skills"] = list(skills)
        specs.insert(0, general_purpose)

    return [
        _compile_raw_subagent(
            spec,
            model=model,
            tools=tools,
            backend=backend,
            permissions=permissions,
            interrupt_on=interrupt_on,
        )
        if _is_raw_synchronous(spec)
        else spec
        for spec in specs
    ]


def _is_synchronous(spec: dict[str, Any]) -> bool:
    return "graph_id" not in spec


def _is_raw_synchronous(spec: dict[str, Any]) -> bool:
    return _is_synchronous(spec) and "runnable" not in spec


def _compile_raw_subagent(
    spec: dict[str, Any],
    *,
    model: Any,
    tools: Sequence[Any],
    backend: Any,
    permissions: list[FilesystemPermission] | None,
    interrupt_on: dict[str, Any] | None,
) -> dict[str, Any]:
    subagent_model = resolve_model(spec.get("model", model))
    subagent_tools = spec.get("tools") if "tools" in spec else tools
    subagent_permissions = spec.get("permissions", permissions)
    middleware: list[Any] = [
        TodoListMiddleware(),
        FilesystemMiddleware(backend=backend, _permissions=subagent_permissions),
        create_summarization_middleware(subagent_model, backend),
        PatchToolCallsMiddleware(),
    ]
    subagent_skills = spec.get("skills")
    if subagent_skills:
        middleware.append(SkillsMiddleware(backend=backend, sources=subagent_skills))
    middleware.extend(spec.get("middleware", []))

    selected_interrupts = spec.get("interrupt_on", interrupt_on)
    filesystem_interrupts = _build_interrupt_on_from_permissions(subagent_permissions or [])
    if filesystem_interrupts or selected_interrupts:
        selected_interrupts = {**filesystem_interrupts, **(selected_interrupts or {})}

    materialized = {
        **spec,
        "model": subagent_model,
        "tools": list(subagent_tools or []),
        "middleware": middleware,
    }
    if selected_interrupts:
        materialized["interrupt_on"] = selected_interrupts

    return {
        "name": materialized["name"],
        "description": materialized["description"],
        "runnable": create_sub_agent(materialized),
    }
