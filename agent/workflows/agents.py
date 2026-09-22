"""Workflow-local specialization of configured MIRA subagents."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.tools import BaseTool

from agent.subagents.compilation import compile_raw_subagent
from agent.subagents.discovery import SubagentDiscovery, resolve_subagent_tool_allowlists


@dataclass(frozen=True, slots=True)
class WorkflowAgentRegistry:
    """Effective subagent definitions and their existing compilation inputs."""

    specs: tuple[Any, ...]
    model: Any
    tools: tuple[Any, ...]
    builtin_tool_names: frozenset[str]
    discovery: SubagentDiscovery
    excluded_tool_names: frozenset[str]
    backend: Any
    skills: tuple[str, ...]
    permissions: tuple[Any, ...]
    interrupt_on: dict[str, Any] | None
    enable_todos: bool

    def specialize(
        self,
        base: str,
        *,
        name: str | None,
        tools: Sequence[str | BaseTool] | None,
        system_prompt: Any,
        response_format: Any,
        inherit: Any,
    ) -> Any:
        """Copy, override, resolve, and compile one effective definition."""
        spec = next(
            (item for item in self.specs if _name(item) == base),
            None,
        )
        if spec is None:
            available = ", ".join(
                sorted(_name(item) for item in self.specs if _name(item))
            )
            raise KeyError(
                f"Unknown MIRA subagent {base!r}. Available subagents: {available or 'none'}."
            )
        requested_override = (
            (name is not None and name != base)
            or tools is not None
            or system_prompt is not inherit
            or response_format is not inherit
        )
        if isinstance(spec, dict) and "graph_id" in spec:
            raise TypeError(
                f"MIRA subagent {base!r} is a remote/opaque graph and cannot be used "
                "as a workflow-local runnable."
            )
        if isinstance(spec, dict) and "runnable" in spec:
            if requested_override:
                raise TypeError(
                    f"MIRA subagent {base!r} is already compiled; workflow-local "
                    "overrides cannot be applied safely."
                )
            return spec["runnable"]
        if not isinstance(spec, dict):
            raise TypeError(
                f"MIRA subagent {base!r} has an unsupported opaque definition."
            )
        if spec.get("mode") == "fork":
            raise TypeError(
                f"MIRA subagent {base!r} uses parent-conversation fork mode and cannot "
                "be inserted as an independent workflow node."
            )

        specialized = dict(spec)
        specialized["middleware"] = list(spec.get("middleware") or [])
        if "skills" not in specialized and self.skills:
            specialized["skills"] = list(self.skills)
        if name is not None:
            specialized["name"] = name
        if system_prompt is not inherit:
            specialized["system_prompt"] = system_prompt
        if response_format is not inherit:
            if response_format is None:
                specialized.pop("response_format", None)
            else:
                specialized["response_format"] = response_format
        if tools is not None:
            specialized["tools"] = list(tools)
            resolved, issues = resolve_subagent_tool_allowlists(
                [specialized],
                self.tools,
                self.builtin_tool_names,
                self.discovery,
                self.excluded_tool_names,
            )
            if issues:
                issue = issues[0]
                raise ValueError(f"{issue.summary}: {issue.details}")
            if not resolved:
                raise ValueError(
                    f"MIRA could not resolve the workflow-local tools for {base!r}."
                )
            specialized = resolved[0]

        compiled = compile_raw_subagent(
            specialized,
            model=self.model,
            tools=self.tools,
            backend=self.backend,
            permissions=list(self.permissions),
            interrupt_on=self.interrupt_on,
            enable_todos=self.enable_todos,
        )
        return compiled["runnable"]


def _name(spec: Any) -> str:
    if isinstance(spec, dict):
        return str(spec.get("name") or "")
    return str(getattr(spec, "name", "") or "")


__all__ = ["WorkflowAgentRegistry"]
