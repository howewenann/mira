"""Project subagent discovery and DeepAgents compilation."""

from __future__ import annotations

from typing import Any

from agent.subagents.compilation import compile_dynamic_subagents

__all__ = [
    "DiscoveredSubagent",
    "SubagentDiscovery",
    "compile_dynamic_subagents",
    "discover_subagents",
    "effective_subagent_specs",
    "load_subagents",
    "resolve_subagent_tool_allowlists",
    "subagent_model_issues",
]


def __getattr__(name: str) -> Any:
    """Load discovery exports lazily to avoid the resources/discovery cycle."""
    if name not in __all__ or name == "compile_dynamic_subagents":
        raise AttributeError(name)
    from agent.subagents import discovery

    return getattr(discovery, name)
