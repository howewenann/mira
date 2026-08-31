"""Project subagent discovery and DeepAgents compilation."""

from agent.subagents.compilation import compile_dynamic_subagents
from agent.subagents.discovery import (
    DiscoveredSubagent,
    SubagentDiscovery,
    discover_subagents,
    effective_subagent_specs,
    load_subagents,
    resolve_subagent_tool_allowlists,
    subagent_model_issues,
)

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
