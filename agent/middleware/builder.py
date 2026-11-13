"""Build the ordered middleware bundle used by MIRA agents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain.agents.middleware import TodoListMiddleware
from langchain.agents.middleware.types import AgentMiddleware

from agent.middleware.code_interpreter import (
    InspectableCodeInterpreterMiddleware as CodeInterpreterMiddleware,
)
from agent.middleware.compaction import (
    MiraCompactionMiddleware,
    create_mira_summarization_middleware,
)
from agent.middleware.context_overflow import ProviderContextOverflowMiddleware
from agent.middleware.execute_tool_description_rewrite import (
    ExecuteToolDescriptionRewriteMiddleware,
)
from agent.middleware.context_report import ContextReportMiddleware
from agent.middleware.file_references import FileReferenceMiddleware
from agent.middleware.model_compatibility import ModelCompatibilityMiddleware
from agent.middleware.mode_transition import ModeTransitionMiddleware
from agent.middleware.repeated_tool_call_guard import RepeatedToolCallMiddleware
from config.settings import (
    READ_ONLY_BUILTIN_TOOLS,
    dynamic_subagents_enabled,
    planning_todos_enabled,
    repeated_tool_call_guard_settings,
)

QUICKJS_PTC_TOOLS = READ_ONLY_BUILTIN_TOOLS
QUICKJS_MEMORY_LIMIT = 64 * 1024 * 1024
QUICKJS_TIMEOUT_SECONDS = 5.0
QUICKJS_PERSISTENCE_MODE = "thread"


@dataclass(frozen=True)
class AgentMiddlewareBundle:
    """Built middleware items and MIRA's shared compaction instances."""

    items: list[Any]
    summarization: Any
    compaction: MiraCompactionMiddleware


def build_agent_middleware(
    *,
    model: Any,
    summary_model: Any | None = None,
    backend: Any,
    workspace: Path,
    settings: dict[str, Any] | None = None,
    ptc_tools: list[str] | None = None,
    extra_middleware: list[AgentMiddleware] | None = None,
) -> AgentMiddlewareBundle:
    """Build MIRA's ordered user middleware bundle for DeepAgents."""
    summarization_middleware = create_mira_summarization_middleware(
        model=model, backend=backend, summary_model=summary_model,
    )
    summarization_tool_middleware = MiraCompactionMiddleware(summarization_middleware)
    middleware: list[Any] = [
        *build_repeated_tool_call_middleware(settings),
        *([TodoListMiddleware()] if planning_todos_enabled(settings) else []),
        summarization_middleware,
        FileReferenceMiddleware(),
        ModelCompatibilityMiddleware(Path(workspace)),
        ProviderContextOverflowMiddleware(),
        CodeInterpreterMiddleware(
            memory_limit=QUICKJS_MEMORY_LIMIT,
            timeout=QUICKJS_TIMEOUT_SECONDS,
            ptc=list(QUICKJS_PTC_TOOLS if ptc_tools is None else ptc_tools),
            subagents=dynamic_subagents_enabled(settings),
            mode=QUICKJS_PERSISTENCE_MODE,
        ),
        summarization_tool_middleware,
        ExecuteToolDescriptionRewriteMiddleware(),
    ]
    middleware.extend(extra_middleware or [])
    middleware.append(ModeTransitionMiddleware())
    middleware.append(ContextReportMiddleware(summarization_middleware))
    return AgentMiddlewareBundle(
        items=middleware, summarization=summarization_middleware,
        compaction=summarization_tool_middleware,
    )


def build_repeated_tool_call_middleware(settings: dict[str, Any] | None) -> list[AgentMiddleware]:
    """Create the opt-in guard using the effective persisted thresholds."""
    guard = repeated_tool_call_guard_settings(settings)
    if not guard["enabled"]:
        return []
    return [RepeatedToolCallMiddleware(
        max_repeats=guard["max_repeats"], max_blocked_attempts=guard["max_blocked_attempts"],
    )]


def subagents_with_repeated_tool_call_guard(
    subagents: list[Any], settings: dict[str, Any] | None,
) -> list[Any]:
    """Protect raw owned specs before native or workflow compilation.

    External compiled/remote graphs remain opaque. Forks replace inherited
    middleware by name in DeepAgents, so each receives exactly one guard.
    """
    if not repeated_tool_call_guard_settings(settings)["enabled"]:
        return list(subagents)
    result = []
    for spec in subagents:
        if not isinstance(spec, dict) or "runnable" in spec or "graph_id" in spec:
            result.append(spec)
            continue
        middleware = list(spec.get("middleware") or [])
        # Saved MIRA settings own the guard configuration for raw specs.
        middleware = [item for item in middleware if not isinstance(item, RepeatedToolCallMiddleware)]
        result.append({**spec, "middleware": [*build_repeated_tool_call_middleware(settings), *middleware]})
    return result


__all__ = [
    "AgentMiddlewareBundle",
    "QUICKJS_MEMORY_LIMIT",
    "QUICKJS_PERSISTENCE_MODE",
    "QUICKJS_PTC_TOOLS",
    "QUICKJS_TIMEOUT_SECONDS",
    "build_agent_middleware",
    "build_repeated_tool_call_middleware",
    "subagents_with_repeated_tool_call_guard",
]
