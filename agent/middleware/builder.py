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
from config.settings import (
    READ_ONLY_BUILTIN_TOOLS,
    dynamic_subagents_enabled,
    planning_todos_enabled,
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
    middleware.append(ContextReportMiddleware(summarization_middleware))
    return AgentMiddlewareBundle(
        items=middleware, summarization=summarization_middleware,
        compaction=summarization_tool_middleware,
    )


__all__ = [
    "AgentMiddlewareBundle",
    "QUICKJS_MEMORY_LIMIT",
    "QUICKJS_PERSISTENCE_MODE",
    "QUICKJS_PTC_TOOLS",
    "QUICKJS_TIMEOUT_SECONDS",
    "build_agent_middleware",
]
