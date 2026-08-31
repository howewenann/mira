"""Build the ordered middleware bundle used by MIRA agents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain.agents.middleware import TodoListMiddleware
from langchain.agents.middleware.types import AgentMiddleware
from langchain_quickjs import CodeInterpreterMiddleware

from agent.middleware.compaction import (
    create_mira_summarization_middleware,
    create_mira_summarization_tool_middleware,
)
from agent.middleware.context_overflow import ProviderContextOverflowMiddleware
from agent.middleware.execute_tool_description_rewrite import (
    ExecuteToolDescriptionRewriteMiddleware,
)
from agent.middleware.context_report import ContextReportMiddleware
from agent.middleware.file_references import FileReferenceMiddleware
from agent.middleware.model_response_normalization import ModelResponseNormalizationMiddleware
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
    """Built middleware items plus the summarization instance MIRA observes."""

    items: list[Any]
    summarization: Any


def build_agent_middleware(
    *,
    model: Any,
    backend: Any,
    workspace: Path,
    settings: dict[str, Any] | None = None,
    ptc_tools: list[str] | None = None,
    extra_middleware: list[AgentMiddleware] | None = None,
) -> AgentMiddlewareBundle:
    """Build MIRA's ordered user middleware bundle for DeepAgents."""
    summarization_middleware = create_mira_summarization_middleware(model=model, backend=backend)
    summarization_tool_middleware = create_mira_summarization_tool_middleware(model=model, backend=backend)
    middleware: list[Any] = [
        *([TodoListMiddleware()] if planning_todos_enabled(settings) else []),
        summarization_middleware,
        FileReferenceMiddleware(),
        ModelResponseNormalizationMiddleware(Path(workspace)),
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
    middleware.append(ContextReportMiddleware())
    return AgentMiddlewareBundle(items=middleware, summarization=summarization_middleware)


__all__ = [
    "AgentMiddlewareBundle",
    "QUICKJS_MEMORY_LIMIT",
    "QUICKJS_PERSISTENCE_MODE",
    "QUICKJS_PTC_TOOLS",
    "QUICKJS_TIMEOUT_SECONDS",
    "build_agent_middleware",
]
