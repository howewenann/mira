"""Construction and ordering of MIRA's default middleware pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain.agents.middleware import TodoListMiddleware
from langchain.agents.middleware.types import AgentMiddleware
from langchain_quickjs import CodeInterpreterMiddleware

from agent.compaction import (
    create_mira_summarization_middleware,
    create_mira_summarization_tool_middleware,
)
from agent.context_overflow import ProviderContextOverflowMiddleware
from agent.middleware.execute_tool_prompt import ExecuteToolPromptMiddleware
from agent.middleware.model_response_normalization import ModelResponseNormalizationMiddleware
from config.settings import dynamic_subagents_enabled, planning_todos_enabled

QUICKJS_PTC_TOOLS = ("ls", "read_file", "glob", "grep")
QUICKJS_MEMORY_LIMIT = 64 * 1024 * 1024
QUICKJS_TIMEOUT_SECONDS = 5.0
QUICKJS_PERSISTENCE_MODE = "thread"


@dataclass(frozen=True)
class AgentMiddlewarePipeline:
    """Middleware items plus the summarization instance MIRA observes."""

    items: list[Any]
    summarization: Any


def build_agent_middleware(
    *,
    model: Any,
    backend: Any,
    workspace: Path,
    settings: dict[str, Any] | None = None,
    extra_middleware: list[AgentMiddleware] | None = None,
) -> AgentMiddlewarePipeline:
    """Build MIRA's ordered user middleware pipeline for DeepAgents."""
    summarization_middleware = create_mira_summarization_middleware(model=model, backend=backend)
    summarization_tool_middleware = create_mira_summarization_tool_middleware(model=model, backend=backend)
    middleware: list[Any] = [
        *([TodoListMiddleware()] if planning_todos_enabled(settings) else []),
        summarization_middleware,
        ModelResponseNormalizationMiddleware(Path(workspace)),
        ProviderContextOverflowMiddleware(),
        CodeInterpreterMiddleware(
            memory_limit=QUICKJS_MEMORY_LIMIT,
            timeout=QUICKJS_TIMEOUT_SECONDS,
            ptc=list(QUICKJS_PTC_TOOLS),
            subagents=dynamic_subagents_enabled(settings),
            mode=QUICKJS_PERSISTENCE_MODE,
        ),
        summarization_tool_middleware,
        ExecuteToolPromptMiddleware(),
    ]
    middleware.extend(extra_middleware or [])
    return AgentMiddlewarePipeline(items=middleware, summarization=summarization_middleware)


__all__ = [
    "AgentMiddlewarePipeline",
    "QUICKJS_MEMORY_LIMIT",
    "QUICKJS_PERSISTENCE_MODE",
    "QUICKJS_PTC_TOOLS",
    "QUICKJS_TIMEOUT_SECONDS",
    "build_agent_middleware",
]
