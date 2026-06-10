"""Agent construction for MIRA's action and planning modes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from deepagents import FilesystemPermission, create_deep_agent
from deepagents.middleware.summarization import create_summarization_tool_middleware
from langchain.agents.middleware.types import AgentMiddleware
from langchain_quickjs import CodeInterpreterMiddleware

from agent.llm import get_llm
from agent.plan_policy import PLAN_DENIED_FS_OPERATIONS, PLAN_PROJECT_WRITE_TOOLS, plan_system_prompt
from agent.resources import build_resources
from agent.tools.specs import collect_tool_specs, tool_name
from config.metadata import ModelMetadata

PLAN_SYSTEM_PROMPT = plan_system_prompt()


def build_agent(
    config: dict[str, Any],
    workspace: Path,
    checkpointer: Any,
    metadata: ModelMetadata | None = None,
) -> Any:
    """Build the normal action agent with read/write filesystem access."""
    agent = _build_agent(
        config=config,
        workspace=workspace,
        checkpointer=checkpointer,
        metadata=metadata,
        permissions=_action_permissions(),
        interrupt_on=_write_interrupts(),
        excluded_tools=(),
    )
    return agent


def build_plan_agent(
    config: dict[str, Any],
    workspace: Path,
    checkpointer: Any,
    metadata: ModelMetadata | None = None,
) -> Any:
    """Build the planning agent with project write tools hidden and denied."""
    agent = _build_agent(
        config=config,
        workspace=workspace,
        checkpointer=checkpointer,
        metadata=metadata,
        permissions=_plan_permissions(),
        system_prompt=PLAN_SYSTEM_PROMPT,
        extra_middleware=[PlanningToolFilter(PLAN_PROJECT_WRITE_TOOLS)],
        interrupt_on=None,
        excluded_tools=PLAN_PROJECT_WRITE_TOOLS,
    )
    return agent


def _build_agent(
    config: dict[str, Any],
    workspace: Path,
    checkpointer: Any,
    metadata: ModelMetadata | None,
    permissions: list[FilesystemPermission],
    system_prompt: str | None = None,
    extra_middleware: list[AgentMiddleware] | None = None,
    interrupt_on: dict[str, Any] | None = None,
    excluded_tools: tuple[str, ...] = (),
) -> Any:
    """Create a DeepAgents agent from shared MIRA wiring.

    MIRA delegates filesystem tools, subagent orchestration, and middleware to
    DeepAgents. Keeping that wiring here separates agent construction from REPL
    control flow.
    """
    model = get_llm(config, metadata=metadata)
    resources = build_resources(Path(workspace))
    backend = resources.backend

    middleware: list[Any] = [
        CodeInterpreterMiddleware(),
        create_summarization_tool_middleware(model=model, backend=backend),
    ]
    middleware.extend(extra_middleware or [])

    agent = create_deep_agent(
        model=model,
        backend=backend,
        middleware=middleware,
        tools=resources.tools,
        skills=resources.skills,
        memory=resources.memory,
        subagents=resources.subagents,
        permissions=permissions,
        system_prompt=system_prompt,
        interrupt_on=interrupt_on,
        checkpointer=checkpointer,
    )
    _attach_tool_specs(
        agent,
        collect_tool_specs(
            backend,
            middleware,
            resources.tools,
            resources.metadata["tools"],
            excluded_tools,
        ),
    )
    _attach_resources(agent, resources.metadata)
    return agent


def _action_permissions() -> list[FilesystemPermission]:
    """Allow the action agent to read and write inside the workspace backend."""
    return [
        FilesystemPermission(
            operations=["write"],
            paths=["/mira-defaults/**"],
            mode="deny",
        ),
        FilesystemPermission(
            operations=["read", "write"],
            paths=["/**"],
            mode="allow",
        ),
    ]


def _plan_permissions() -> list[FilesystemPermission]:
    """Deny writes as a backstop while planning mode is active."""
    return [
        FilesystemPermission(
            operations=list(PLAN_DENIED_FS_OPERATIONS),
            paths=["/**"],
            mode="deny",
        ),
    ]


def _write_interrupts() -> dict[str, dict[str, list[str]]]:
    """Require human approval before the action agent writes project files."""
    return {
        "write_file": {"allowed_decisions": ["approve", "edit", "reject", "respond"]},
        "edit_file": {"allowed_decisions": ["approve", "edit", "reject", "respond"]},
    }


class PlanningToolFilter(AgentMiddleware[Any, Any, Any]):
    """Middleware that removes project write tools from planning model calls."""

    def __init__(self, excluded_tools: tuple[str, ...]) -> None:
        """Store tool names that should be hidden from the model."""
        self.excluded_tools = set(excluded_tools)

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        """Filter tools for synchronous LangChain model calls."""
        return handler(self._filter_request(request))

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        """Filter tools for asynchronous LangChain model calls."""
        return await handler(self._filter_request(request))

    def _filter_request(self, request: Any) -> Any:
        """Return a request copy with excluded tools removed."""
        tools = [tool for tool in request.tools if tool_name(tool) not in self.excluded_tools]
        return request.override(tools=tools)


_tool_name = tool_name


def _attach_tool_specs(agent: Any, specs: list[dict[str, str]]) -> None:
    """Attach tool display metadata used by the REPL."""
    try:
        agent.mira_tool_specs = specs
    except AttributeError:
        return


def _attach_resources(agent: Any, resources: dict[str, list[dict[str, str]]]) -> None:
    """Attach resource display metadata used by the REPL."""
    try:
        agent.mira_resources = resources
    except AttributeError:
        return
