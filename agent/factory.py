"""Agent construction for MIRA's action and planning modes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from deepagents import FilesystemPermission, create_deep_agent
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.subagents import TASK_TOOL_DESCRIPTION
from deepagents.middleware.summarization import create_summarization_tool_middleware
from langchain.agents.middleware.todo import TodoListMiddleware
from langchain.agents.middleware.types import AgentMiddleware
from langchain_quickjs import CodeInterpreterMiddleware

from agent.llm import get_llm
from agent.plan_policy import PLAN_DENIED_FS_OPERATIONS, PLAN_PROJECT_WRITE_TOOLS, plan_system_prompt
from agent.resources import build_resources

PLAN_SYSTEM_PROMPT = plan_system_prompt()


def build_agent(config: dict[str, Any], workspace: Path, checkpointer: Any) -> Any:
    """Build the normal action agent with read/write filesystem access."""
    agent = _build_agent(
        config=config,
        workspace=workspace,
        checkpointer=checkpointer,
        permissions=_action_permissions(),
        interrupt_on=_write_interrupts(),
        excluded_tools=(),
    )
    return agent


def build_plan_agent(config: dict[str, Any], workspace: Path, checkpointer: Any) -> Any:
    """Build the planning agent with project write tools hidden and denied."""
    agent = _build_agent(
        config=config,
        workspace=workspace,
        checkpointer=checkpointer,
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
    model = get_llm(config)
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
        _tool_specs(
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
        "write_file": {"allowed_decisions": ["approve", "edit", "reject"]},
        "edit_file": {"allowed_decisions": ["approve", "edit", "reject"]},
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
        tools = [tool for tool in request.tools if _tool_name(tool) not in self.excluded_tools]
        return request.override(tools=tools)


def _tool_name(tool: Any) -> str | None:
    """Extract a tool name from either a dict tool or object tool."""
    if isinstance(tool, dict):
        name = tool.get("name")
        return name if isinstance(name, str) else None

    name = getattr(tool, "name", None) or getattr(tool, "__name__", None)
    return name if isinstance(name, str) else None


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


def _tool_specs(
    backend: Any,
    middleware: list[Any],
    tools: list[Any],
    tool_metadata: list[dict[str, str]],
    excluded_tools: tuple[str, ...],
) -> list[dict[str, str]]:
    """Collect display metadata from configured tool providers."""
    blocked = set(excluded_tools)
    specs: list[dict[str, str]] = []

    providers = [
        TodoListMiddleware(),
        FilesystemMiddleware(backend=backend),
        *middleware,
    ]

    for provider in providers:
        for tool in getattr(provider, "tools", []):
            _append_tool_spec(specs, tool, blocked)

    if "task" not in blocked:
        specs.append({"name": "task", "description": TASK_TOOL_DESCRIPTION.strip()})

    metadata_by_name = {item["name"]: item for item in tool_metadata}
    for tool in tools:
        name = _tool_name(tool)
        _append_tool_spec(specs, tool, blocked, metadata_by_name.get(name or ""))

    return specs


def _append_tool_spec(
    specs: list[dict[str, str]],
    tool: Any,
    blocked: set[str],
    metadata: dict[str, str] | None = None,
) -> None:
    """Append tool metadata when a supported name is available."""
    name = _tool_name(tool)
    if not name or name in blocked:
        return

    spec = {"name": name, "description": _tool_description(tool)}
    if metadata:
        spec.update(metadata)
        spec["description"] = spec["description"] or _tool_description(tool)

    for index, existing in enumerate(specs):
        if existing["name"] != name:
            continue
        if not spec["description"]:
            spec["description"] = existing.get("description", "")
        specs[index] = spec
        return

    specs.append(spec)


def _tool_description(tool: Any) -> str:
    """Return a concise tool description from metadata or docstring."""
    if isinstance(tool, dict):
        description = tool.get("description")
        return str(description).strip() if description else ""

    description = getattr(tool, "description", None)
    if description:
        return str(description).strip()

    doc = getattr(tool, "__doc__", None)
    return doc.strip().splitlines()[0] if isinstance(doc, str) and doc.strip() else ""
