from pathlib import Path
from typing import Any

from deepagents import FilesystemPermission, create_deep_agent
from deepagents.backends import FilesystemBackend
from deepagents.middleware.summarization import create_summarization_tool_middleware
from langchain.agents.middleware.types import AgentMiddleware
from langchain_quickjs import CodeInterpreterMiddleware

from agent.llm import get_llm
from agent.plan_policy import PLAN_DENIED_FS_OPERATIONS, PLAN_PROJECT_WRITE_TOOLS, plan_system_prompt

SKILLS = []
MEMORY = []
SUBAGENTS = []

PLAN_SYSTEM_PROMPT = plan_system_prompt()


def build_agent(config: dict, workspace: Path, checkpointer):
    return _build_agent(
        config=config,
        workspace=workspace,
        checkpointer=checkpointer,
        permissions=_action_permissions(),
        interrupt_on=_write_interrupts(),
    )


def build_plan_agent(config: dict, workspace: Path, checkpointer):
    return _build_agent(
        config=config,
        workspace=workspace,
        checkpointer=checkpointer,
        permissions=_plan_permissions(),
        system_prompt=PLAN_SYSTEM_PROMPT,
        extra_middleware=[ToolNameFilterMiddleware(PLAN_PROJECT_WRITE_TOOLS)],
        interrupt_on=None,
    )


def _build_agent(
    config: dict,
    workspace: Path,
    checkpointer,
    permissions: list[FilesystemPermission],
    system_prompt: str | None = None,
    extra_middleware: list[AgentMiddleware] | None = None,
    interrupt_on: dict | None = None,
):
    model = get_llm(config)
    backend = FilesystemBackend(root_dir=str(workspace), virtual_mode=True)

    middleware = [
        CodeInterpreterMiddleware(),
        create_summarization_tool_middleware(model=model, backend=backend),
    ]
    middleware.extend(extra_middleware or [])

    return create_deep_agent(
        model=model,
        backend=backend,
        middleware=middleware,
        skills=SKILLS,
        memory=MEMORY,
        subagents=SUBAGENTS,
        permissions=permissions,
        system_prompt=system_prompt,
        interrupt_on=interrupt_on,
        checkpointer=checkpointer,
    )


def _action_permissions() -> list[FilesystemPermission]:
    return [
        FilesystemPermission(
            operations=["read", "write"],
            paths=["/**"],
            mode="allow",
        ),
    ]


def _plan_permissions() -> list[FilesystemPermission]:
    return [
        FilesystemPermission(
            operations=list(PLAN_DENIED_FS_OPERATIONS),
            paths=["/**"],
            mode="deny",
        ),
    ]


def _write_interrupts() -> dict:
    return {
        "write_file": {"allowed_decisions": ["approve", "edit", "reject"]},
        "edit_file": {"allowed_decisions": ["approve", "edit", "reject"]},
    }


class ToolNameFilterMiddleware(AgentMiddleware[Any, Any, Any]):
    def __init__(self, excluded_tools: tuple[str, ...]) -> None:
        self.excluded_tools = set(excluded_tools)

    def wrap_model_call(self, request, handler):
        return handler(self._filter_request(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._filter_request(request))

    def _filter_request(self, request):
        tools = [tool for tool in request.tools if _tool_name(tool) not in self.excluded_tools]
        return request.override(tools=tools)


def _tool_name(tool) -> str | None:
    if isinstance(tool, dict):
        name = tool.get("name")
        return name if isinstance(name, str) else None

    name = getattr(tool, "name", None)
    return name if isinstance(name, str) else None
