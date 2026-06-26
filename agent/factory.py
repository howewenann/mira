"""Agent construction for MIRA's action and planning modes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from deepagents import FilesystemPermission, HarnessProfile, create_deep_agent, register_harness_profile
from langchain_core.messages import AIMessage
from langchain.agents.middleware.types import AgentMiddleware
from langchain_quickjs import CodeInterpreterMiddleware

from agent.compaction import (
    create_mira_summarization_middleware as create_summarization_middleware,
)
from agent.compaction import (
    create_mira_summarization_tool_middleware as create_summarization_tool_middleware,
)
from agent.context_overflow import ProviderContextOverflowMiddleware
from agent.llm import get_llm
from agent.plan_policy import PLAN_DENIED_FS_OPERATIONS, PLAN_PROJECT_WRITE_TOOLS, PRESENT_PLAN_TOOL, plan_system_prompt
from agent.resources import build_resources
from agent.tools.specs import collect_tool_specs, tool_name as resource_tool_name
from config.metadata import ModelMetadata
from config.settings import EXECUTE_TOOL, hitl_settings, tool_always_allow, tool_enabled

SETTINGS_INTERRUPTS = "__mira_settings_interrupts__"
ACTION_EXCLUDED_TOOLS = (PRESENT_PLAN_TOOL,)
PLAN_EXCLUDED_TOOLS = (*PLAN_PROJECT_WRITE_TOOLS, EXECUTE_TOOL)
_REGISTERED_SUMMARIZATION_PROFILE_KEYS: set[str] = set()

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
        extra_middleware=[PlanningToolFilter(ACTION_EXCLUDED_TOOLS)],
        interrupt_on=SETTINGS_INTERRUPTS,
        excluded_tools=ACTION_EXCLUDED_TOOLS,
        enable_execute_backend=tool_enabled(config, EXECUTE_TOOL),
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
        extra_middleware=[PlanningToolFilter(PLAN_EXCLUDED_TOOLS)],
        interrupt_on=None,
        excluded_tools=PLAN_EXCLUDED_TOOLS,
        enable_execute_backend=False,
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
    interrupt_on: dict[str, Any] | str | None = None,
    excluded_tools: tuple[str, ...] = (),
    enable_execute_backend: bool = False,
) -> Any:
    """Create a DeepAgents agent from shared MIRA wiring.

    MIRA delegates filesystem tools, subagent orchestration, and middleware to
    DeepAgents. Keeping that wiring here separates agent construction from REPL
    control flow.
    """
    model = get_llm(config, metadata=metadata)
    resources = build_resources(
        Path(workspace),
        settings=(config or {}).get("settings"),
        enable_execute=enable_execute_backend,
    )
    backend = resources.backend
    permissions = [] if enable_execute_backend else permissions
    excluded_tools = effective_excluded_tools(config, excluded_tools, enable_execute_backend)

    _register_summarization_exclusion(config, model)
    summarization_middleware = create_summarization_middleware(model=model, backend=backend)
    summarization_tool_middleware = create_summarization_tool_middleware(model=model, backend=backend)
    middleware: list[Any] = [
        summarization_middleware,
        FilesystemToolArgNormalizer(Path(workspace)),
        ProviderContextOverflowMiddleware(),
        CodeInterpreterMiddleware(ptc=["task"], skills_backend=backend),
        summarization_tool_middleware,
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
        interrupt_on=_write_interrupts(config, resources.metadata["tools"])
        if interrupt_on == SETTINGS_INTERRUPTS
        else interrupt_on,
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
    _attach_backend(agent, backend)
    _attach_summarization(agent, summarization_middleware)
    return agent


def _register_summarization_exclusion(config: dict[str, Any] | None, model: Any | None = None) -> None:
    """Ask DeepAgents not to auto-add a second summarization middleware."""
    keys = _summarization_profile_keys(config, model)

    for key in keys:
        if key in _REGISTERED_SUMMARIZATION_PROFILE_KEYS:
            continue
        register_harness_profile(
            key,
            HarnessProfile(excluded_middleware=frozenset({"SummarizationMiddleware"})),
        )
        _REGISTERED_SUMMARIZATION_PROFILE_KEYS.add(key)


def _summarization_profile_keys(config: dict[str, Any] | None, model: Any | None = None) -> list[str]:
    """Return DeepAgents harness profile keys that may match this model."""
    candidates: list[str] = []

    provider = str((config or {}).get("llm_provider") or "").strip().lower()
    model_name = str((config or {}).get("llm_model") or "").strip()
    if provider and model_name:
        candidates.append(f"{provider}:{model_name}")
    if provider:
        candidates.append(provider)

    try:
        from deepagents._models import get_model_identifier, get_model_provider

        resolved_provider = str(get_model_provider(model) or "").strip()
        identifier = str(get_model_identifier(model) or "").strip()
    except Exception:
        resolved_provider = ""
        identifier = ""

    if resolved_provider and identifier and ":" not in identifier:
        candidates.append(f"{resolved_provider}:{identifier}")
    if identifier and ":" in identifier:
        candidates.append(identifier)
    if resolved_provider:
        candidates.append(resolved_provider)

    keys: list[str] = []
    for key in candidates:
        if key and key not in keys:
            keys.append(key)
    return keys


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


def _write_interrupts(
    config: dict[str, Any] | None = None,
    tool_metadata: list[dict[str, str]] | None = None,
) -> dict[str, dict[str, list[str]]]:
    """Return human approval policy for action-mode tools."""
    tools = hitl_settings(config).get("tools", {})
    interrupts: dict[str, dict[str, list[str]]] = {}
    if not isinstance(tools, dict):
        return interrupts
    for name, spec in tools.items():
        if not isinstance(name, str) or not isinstance(spec, dict):
            continue
        if not tool_enabled(config, name):
            continue
        if spec.get("always_allow") is True:
            continue
        interrupts[name] = {"allowed_decisions": ["approve", "edit", "reject"]}
    for item in tool_metadata or []:
        name = item.get("name")
        if not name or item.get("source") != "project":
            continue
        if not tool_enabled(config, name) or tool_always_allow(config, name):
            continue
        interrupts[name] = {"allowed_decisions": ["approve", "edit", "reject"]}
    return interrupts


def effective_excluded_tools(
    config: dict[str, Any] | None,
    excluded_tools: tuple[str, ...],
    enable_execute_backend: bool,
) -> tuple[str, ...]:
    """Return tool specs that should be hidden from the UI/model metadata."""
    blocked = set(excluded_tools)
    if not enable_execute_backend or not tool_enabled(config, EXECUTE_TOOL):
        blocked.add(EXECUTE_TOOL)
    return tuple(blocked)


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
        tools = [tool for tool in request.tools if resource_tool_name(tool) not in self.excluded_tools]
        return request.override(tools=tools)


class FilesystemToolArgNormalizer(AgentMiddleware[Any, Any, Any]):
    """Normalize common file-tool arg shapes before HITL and execution."""

    FILE_PATH_TOOLS = {"read_file", "write_file", "edit_file"}

    def __init__(self, workspace: Path) -> None:
        self.workspace = Path(workspace).expanduser().resolve()

    def after_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        messages = state.get("messages") if isinstance(state, dict) else getattr(state, "messages", None)
        if not messages:
            return None

        last_ai_msg = next((msg for msg in reversed(messages) if isinstance(msg, AIMessage)), None)
        if last_ai_msg is None or not last_ai_msg.tool_calls:
            return None

        changed = False
        normalized_calls = []
        for call in last_ai_msg.tool_calls:
            normalized, call_changed = self._normalize_tool_call(call)
            normalized_calls.append(normalized)
            changed = changed or call_changed

        normalized_content, content_changed = self._normalize_content_blocks(last_ai_msg.content)
        changed = changed or content_changed

        if not changed:
            return None

        last_ai_msg.tool_calls = normalized_calls
        if content_changed:
            last_ai_msg.content = normalized_content
        return {"messages": [last_ai_msg]}

    async def aafter_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self.after_model(state, runtime)

    def _normalize_tool_call(self, call: Any) -> tuple[Any, bool]:
        if not isinstance(call, dict):
            return call, False
        if call.get("name") not in self.FILE_PATH_TOOLS:
            return call, False

        args = call.get("args")
        if not isinstance(args, dict):
            return call, False

        normalized_args = dict(args)
        changed = False
        if "file_path" not in normalized_args and "path" in normalized_args:
            normalized_args["file_path"] = normalized_args.pop("path")
            changed = True

        file_path = normalized_args.get("file_path")
        if isinstance(file_path, str):
            normalized_path = self._normalize_workspace_path(file_path)
            if normalized_path != file_path:
                normalized_args["file_path"] = normalized_path
                changed = True

        if not changed:
            return call, False

        return {**call, "args": normalized_args}, True

    def _normalize_content_blocks(self, content: Any) -> tuple[Any, bool]:
        if not isinstance(content, list):
            return content, False

        changed = False
        normalized_blocks = []
        for block in content:
            normalized, block_changed = self._normalize_tool_call(block)
            normalized_blocks.append(normalized)
            changed = changed or block_changed
        return normalized_blocks, changed

    def _normalize_workspace_path(self, value: str) -> str:
        if value.startswith("/"):
            return value
        try:
            path = Path(value).expanduser()
        except (OSError, RuntimeError):
            return value
        if not path.is_absolute():
            return value

        try:
            relative = path.resolve().relative_to(self.workspace)
        except (OSError, RuntimeError, ValueError):
            return value
        return f"/{relative.as_posix()}"


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


def _attach_backend(agent: Any, backend: Any) -> None:
    """Attach the workspace backend for approved filesystem fallback execution."""
    try:
        agent.mira_backend = backend
    except AttributeError:
        return


def _attach_summarization(agent: Any, summarization: Any) -> None:
    """Attach DeepAgents summarization for post-turn compaction."""
    try:
        agent.mira_summarization = summarization
    except AttributeError:
        return
