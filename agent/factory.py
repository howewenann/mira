"""Agent construction for MIRA's action and planning modes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from deepagents import FilesystemPermission, HarnessProfile, create_deep_agent, register_harness_profile
from deepagents.middleware import FilesystemMiddleware
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.agents.middleware.types import AgentMiddleware

from agent.llm import get_llm, get_rubric_model_name
from agent.middleware import (
    CorrectionMiddleware,
    ModelToolVisibilityMiddleware,
    PlanningStageEnforcementMiddleware,
    ProjectToolErrorMiddleware,
    QUICKJS_PTC_TOOLS,
    build_agent_middleware,
)
from agent.middleware.rubric import MiraRubricMiddleware as RubricMiddleware
from agent.planning.response_status import PlanningResponseStatusRule
from agent.planning.tool_context import PlanningToolContext
from agent.planning.criteria import SuccessCriteriaService
from agent.planning.policy import (
    FINALIZE_GOAL_TOOL,
    FINALIZE_PLAN_TOOL,
    PLAN_DENIED_FS_OPERATIONS,
    PLAN_DISABLED_TOOLS,
    PREPARE_GOAL_TOOL,
    PREPARE_PLAN_TOOL,
    SHARED_QUESTION_POLICY,
    plan_system_prompt,
)
from agent.resources import build_resources
from agent.subagent_compilation import compile_dynamic_subagents
from agent.tools.specs import backend_supports_delete, collect_tool_specs, tool_name
from config.metadata import ModelMetadata
from config.settings import (
    EXECUTE_TOOL,
    INBUILT_DANGEROUS_TOOLS,
    PTC_INAPPLICABLE_TOOLS,
    dynamic_subagent_response_schema_enabled,
    dynamic_subagents_enabled,
    hitl_settings,
    planning_todos_enabled,
    planning_response_status_max_retries,
    rubric_enabled,
    rubric_max_iterations,
    tool_always_allow,
    tool_enabled,
    tool_policy,
    tool_plan_access,
    tool_ptc,
    tool_rubric_access,
    mcp_tool_policy,
    mcp_server_enabled,
    RUBRIC_MODEL,
    SUMMARIZATION_MODEL,
    model_assignment,
    middleware_span_mode,
)
from tracing.middleware_spans import middleware_span_policy

SETTINGS_INTERRUPTS = "__mira_settings_interrupts__"
ACTION_EXCLUDED_TOOLS = (
    PREPARE_PLAN_TOOL,
    PREPARE_GOAL_TOOL,
    FINALIZE_PLAN_TOOL,
    FINALIZE_GOAL_TOOL,
)
PLAN_EXCLUDED_TOOLS = PLAN_DISABLED_TOOLS
_REGISTERED_SUMMARIZATION_PROFILE_KEYS: set[str] = set()

PLAN_SYSTEM_PROMPT = plan_system_prompt()
ACT_SYSTEM_PROMPT = """You are MIRA, a general-purpose agent.

Follow the user's request and the instructions available in your context.
Use the available tools when they help achieve the requested outcome.

Inspect relevant context before consequential actions.

{question_policy}

Respect tool permissions and approval requirements.
Be accurate about what you completed, what failed, and what remains unresolved.
""".format(question_policy=SHARED_QUESTION_POLICY)


def build_agent(
    config: dict[str, Any],
    workspace: Path,
    checkpointer: Any,
    metadata: ModelMetadata | None = None,
    mcp_manager: Any | None = None,
    resources: Any | None = None,
) -> Any:
    """Build the normal action agent with read/write filesystem access."""
    agent = _build_agent(
        config=config,
        workspace=workspace,
        checkpointer=checkpointer,
        metadata=metadata,
        permissions=_action_permissions(),
        system_prompt=ACT_SYSTEM_PROMPT,
        interrupt_on=SETTINGS_INTERRUPTS,
        excluded_tools=ACTION_EXCLUDED_TOOLS,
        enable_execute_backend=tool_enabled(config, EXECUTE_TOOL),
        enable_rubric=rubric_enabled(config),
        mcp_manager=mcp_manager,
        planning=False,
        resources=resources,
    )
    return agent


def build_plan_agent(
    config: dict[str, Any],
    workspace: Path,
    checkpointer: Any,
    metadata: ModelMetadata | None = None,
    mcp_manager: Any | None = None,
    resources: Any | None = None,
) -> Any:
    """Build the planning agent with project write tools hidden and denied."""
    agent = _build_agent(
        config=config,
        workspace=workspace,
        checkpointer=checkpointer,
        metadata=metadata,
        permissions=_plan_permissions(),
        system_prompt=PLAN_SYSTEM_PROMPT,
        interrupt_on=SETTINGS_INTERRUPTS,
        excluded_tools=PLAN_EXCLUDED_TOOLS,
        enable_execute_backend=False,
        extra_middleware=[
            PlanningStageEnforcementMiddleware(),
            CorrectionMiddleware(
                rules=(
                    PlanningResponseStatusRule(workflow="plan"),
                    PlanningResponseStatusRule(workflow="goal"),
                ),
                max_retries=planning_response_status_max_retries(config),
            ),
        ],
        omitted_tools=(),
        mcp_manager=mcp_manager,
        planning=True,
        resources=resources,
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
    enable_rubric: bool = False,
    omitted_tools: tuple[str, ...] = (PREPARE_GOAL_TOOL, PREPARE_PLAN_TOOL),
    mcp_manager: Any | None = None,
    planning: bool = False,
    resources: Any | None = None,
) -> Any:
    """Create a DeepAgents agent from shared MIRA wiring.

    MIRA delegates filesystem tools, subagent orchestration, and middleware to
    DeepAgents. Keeping that wiring here separates agent construction from REPL
    control flow.
    """
    model = get_llm(config, metadata=metadata)
    resources = resources or build_resources(
        Path(workspace),
        settings=(config or {}).get("settings"),
        enable_execute=enable_execute_backend,
        config=config,
    )
    backend = resources.backend
    local_metadata = resources.metadata["tools"]
    active_local_names = {tool_name(tool) for tool in resources.tools}
    project_tool_names = frozenset(
        str(item["name"])
        for item in local_metadata
        if item.get("source") == "project" and item.get("name") in active_local_names
    )
    permissions = [] if enable_execute_backend else permissions
    excluded_tools = effective_excluded_tools(config, excluded_tools, enable_execute_backend)
    if not backend_supports_delete(backend):
        excluded_tools = (*excluded_tools, "delete")
    rubric_model = model
    if enable_rubric and model_assignment(config, RUBRIC_MODEL):
        rubric_model = get_llm(config, role=RUBRIC_MODEL)
    summarization_model = (
        get_llm(config, role=SUMMARIZATION_MODEL)
        if model_assignment(config, SUMMARIZATION_MODEL)
        else model
    )
    _register_summarization_exclusion(config, summarization_model)
    mcp_tools: list[Any] = []
    mcp_metadata: list[dict[str, str]] = []
    if mcp_manager is not None:
        mcp_tools, mcp_metadata = mcp_manager.tools_for_mode(
            (config or {}).get("settings"),
            planning=planning,
        )
    resolved_interrupt_on = (
        _write_interrupts(config, [*local_metadata, *mcp_metadata], planning=planning)
        if interrupt_on == SETTINGS_INTERRUPTS
        else interrupt_on
    )
    if isinstance(resolved_interrupt_on, dict):
        resolved_interrupt_on = {
            name: rule for name, rule in resolved_interrupt_on.items() if name not in excluded_tools
        }
    tools = [
        tool
        for tool in resources.tools
        if tool_name(tool) not in omitted_tools
        and (not planning or _local_tool_available_in_plan(config, local_metadata, tool_name(tool)))
    ]
    tools.extend(mcp_tools)
    rubric_middleware: list[AgentMiddleware] = []
    if enable_rubric:
        rubric_tools, rubric_interrupts = effective_rubric_tools(
            config,
            backend,
            tools,
            [*local_metadata, *mcp_metadata],
            excluded_tools,
        )
        rubric_middleware.append(
            RubricMiddleware(
                model=rubric_model,
                verifier_tools=rubric_tools,
                verifier_middleware=(
                    [HumanInTheLoopMiddleware(interrupt_on=rubric_interrupts)] if rubric_interrupts else []
                ),
                max_iterations=rubric_max_iterations(config),
            )
        )
    extra_middleware = [
        *(extra_middleware or []),
        *rubric_middleware,
        *([ProjectToolErrorMiddleware(project_tool_names)] if project_tool_names else []),
        ModelToolVisibilityMiddleware(excluded_tools),
    ]
    subagents = subagents_with_project_tool_errors(resources.subagents, project_tool_names)
    settings = (config or {}).get("settings")
    with middleware_span_policy(middleware_span_mode(settings)):
        if dynamic_subagents_enabled(settings) and not dynamic_subagent_response_schema_enabled(settings):
            subagents = compile_dynamic_subagents(
                subagents,
                model=model,
                tools=tools,
                backend=backend,
                skills=resources.skills,
                permissions=permissions,
                interrupt_on=resolved_interrupt_on,
                enable_todos=planning_todos_enabled(settings),
            )

        middleware_stack = build_agent_middleware(
            model=summarization_model,
            backend=backend,
            workspace=Path(workspace),
            settings=settings,
            ptc_tools=effective_ptc_tool_names(
                config,
                tools,
                [*local_metadata, *mcp_metadata],
                excluded_tools,
            ),
            extra_middleware=extra_middleware,
        )

        agent = create_deep_agent(
            model=model,
            backend=backend,
            middleware=middleware_stack.items,
            tools=tools,
            skills=resources.skills,
            memory=resources.memory,
            subagents=subagents,
            permissions=permissions,
            system_prompt=system_prompt,
            interrupt_on=resolved_interrupt_on,
            checkpointer=checkpointer,
            context_schema=PlanningToolContext if planning else None,
        )
    _attach_tool_specs(
        agent,
        collect_tool_specs(
            backend,
            middleware_stack.items,
            tools,
            [*local_metadata, *mcp_metadata],
            excluded_tools,
        ),
    )
    combined_metadata = {**resources.metadata, "tools": [*local_metadata, *mcp_metadata]}
    _attach_resources(agent, combined_metadata)
    _attach_context_report_config(
        agent,
        system_prompt=system_prompt or "",
        memory_sources=resources.memory,
        skill_sources=resources.skills,
    )
    _attach_tool_failures(agent, resources.tool_failures)
    _attach_resource_issues(agent, resources.issues, resources.subagent_discovery)
    _attach_backend(agent, backend, resources.project_backend)
    _attach_summarization(agent, middleware_stack.summarization)
    if enable_rubric:
        _attach_rubric_model_name(agent, get_rubric_model_name(config))
    if planning:
        _attach_planning_context(
            agent,
            PlanningToolContext(SuccessCriteriaService(config, metadata=metadata)),
        )
    return agent


def subagents_with_project_tool_errors(
    subagents: list[Any],
    project_tool_names: frozenset[str],
) -> list[Any]:
    """Add project-tool recovery to raw subagents without mutating user specs."""
    if not project_tool_names:
        return list(subagents)

    prepared: list[Any] = []
    for spec in subagents:
        if not isinstance(spec, dict) or "graph_id" in spec or "runnable" in spec:
            prepared.append(spec)
            continue
        copied = dict(spec)
        middleware = list(spec.get("middleware") or [])
        if not any(isinstance(item, ProjectToolErrorMiddleware) for item in middleware):
            middleware.append(ProjectToolErrorMiddleware(project_tool_names))
        copied["middleware"] = middleware
        prepared.append(copied)
    return prepared


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

    resolved_provider = _model_provider(model)
    identifier = _model_identifier(model)

    if resolved_provider and identifier and ":" not in identifier:
        candidates.append(f"{resolved_provider}:{identifier}")
    if identifier and ":" in identifier:
        candidates.append(identifier)
    if resolved_provider:
        candidates.append(resolved_provider)

    keys: list[str] = []
    for key in candidates:
        if _valid_summarization_profile_key(key) and key not in keys:
            keys.append(key)
    return keys


def _model_identifier(model: Any) -> str:
    """Return a public model identifier without importing DeepAgents internals."""
    for attribute in ("model_name", "model", "model_id"):
        value = getattr(model, attribute, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _model_provider(model: Any) -> str:
    """Return LangChain's tracing provider label when a model exposes one."""
    get_params = getattr(model, "_get_ls_params", None)
    if not callable(get_params):
        return ""
    try:
        params = get_params()
    except (AttributeError, TypeError, NotImplementedError):
        return ""
    if not isinstance(params, dict):
        return ""
    value = params.get("ls_provider")
    return value.strip() if isinstance(value, str) else ""


def _valid_summarization_profile_key(key: str) -> bool:
    """Return whether a generated key fits DeepAgents' registry shape."""
    if not key or key != key.strip() or key.count(":") > 1:
        return False
    if ":" not in key:
        return True
    provider, model = key.split(":", 1)
    return bool(provider and model and provider == provider.strip() and model == model.strip())


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
    *,
    planning: bool = False,
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
        if planning and not tool_plan_access(config, name):
            continue
        if spec.get("always_allow") is True:
            continue
        interrupts[name] = {"allowed_decisions": ["approve", "edit", "reject"]}
    for item in tool_metadata or []:
        name = item.get("name")
        if not name or item.get("source") not in {"project", "mcp"}:
            continue
        if item.get("source") == "mcp":
            policy = mcp_tool_policy(config, item.get("server", ""), item.get("original_name", ""))
            if not policy.enabled or (planning and not policy.plan_access) or policy.always_allow:
                continue
        elif (
            not tool_enabled(config, name)
            or (planning and not tool_plan_access(config, name))
            or tool_always_allow(config, name)
        ):
            continue
        interrupts[name] = {"allowed_decisions": ["approve", "edit", "reject"]}
    return interrupts


def _local_tool_available_in_plan(
    config: dict[str, Any] | None,
    metadata: list[dict[str, str]],
    name: str,
) -> bool:
    info = next((item for item in metadata if item.get("name") == name), None)
    if info is None or info.get("source") != "project":
        return True
    return tool_enabled(config, name) and tool_plan_access(config, name)


def effective_ptc_tool_names(
    config: dict[str, Any] | None,
    tools: list[Any],
    metadata: list[dict[str, str]],
    excluded_tools: tuple[str, ...] = (),
) -> list[str]:
    """Resolve PTC names from tools available to this agent construction.

    QuickJS applies its own final filter against each live model request, so
    later middleware visibility changes remain authoritative.
    """
    excluded = set(excluded_tools)
    names = [name for name in QUICKJS_PTC_TOOLS if name not in excluded]

    for name in INBUILT_DANGEROUS_TOOLS:
        if name in PTC_INAPPLICABLE_TOOLS or name in excluded:
            continue
        if tool_enabled(config, name) and tool_ptc(config, name):
            names.append(name)

    metadata_by_name = {
        item.get("name", ""): item
        for item in metadata
        if item.get("name")
    }
    for tool in tools:
        name = tool_name(tool)
        if name in excluded or name in names:
            continue
        item = metadata_by_name.get(name, {})
        source = item.get("source")
        if source == "project" and tool_enabled(config, name) and tool_ptc(config, name):
            names.append(name)
        elif source == "mcp":
            policy = mcp_tool_policy(
                config,
                item.get("server", ""),
                item.get("original_name", ""),
            )
            if mcp_server_enabled(config, item.get("server", "")) and policy.enabled and policy.ptc:
                names.append(name)
    return names


def effective_rubric_tools(
    config: dict[str, Any] | None,
    backend: Any,
    tools: list[Any],
    metadata: list[dict[str, str]],
    excluded_tools: tuple[str, ...] = (),
) -> tuple[list[Any], dict[str, Any]]:
    """Build the grader's enabled tool surface and normal HITL policy."""
    excluded = set(excluded_tools)
    filesystem_names = list(QUICKJS_PTC_TOOLS)
    if (
        EXECUTE_TOOL not in excluded
        and tool_enabled(config, EXECUTE_TOOL)
        and tool_rubric_access(config, EXECUTE_TOOL)
    ):
        filesystem_names.append(EXECUTE_TOOL)
    resolved = list(FilesystemMiddleware(backend=backend, tools=filesystem_names).tools)
    selected = set(filesystem_names) - set(QUICKJS_PTC_TOOLS)
    metadata_by_name = {item.get("name", ""): item for item in metadata if item.get("name")}

    for tool in tools:
        name = tool_name(tool)
        item = metadata_by_name.get(name, {})
        source = item.get("source")
        if name in excluded or source not in {"project", "mcp"}:
            continue
        policy = tool_policy(config, name)
        enabled = policy.enabled
        if source == "mcp":
            server = item.get("server", "")
            policy = mcp_tool_policy(config, server, item.get("original_name", ""))
            enabled = mcp_server_enabled(config, server) and policy.enabled
        if not enabled or not policy.rubric:
            continue
        resolved.append(tool)
        selected.add(name)
    interrupts = _write_interrupts(config, metadata)
    return resolved, {name: rule for name, rule in interrupts.items() if name in selected}


def effective_excluded_tools(
    config: dict[str, Any] | None,
    excluded_tools: tuple[str, ...],
    enable_execute_backend: bool,
) -> tuple[str, ...]:
    """Return tool specs that should be hidden from the UI/model metadata."""
    blocked = set(excluded_tools)
    tools = hitl_settings(config).get("tools", {})
    if isinstance(tools, dict):
        blocked.update(
            name
            for name, spec in tools.items()
            if isinstance(name, str) and isinstance(spec, dict) and spec.get("enabled") is False
        )
    if not enable_execute_backend or not tool_enabled(config, EXECUTE_TOOL):
        blocked.add(EXECUTE_TOOL)
    return tuple(blocked)


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


def _attach_context_report_config(
    agent: Any,
    *,
    system_prompt: str,
    memory_sources: list[str],
    skill_sources: list[str],
) -> None:
    """Attach the exact construction inputs needed for local context audits."""
    try:
        agent.mira_context_report_config = {
            "system_prompt": system_prompt,
            "memory_sources": tuple(memory_sources),
            "skill_sources": tuple(skill_sources),
        }
    except AttributeError:
        return


def _attach_tool_failures(agent: Any, failures: list[Any]) -> None:
    """Attach optional project resource failures for terminal and TUI surfaces."""
    try:
        agent.mira_tool_failures = list(failures)
    except AttributeError:
        return


def _attach_resource_issues(agent: Any, issues: list[Any], discovery: Any) -> None:
    """Expose discovery diagnostics to the TUI without coupling construction."""
    try:
        agent.mira_resource_issues = list(issues)
        agent.mira_subagent_discovery = discovery
    except Exception:
        return


def _attach_backend(agent: Any, backend: Any, project_backend: Any) -> None:
    """Attach the workspace backend for approved filesystem fallback execution."""
    try:
        agent.mira_backend = backend
        agent.mira_project_backend = project_backend
    except AttributeError:
        return


def _attach_summarization(agent: Any, summarization: Any) -> None:
    """Attach DeepAgents summarization for post-turn compaction."""
    try:
        agent.mira_summarization = summarization
    except AttributeError:
        return


def _attach_rubric_model_name(agent: Any, model_name: str) -> None:
    """Attach the effective grader identity for progress and durable results."""
    try:
        agent.mira_rubric_model_name = model_name
    except AttributeError:
        return


def _attach_planning_context(agent: Any, context: PlanningToolContext) -> None:
    """Attach the per-agent dependencies passed to formal planning tools."""
    try:
        agent.mira_planning_context = context
    except AttributeError:
        return
