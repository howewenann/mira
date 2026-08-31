"""Headless MIRA application lifecycle and shared resource ownership."""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agent.planning.policy import (
    FINALIZE_GOAL_TOOL,
    FINALIZE_PLAN_TOOL,
    PLAN_DISABLED_TOOLS,
    PLANNING_STAGE_GOAL_FINALIZE,
    PLANNING_STAGE_GOAL_RESEARCH,
    PLANNING_STAGE_PLAN_FINALIZE,
    PLANNING_STAGE_PLAN_RESEARCH,
    PREPARE_GOAL_TOOL,
    PREPARE_PLAN_TOOL,
    SHOW_GOAL_TOOL,
    SHOW_PLAN_TOOL,
)
from agent.tools.specs import mira_environment_label
from config.settings import rubric_enabled, rubric_max_iterations
from core.interface import Frontend, FrontendEmitter, MCPApprovalRequest
from core.execution.turns import plan_thread_id
from session.context import mark_resume_context_pending
from session.dashboard import ensure_dashboard
from session.goals import current_goal
from session.plans import current_plan

if TYPE_CHECKING:
    from core.application.session import MiraSession


DEFAULT_TOOL_SPECS = [
    {"name": "ask_user", "description": "Ask the user to choose between concrete next steps when MIRA is blocked."},
    {"name": "finalize_plan", "description": "Finalize a structured implementation Plan for explicit user review."},
    {"name": "prepare_plan", "description": "Begin criteria-first construction of a decision-complete Plan."},
    {"name": "prepare_goal", "description": "Begin criteria-first construction of a decision-complete Goal."},
    {"name": "finalize_goal", "description": "Finalize a Goal title after Success Criteria generation."},
    {"name": "show_plan", "description": "Render the exact retained current Plan."},
    {"name": "show_goal", "description": "Render the exact retained current Goal."},
    *({"name": name, "description": ""} for name in (
        "write_todos", "ls", "read_file", "write_file", "edit_file", "glob", "grep", "eval", "task"
    )),
]


class MiraApplication:
    """Small owner of shared MIRA resources and headless session creation."""

    def __init__(self, *, frontend: Frontend, workspace: Path | str, **state: Any) -> None:
        self.frontend = frontend
        self.workspace = Path(workspace)
        for key, value in state.items():
            setattr(self, key, value)
        self._sessions: dict[str, MiraSession] = {}
        self._closed = False

    @classmethod
    async def start(
        cls,
        *,
        workspace: Path | str,
        frontend: Frontend,
        config: dict[str, Any] | None = None,
    ) -> "MiraApplication":
        """Build MIRA's native agents/resources without constructing a UI."""
        from agent.factory import build_agent, build_plan_agent
        from agent.llm import active_model_issues, get_llm, get_model_name, model_unavailable_message
        from agent.mcp import MCPManager
        from agent.resources import build_resources, configure_subagents
        from agent.subagents.discovery import subagent_model_issues
        from config.metadata import ModelMetadata, infer_model_metadata
        from config.runtime import LaunchOptions, load_effective_config
        from session.checkpoint import make_checkpointer
        from session.store import SessionStore

        workspace = Path(workspace).expanduser().resolve()
        if config is None:
            config = load_effective_config(workspace, LaunchOptions())
        store = SessionStore(Path(config["session_dir"]))
        checkpointer = make_checkpointer()
        mcp_manager = MCPManager(workspace)
        emitter = FrontendEmitter(frontend)

        async def approve_mcp(state: Any, preview: str) -> Any:
            return await frontend.request(MCPApprovalRequest(state, preview))

        emitter.mcp("initializing")
        try:
            await mcp_manager.initialize(approve_mcp)
        except Exception as exc:
            emitter.mcp("error", detail=str(exc))
            with suppress(Exception):
                await mcp_manager.shutdown()
            raise
        emitter.mcp("initialized", detail=mcp_snapshot(mcp_manager))
        resources = build_resources(workspace, settings=config.get("settings"), config=None)
        if resources.subagent_discovery.complete and config.get("settings_valid", True):
            from config.settings import prune_subagent_settings, save_settings

            current_settings = config.get("settings") or {}
            pruned = prune_subagent_settings(
                current_settings,
                {item.name for item in resources.subagent_discovery.items},
            )
            if pruned != current_settings and save_settings(workspace, pruned):
                config["settings"] = pruned
        assignment_issues = [
            *active_model_issues(config),
            *subagent_model_issues(resources.subagent_discovery, config),
        ]
        issues = [
            *(config.get("issues") or []),
            *mcp_manager.issues,
            *mcp_manager.prompt_registry.issues,
            *resources.issues,
            *assignment_issues,
        ]
        blocking = not config.get("settings_valid", True) or bool(assignment_issues)
        metadata = ModelMetadata(
            context_tokens=int((config.get("settings") or {}).get("models", {}).get("context_limit_tokens", 32768)),
            context_source="settings.models.context_limit_tokens",
        )
        agent = None
        plan_agent = None
        if not blocking:
            action_resources = configure_subagents(resources, config)
            plan_resources = build_resources(
                workspace,
                create_examples=False,
                settings=config.get("settings"),
                enable_execute=False,
                config=config,
                subagent_discovery=resources.subagent_discovery,
            )
            inspect_model = get_llm(config, metadata=ModelMetadata())
            metadata = await infer_model_metadata(config, model=inspect_model)
            agent = build_agent(
                config=config,
                workspace=workspace,
                checkpointer=checkpointer,
                metadata=metadata,
                mcp_manager=mcp_manager,
                resources=action_resources,
            )
            plan_agent = build_plan_agent(
                config=config,
                workspace=workspace,
                checkpointer=checkpointer,
                metadata=metadata,
                mcp_manager=mcp_manager,
                resources=plan_resources,
            )
        from core.diagnostics.issues import unique_issues

        issues = unique_issues(
            [
                *issues,
                *getattr(agent, "mira_resource_issues", []),
            ]
        )
        return cls(
            frontend=frontend,
            workspace=workspace,
            agent=agent,
            plan_agent=plan_agent,
            config=config,
            model_name=get_model_name(config),
            context_limit_tokens=metadata.context_tokens,
            context_limit_source=metadata.context_source,
            store=store,
            checkpointer=checkpointer,
            mcp_manager=mcp_manager,
            tool_failures=resources.tool_failures,
            issues=issues,
            resource_metadata=resources.metadata,
            project_backend=resources.project_backend,
            agent_unavailable_message=model_unavailable_message(config) if blocking else "",
        )

    async def open_session(self, session_id: str | None = None, resume: bool = False) -> MiraSession:
        """Load one MIRA session; its ID remains distinct from graph thread IDs."""
        from core.application.session import MiraSession

        if self._closed:
            raise RuntimeError("MIRA application is shut down")
        record = self.store.load(session_id, resume=resume, workspace=self.workspace)
        mark_resume_context_pending(record, resumed=bool(session_id or resume))
        ensure_dashboard(
            record,
            model_name=self.model_name,
            context_limit_tokens=self.context_limit_tokens,
            context_limit_source=self.context_limit_source,
        )
        session = MiraSession(self, record)
        FrontendEmitter(self.frontend, session_id=session.id).session_state(
            "opened",
            {"resume": bool(session_id or resume)},
        )
        return session

    async def shutdown(self, *, persist_sessions: bool = True) -> None:
        """Close sessions and shared external resources."""
        if self._closed:
            return
        for session in tuple(self._sessions.values()):
            if session.runtime_state != "closed":
                await session.close(persist=persist_sessions)
        mcp_manager = getattr(self, "mcp_manager", None)
        if mcp_manager is not None:
            await mcp_manager.shutdown()
        self._closed = True


def initial_mode(
    agent: Any,
    plan_agent: Any,
    settings: dict[str, Any] | None = None,
    session: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return MIRA's mutable semantic state for one application session."""
    return {
        "planning": False,
        "current_plan": current_plan(session or {}),
        "executing_plan": False,
        "current_goal": current_goal(session or {}),
        "executing_goal": False,
        "plan_revision": None,
        "goal_staging": None,
        "goal_revision": None,
        "planning_stage": PLANNING_STAGE_PLAN_RESEARCH,
        "rubric_enabled": rubric_enabled(settings),
        "rubric_max_iterations": rubric_max_iterations(settings),
        "plan_counter": 0,
        "plan_runs": 0,
        "plan_thread_id": plan_thread_id(session) if session else "",
        "action_tools": tool_specs(agent),
        "planning_tools": tool_specs(plan_agent),
        "resources": resource_specs(agent),
    }


def select_mode(
    session: dict[str, Any],
    mode: dict[str, Any],
    requested: str,
) -> None:
    """Apply ACT/PLAN semantics to authoritative session state."""
    normalized = requested.strip().lower()
    if normalized not in {"act", "action", "plan", "planning"}:
        raise ValueError(f"unknown MIRA mode: {requested}")
    planning = normalized in {"plan", "planning"}
    mode["planning"] = planning
    mode["planning_stage"] = PLANNING_STAGE_PLAN_RESEARCH
    if planning:
        mode["plan_thread_id"] = plan_thread_id(session)
        mark_resume_context_pending(session, resumed=True)


def refresh_agent_specs(mode: dict[str, Any], agent: Any, plan_agent: Any) -> None:
    mode["action_tools"] = tool_specs(agent)
    mode["planning_tools"] = tool_specs(plan_agent)
    mode["resources"] = resource_specs(agent)


def available_tools(mode: dict[str, Any], *, planning: bool) -> list[dict[str, str]]:
    key = "planning_tools" if planning else "action_tools"
    tools = mode.get(key)
    if isinstance(tools, list) and tools:
        normalized = normalize_tool_specs(tools)
        if planning:
            if mode.get("planning_stage") in {PLANNING_STAGE_PLAN_FINALIZE, PLANNING_STAGE_GOAL_FINALIZE}:
                expected = FINALIZE_GOAL_TOOL if mode.get("planning_stage") == PLANNING_STAGE_GOAL_FINALIZE else FINALIZE_PLAN_TOOL
                return [tool for tool in normalized if tool["name"] == expected]
            expected_prepare = PREPARE_GOAL_TOOL if mode.get("planning_stage") == PLANNING_STAGE_GOAL_RESEARCH else PREPARE_PLAN_TOOL
            expected_show = SHOW_GOAL_TOOL if mode.get("planning_stage") == PLANNING_STAGE_GOAL_RESEARCH else SHOW_PLAN_TOOL
            hidden = {PREPARE_PLAN_TOOL, PREPARE_GOAL_TOOL, FINALIZE_PLAN_TOOL, FINALIZE_GOAL_TOOL} - {expected_prepare}
            visible = [tool for tool in normalized if tool["name"] not in hidden]
            visible.sort(key=lambda tool: tool["name"] != expected_show)
            return visible
        return normalized
    if not planning:
        blocked = {FINALIZE_PLAN_TOOL, FINALIZE_GOAL_TOOL, PREPARE_PLAN_TOOL, PREPARE_GOAL_TOOL}
        return [tool for tool in DEFAULT_TOOL_SPECS if tool["name"] not in blocked]
    blocked = set(PLAN_DISABLED_TOOLS)
    return [tool for tool in DEFAULT_TOOL_SPECS if tool["name"] not in blocked]


def tool_specs(agent: Any) -> list[dict[str, str]]:
    explicit = getattr(agent, "mira_tool_specs", None)
    if isinstance(explicit, list) and explicit:
        return normalize_tool_specs(explicit)
    get_tools = getattr(agent, "get_tools", None)
    if callable(get_tools):
        return normalize_tool_specs(get_tools())
    tools = getattr(agent, "tools", None)
    return normalize_tool_specs(tools) if isinstance(tools, list | tuple) else DEFAULT_TOOL_SPECS.copy()


def resource_specs(agent: Any) -> dict[str, list[dict[str, str]]]:
    resources = getattr(agent, "mira_resources", None)
    if not isinstance(resources, dict):
        return {"memories": [], "skills": [], "subagents": [], "tools": []}
    return {key: normalize_resource_items(resources.get(key, [])) for key in ("memories", "skills", "subagents", "tools")}


def resources_for(mode: dict[str, Any], key: str) -> list[dict[str, str]]:
    resources = mode.get("resources")
    return normalize_resource_items(resources.get(key, [])) if isinstance(resources, dict) else []


def normalize_resource_items(items: Any) -> list[dict[str, str]]:
    if not isinstance(items, list | tuple):
        return []
    normalized = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name, path, source = (str(item.get(key) or "") for key in ("name", "path", "source"))
        if name and path and source:
            normalized.append({"name": name, "path": path, "source": source, "replaces": str(item.get("replaces") or "")})
    return normalized


def normalize_tool_specs(tools: list[Any] | tuple[Any, ...]) -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []
    environment = mira_environment_label()
    for tool in tools:
        name = tool_name(tool)
        if not name:
            continue
        spec = {
            "name": name,
            "description": first_sentence(tool_description(tool)),
            "source": "built-in",
            "runtime": "MIRA",
            "environment": environment,
        }
        if isinstance(tool, dict):
            for key in ("source", "replaces", "path", "runtime", "environment"):
                if key in tool:
                    spec[key] = str(tool.get(key) or "")
        specs.append(spec)
    return specs


def tool_name(tool: Any) -> str:
    if isinstance(tool, dict):
        return str(tool.get("name") or "")
    return str(getattr(tool, "name", None) or getattr(tool, "__name__", None) or "")


def tool_description(tool: Any) -> str:
    if isinstance(tool, dict):
        return str(tool.get("description") or "").strip()
    description = getattr(tool, "description", None)
    if description:
        return str(description).strip()
    doc = getattr(tool, "__doc__", None)
    return doc.strip().splitlines()[0] if isinstance(doc, str) and doc.strip() else ""


def first_sentence(value: str) -> str:
    text = " ".join(line.strip() for line in value.splitlines() if line.strip())
    for index, character in enumerate(text):
        if character in {".", "!", "?"}:
            return text[: index + 1]
    return text


def mcp_snapshot(manager: Any | None) -> dict[str, Any]:
    """Return safe MCP status/capability metadata without live clients."""
    if manager is None:
        return {"servers": (), "issues": (), "capabilities": {}}
    servers = []
    for state in getattr(manager, "servers", {}).values():
        servers.append(
            {
                "name": str(getattr(state, "name", "") or ""),
                "transport": str(getattr(state, "transport", "") or ""),
                "status": str(getattr(state, "status", "") or ""),
                "error": str(getattr(state, "error", "") or ""),
                "tool_count": len(getattr(state, "tools", ()) or ()),
                "prompt_count": len(getattr(state, "prompts", ()) or ()),
                "resource_count": len(getattr(state, "resources", ()) or ()),
            }
        )
    return {
        "servers": tuple(servers),
        "issues": tuple(str(issue) for issue in (getattr(manager, "issues", ()) or ())),
        "capabilities": {
            "tools": sum(server["tool_count"] for server in servers),
            "prompts": sum(server["prompt_count"] for server in servers),
            "resources": sum(server["resource_count"] for server in servers),
        },
    }


__all__ = [
    "DEFAULT_TOOL_SPECS",
    "MiraApplication",
    "available_tools",
    "initial_mode",
    "mcp_snapshot",
    "normalize_tool_specs",
    "refresh_agent_specs",
    "resource_specs",
    "resources_for",
    "select_mode",
    "tool_specs",
]
