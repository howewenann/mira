"""Headless MIRA application lifecycle and shared resource ownership."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from contextlib import contextmanager, suppress
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
from config.settings import MAIN_MODEL, model_assignment, rubric_enabled, rubric_max_iterations
from core.interface import Frontend, FrontendEmitter, MCPApprovalRequest, NullFrontend
from core.execution.turns import plan_thread_id
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
    *(
        {"name": name, "description": ""}
        for name in (
            "write_todos",
            "ls",
            "read_file",
            "write_file",
            "edit_file",
            "glob",
            "grep",
            "validate_skill",
            "eval",
            "task",
        )
    ),
]


class MiraApplication:
    """Small owner of shared MIRA resources and headless session creation."""

    def __init__(self, *, frontend: Frontend, workspace: Path | str, **state: Any) -> None:
        self.frontend = frontend
        self.workspace = Path(workspace)
        for key, value in state.items():
            setattr(self, key, value)
        self.effective_model_alias = str(model_assignment(getattr(self, "config", None), MAIN_MODEL) or "") if getattr(self, "agent", None) is not None else ""
        self._sessions: dict[str, MiraSession] = {}
        self._closed = False
        self._runtime_transition = False
        self._policy_refresh_generation = 0
        self._policy_refreshed_generation = 0
        self._agent_refresh_lock = asyncio.Lock()
        from core.application.mcp import MiraMCP

        self.mcp = MiraMCP(self)
        from agent.workflows import MiraWorkflowAPI

        self.workflows = MiraWorkflowAPI(lambda: self.agent)
        self.reload_workflows()

    def reload_workflows(self) -> Any:
        """Replace the process-local workspace Workflow registry."""
        from agent.workflows import discover_workflows
        from core.diagnostics.issues import unique_issues

        previous = tuple(getattr(getattr(self, "workflow_registry", None), "issues", ()))
        registry = discover_workflows(self.workspace, self.workflows)
        self.workflow_registry = registry
        current_issues = [
            issue
            for issue in list(getattr(self, "issues", []) or [])
            if issue not in previous
        ]
        self.issues = unique_issues([*current_issues, *registry.issues])
        return registry

    @classmethod
    async def start(
        cls,
        *,
        workspace: Path | str,
        frontend: Frontend | None = None,
        config: dict[str, Any] | None = None,
        launch_options: Any = None,
    ) -> "MiraApplication":
        """Build MIRA's native agents/resources without constructing a UI."""
        from agent.mcp import MCPManager
        from config.runtime import LaunchOptions, load_effective_config
        from session.checkpoint import make_checkpointer
        from session.store import SessionStore

        workspace = Path(workspace).expanduser().resolve()
        frontend = frontend if frontend is not None else NullFrontend()
        if config is None:
            config = load_effective_config(workspace, launch_options or LaunchOptions())
        elif launch_options is not None:
            config = {**config, "llm_direct": launch_options.llm_direct}
        store = SessionStore(Path(config["session_dir"]))
        checkpointer = make_checkpointer()
        mcp_manager = MCPManager(workspace)
        emitter = FrontendEmitter(frontend)
        mcp_manager.set_activity_handler(
            lambda phase, detail: emitter.mcp(
                phase,
                server=str(detail.get("active_server") or ""),
                detail=detail,
            )
        )

        async def approve_mcp(state: Any, preview: str) -> Any:
            from core.interface.catalog import MCPApprovalInfo

            return await frontend.request(MCPApprovalRequest(MCPApprovalInfo(state.name, state.transport), preview))

        emitter.mcp("initializing")
        try:
            await mcp_manager.initialize(approve_mcp)
        except Exception as exc:
            emitter.mcp("error", detail=str(exc))
            with suppress(Exception):
                await mcp_manager.shutdown()
            from tracing.bootstrap import shutdown_tracing

            shutdown_tracing()
            raise
        emitter.mcp("initialized", detail=mcp_snapshot(mcp_manager))
        from core.application.runtime import reconstruct
        from config.runtime import LaunchOptions

        application = cls(
            frontend=frontend, workspace=workspace, agent=None, plan_agent=None,
            config=config, model_name="", context_limit_tokens=None,
            context_limit_source="unknown", store=store, checkpointer=checkpointer,
            mcp_manager=mcp_manager, tool_failures=[], issues=[], resource_metadata={},
            project_backend=None, agent_unavailable_message="", action_resources=None,
            agent_metadata=None, launch_options=launch_options or LaunchOptions(llm_direct=bool(config.get("llm_direct"))),
        )
        try:
            await reconstruct(application, config)
        except BaseException:
            try:
                await mcp_manager.shutdown()
            finally:
                from tracing.bootstrap import shutdown_tracing

                shutdown_tracing()
            raise
        mcp_manager.set_change_handler(application.mcp_registry_changed)
        return application

    @staticmethod
    def new_session_id() -> str:
        """Return an ID using MIRA's durable session naming convention."""
        from session.store import new_session_id

        return new_session_id()

    async def open_session(self, session_id: str | None = None, resume: bool = False) -> MiraSession:
        """Load one MIRA session; its ID remains distinct from graph thread IDs."""
        import aiosqlite

        from core.application.session import MiraSession
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        from session.graph_state import make_session_graph

        if self._closed:
            raise RuntimeError("MIRA application is shut down")
        if self._runtime_transition:
            raise RuntimeError("MIRA runtime configuration is changing")
        record = self.store.load(session_id, resume=resume, workspace=self.workspace)
        original_artifacts = (record.get("current_plan"), record.get("current_goal"))
        previous = self._sessions.get(str(record["id"]))
        if previous is not None:
            await previous.close()
        ensure_dashboard(
            record,
            model_name=self.model_name,
            context_limit_tokens=self.context_limit_tokens,
            context_limit_source=self.context_limit_source,
        )
        if not hasattr(self.store, "database_path"):
            session = MiraSession(self, record)
            FrontendEmitter(self.frontend, session_id=session.id).session_state(
                "opened", {"resume": bool(session_id or resume)}
            )
            return session
        connection = await aiosqlite.connect(str(self.store.database_path(str(record["id"]))))
        checkpointer = AsyncSqliteSaver(connection)
        try:
            if self.agent is not None and getattr(self, "action_resources", None) is not None:
                agent, plan_agent = build_agent_pair(
                    config=self.config,
                    workspace=self.workspace,
                    checkpointer=checkpointer,
                    metadata=self.agent_metadata,
                    mcp_manager=self.mcp_manager,
                    action_resources=self.action_resources,
                )
            else:
                agent, plan_agent = self.agent, self.plan_agent
        except BaseException:
            await connection.close()
            raise
        session = MiraSession(
            self, record, agent=agent, plan_agent=plan_agent,
            checkpointer=checkpointer, semantic_graph=make_session_graph(checkpointer),
        )
        try:
            await session.restore_graph_state()
        except BaseException:
            await session.close(persist=False)
            self._sessions.pop(session.id, None)
            raise
        self.agent, self.plan_agent, self.checkpointer = agent, plan_agent, checkpointer
        if (record.get("current_plan"), record.get("current_goal")) != original_artifacts:
            self.store.save_metadata(record)
        FrontendEmitter(self.frontend, session_id=session.id).session_state(
            "opened",
            {"resume": bool(session_id or resume)},
        )
        return session

    def session_exists(self, session_id: str) -> bool:
        """Return whether an explicit durable session already exists."""
        if not session_id:
            return False
        try:
            return self.store.path(session_id).is_file()
        except ValueError:
            return False

    def list_sessions(self) -> list[dict[str, Any]]:
        """List detached history summaries without transcript or checkpoint contents."""
        summaries = []
        for path in self.store.session_paths():
            try:
                record = self.store.read(path)
            except (OSError, ValueError):
                continue
            summary = {key: deepcopy(record.get(key)) for key in (
                "id", "title", "custom_title", "pinned", "workspace", "created_at", "updated_at", "turns", "dashboard",
            )}
            summary["latest_user_prompt"] = next((str(item.get("text") or "") for item in reversed(record.get("events", ()))
                if isinstance(item, dict) and item.get("type") == "user" and item.get("text")), "")
            summaries.append(summary)
        return summaries

    def update_session(self, session_id: str, *, title: str | None = None, pinned: bool | None = None) -> None:
        """Rename or pin a saved conversation without changing its recency."""
        session = self._sessions.get(session_id)
        record = session.record if session is not None and session.runtime_state != "closed" else self.store.read(self.store.path(session_id))
        if title is not None:
            record["custom_title"] = " ".join(title.split())
        if pinned is not None:
            record["pinned"] = bool(pinned)
        self.store.save_metadata(record)

    async def delete_session(self, session_id: str) -> bool:
        """Close a live checkpoint connection before deleting its files."""
        session = self._sessions.get(session_id)
        if session is not None:
            await session.close()
        return self.store.delete(session_id)

    async def clear_sessions(self) -> int:
        """Close every checkpoint connection before deleting saved sessions."""
        for session in tuple(self._sessions.values()):
            await session.close()
        return self.store.clear_all()

    def settings(self) -> dict[str, Any]:
        """Return detached workspace settings for an editor."""
        from config.settings import load_settings

        return deepcopy(load_settings(self.workspace))

    def commands(self, kind: str | None = None) -> tuple[Any, ...]:
        """Describe prompt, skill and workflow commands without executable factories."""
        from agent.skills import SkillRegistry
        from core.application.mcp import prompt_info
        from core.interface.catalog import CommandInfo

        result = []
        if kind in {None, "prompt"} and self.mcp_manager is not None:
            result.extend(prompt_info(spec) for spec in self.mcp_manager.prompt_registry.specs.values())
        if kind in {None, "skill"}:
            for command, skill in SkillRegistry((self.resource_metadata or {}).get("skills", [])).commands.items():
                result.append(CommandInfo(command, command, str(skill.get("description") or ""), kind="skill", name=str(skill.get("name") or "")))
        if kind in {None, "workflow"}:
            result.extend(CommandInfo(spec.command, spec.usage, kind="workflow", name=spec.name, arguments=deepcopy(spec.inputs))
                          for spec in self.workflow_registry.specs.values())
        return tuple(sorted(result, key=lambda item: item.command.casefold()))

    def workflow_invocation(self, text: str) -> tuple[Any, dict[str, Any]] | None:
        """Validate launch arguments and return metadata without graph factories."""
        from core.interface.catalog import CommandInfo

        resolved = self.workflow_registry.resolve(text)
        if resolved is None:
            return None
        spec, inputs = resolved
        return CommandInfo(spec.command, spec.usage, kind="workflow", name=spec.name,
                           arguments=deepcopy(spec.inputs)), deepcopy(inputs)

    async def prepare_input(self, text: str) -> Any:
        """Resolve an existing skill or prompt command into native input messages."""
        from agent.skills import SkillRegistry, prepare_skill

        registry = SkillRegistry((self.resource_metadata or {}).get("skills", []))
        prepared = prepare_skill(text, registry)
        if prepared is not None:
            return prepared
        if self.mcp_manager is not None:
            if text.startswith("/mcp__"):
                await self.mcp_manager.discover_prompts()
            return await self.mcp_manager.prompt_registry.resolve(text)
        return None

    def runtime(self) -> Any:
        """Describe the effective model connection without exposing credentials."""
        from config.runtime import LaunchOptions, build_runtime_snapshot

        return build_runtime_snapshot(self.config, getattr(self, "launch_options", LaunchOptions()), model_name=self.model_name)

    def configuration(self) -> dict[str, Any]:
        """Describe editable settings and available profiles without model credentials."""
        from core.application.preferences import configuration_view

        if self.config.get("tracing_registry") is None:
            from config.tracing import load_tracing_registry

            self.config["tracing_registry"] = load_tracing_registry(self.workspace)
        return {
            **configuration_view(self.config),
            "issues": deepcopy(list(getattr(self, "issues", []))),
            "tool_failures": deepcopy(list(getattr(self, "tool_failures", []))),
            "agent_unavailable_message": str(getattr(self, "agent_unavailable_message", "")),
        }

    def tracing_preview(self, settings: dict[str, Any]) -> str:
        """Describe effective tracing preferences with credentials redacted."""
        from core.application.preferences import _tracing_preview
        from config.tracing import load_tracing_registry

        registry = self.config.get("tracing_registry") or load_tracing_registry(self.workspace)
        return _tracing_preview(settings, registry)

    def report_error(self, error: BaseException, *, source: str, session_id: str | None = None, context: dict[str, Any] | None = None) -> Path:
        """Record a consumer error using the existing diagnostic reporting policy."""
        from core.diagnostics.error_report import write_error_report
        from core.diagnostics.logging import get_diagnostics_logger

        path = write_error_report(error, workspace=self.workspace, source=source, session_id=session_id, context=context)
        get_diagnostics_logger().exception("%s failed; error report: %s", source, path)
        return path

    async def list_files(self) -> list[str]:
        """List workspace files using the configured native backend."""
        from core.application.files import discover_project_files

        return await discover_project_files(self.project_backend) if self.project_backend is not None else []

    async def clear_uploads(self) -> bool:
        """Remove all existing imported-file state through the project backend."""
        from core.application.uploads import clear_local_uploads

        self._require_idle()
        return await clear_local_uploads(self.project_backend)

    def clear_diagnostics(self) -> int:
        """Remove saved error reports using the existing retention location."""
        from core.diagnostics.error_report import clear_error_reports

        return clear_error_reports(self.workspace)

    async def clear_history(self) -> tuple[int, int]:
        """Clear saved conversations and their existing compaction archives."""
        self._require_idle()
        count = await self.clear_sessions()
        return count, self.store.clear_compactions()

    def prompt_history(self) -> list[str]:
        from core.application.input_history import read_prompt_history

        return read_prompt_history(self.workspace / ".mira" / "history.txt")

    def record_prompt(self, text: str) -> None:
        from core.application.input_history import append_prompt_history

        append_prompt_history(self.workspace / ".mira" / "history.txt", text)

    def clear_prompt_history(self) -> None:
        path = self.workspace / ".mira" / "history.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    def configure_git(self, preference: bool) -> tuple[bool, str, dict[str, Any]]:
        """Apply the existing workspace Git-protection choice without UI input."""
        from core.application.preferences import configure_git
        from core.workspace import GIT_NOT_CONFIGURED_SUMMARY, git_protection_issue

        self._require_idle()
        ok, message, updated = configure_git(self.workspace, preference)
        if not ok:
            return ok, message, updated
        self.config["settings"] = updated
        issue = git_protection_issue(self.workspace, preference)
        self.issues = [item for item in self.issues if item.summary != GIT_NOT_CONFIGURED_SUMMARY]
        if issue is not None:
            self.issues.append(issue)
        self.config["issues"] = [item for item in self.config.get("issues", []) if item.summary != GIT_NOT_CONFIGURED_SUMMARY]
        if issue is not None:
            self.config["issues"].append(issue)
        self._runtime_changed()
        return True, message, deepcopy(updated)

    def _runtime_changed(self) -> None:
        from core.interface.events import RuntimeEvent

        self.frontend.emit(RuntimeEvent(kind="configuration", state="changed"))

    async def reload(self, *, restart_mcp: bool = False) -> Any:
        """Reload workspace configuration and agents, optionally restarting MCP."""
        from config.runtime import LaunchOptions, load_effective_config
        from core.application.runtime import reconstruct

        with self._changing_runtime():
            config = load_effective_config(self.workspace, getattr(self, "launch_options", LaunchOptions()), override_dotenv=True)
            if self.mcp_manager is not None:
                if restart_mcp:
                    self.mcp_manager.set_change_handler(None)
                    try:
                        await self.mcp_manager.reload()
                    finally:
                        self.mcp_manager.set_change_handler(self.mcp_registry_changed)
                else:
                    self.mcp_manager.prompt_registry.reload_local()
            await reconstruct(self, config)
            return self.runtime()

    async def refresh_model(self) -> None:
        """Refresh provider context metadata, rebuilding only when it changes."""
        with self._changing_runtime():
            await self._refresh_model()

    async def _refresh_model(self, session: Any = None) -> None:
        from agent.llm import get_llm, resolve_model_profile
        from config.llm import ConfigError
        from config.metadata import ModelMetadata, infer_model_metadata
        from core.application.runtime import reconstruct

        async with self._agent_refresh_lock:
            def another_operation() -> bool:
                return any(owner is not session and (owner.runtime_state in {"running", "cancelling"} or owner._transitioning)
                           for owner in self._sessions.values())

            if another_operation():
                return
            try:
                resolve_model_profile(self.config)
            except ConfigError:
                return
            metadata = await infer_model_metadata(self.config, model=get_llm(self.config, metadata=ModelMetadata()))
            if not another_operation() and metadata.context_tokens and metadata.context_tokens != self.context_limit_tokens:
                await reconstruct(self, self.config, inspect_model=False, metadata=metadata)

    @contextmanager
    def _changing_runtime(self):
        self._require_idle()
        self._runtime_transition = True
        try:
            yield
        finally:
            self._runtime_transition = False

    def _require_idle(self) -> None:
        if self._closed:
            raise RuntimeError("MIRA application is shut down")
        if self._runtime_transition:
            raise RuntimeError("MIRA runtime configuration is already changing")
        if any(session.runtime_state in {"running", "cancelling"} or session._transitioning for session in self._sessions.values()):
            raise RuntimeError("Finish the active operation before changing runtime configuration")

    async def apply_settings(self, settings: dict[str, Any], *, reset_mcp: bool = False) -> tuple[bool, str]:
        """Persist a configuration change and transactionally rebuild native agents."""
        from contextlib import nullcontext
        from config.settings import middleware_span_mode, normalize_settings, save_settings
        from core.application.runtime import reconstruct

        with self._changing_runtime():
            settings = normalize_settings(deepcopy(settings))
            previous = self.config
            old_settings = previous.get("settings") or self.settings()
            changed = old_settings != settings
            if not changed and not reset_mcp:
                return True, "settings unchanged"
            if changed and not save_settings(self.workspace, settings):
                return False, "could not save .mira/settings.yml"
            config = {**previous, "settings": settings}
            tracing_changed = any((old_settings.get("tracing") or {}).get(key) != (settings.get("tracing") or {}).get(key) for key in ("enabled", "profile"))
            try:
                policy = self.mcp_manager.reset_policy() if reset_mcp and self.mcp_manager is not None else nullcontext(False)
                async with policy as mcp_changed:
                    if not changed and not mcp_changed:
                        return True, "settings unchanged"
                    if tracing_changed and all(old_settings.get(key) == value for key, value in settings.items() if key != "tracing"):
                        self.config = config
                        self._runtime_changed()
                        return True, "tracing settings saved; Reload Runtime required"
                    await reconstruct(self, config, inspect_model=old_settings.get("models") != settings.get("models"))
            except Exception as exc:
                restored = save_settings(self.workspace, old_settings)
                self.config = previous
                if reset_mcp:
                    await reconstruct(self, previous, inspect_model=False)
                return False, (f"settings not saved: {exc}" if old_settings.get("models") != settings.get("models") else f"settings not saved; agent rebuild failed: {exc}") + ("" if restored else "; could not restore .mira/settings.yml")
            if tracing_changed:
                return True, "settings saved; agents rebuilt; tracing requires Reload Runtime"
            if old_settings.get("models") != settings.get("models") and settings["models"].get("main") is None:
                return True, "settings saved; agents rebuilt; Main is unset; select a model with /models"
            if middleware_span_mode(old_settings) != middleware_span_mode(settings):
                return True, "middleware span setting saved; agents rebuilt; no reload required"
            return True, "settings saved; agents rebuilt; no reload required"

    def persist_tool_always_allow(self, agent: Any, action: dict[str, Any]) -> None:
        """Persist one interrupted tool's existing MIRA always-allow policy."""
        from config.settings import (
            load_settings,
            save_settings,
            set_mcp_tool_policy_value,
            set_tool_always_allow,
        )

        name = str(action.get("name") or "")
        metadata = next(
            (
                item
                for item in (getattr(agent, "mira_tool_specs", None) or ())
                if isinstance(item, dict) and item.get("name") == name
            ),
            {},
        )
        settings = load_settings(self.workspace)
        if not name:
            raise RuntimeError("cannot persist Always allow for an unnamed tool")
        if metadata.get("source") == "mcp":
            server = str(metadata.get("server") or "")
            original_name = str(metadata.get("original_name") or "")
            if not server or not original_name:
                raise RuntimeError(f"cannot persist Always allow for MCP tool {name}: missing identity metadata")
            updated = set_mcp_tool_policy_value(
                settings, server, original_name, "always_allow", True
            )
        else:
            updated = set_tool_always_allow(settings, name, True)
        if not save_settings(self.workspace, updated):
            raise RuntimeError(f"could not persist Always allow for {name}")
        self.config["settings"] = updated
        self._policy_refresh_generation += 1

    async def refresh_agents_for_policy(self) -> None:
        """Install one narrow Act/Plan rebuild after completed policy-changing turns."""
        async with self._agent_refresh_lock:
            generation = self._policy_refresh_generation
            if generation <= self._policy_refreshed_generation:
                return
            from core.application.runtime import reconstruct

            await reconstruct(self, self.config, inspect_model=False, policy_only=True)
            self._policy_refreshed_generation = generation

    async def mcp_registry_changed(self) -> None:
        """Rebuild resource projections after the shared MCP lifecycle changes."""
        from config.settings import load_settings

        previous_models = (self.config.get("settings") or {}).get("models")
        self.config["settings"] = load_settings(self.workspace)
        from core.application.runtime import reconstruct

        async with self._agent_refresh_lock:
            await reconstruct(self, self.config, inspect_model=False,
                              preserve_model=previous_models == self.config["settings"].get("models"))

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
        from tracing.bootstrap import shutdown_tracing

        shutdown_tracing()
        self._closed = True


def build_agent_pair(
    *,
    config: dict[str, Any],
    workspace: Path,
    checkpointer: Any,
    metadata: Any,
    mcp_manager: Any,
    action_resources: Any,
) -> tuple[Any, Any]:
    """Build the shared Act/Plan pair without touching live MCP connections."""
    from agent.factory import build_agent, build_plan_agent
    from agent.resources import build_resources

    plan_resources = build_resources(
        workspace,
        create_examples=False,
        settings=config.get("settings"),
        enable_execute=False,
        config=config,
        subagent_discovery=action_resources.subagent_discovery,
        mcp_manager=mcp_manager,
    )
    common = {
        "config": config,
        "workspace": workspace,
        "checkpointer": checkpointer,
        "metadata": metadata,
        "mcp_manager": mcp_manager,
    }
    return (
        build_agent(resources=action_resources, **common),
        build_plan_agent(resources=plan_resources, **common),
    )


def initial_mode(
    agent: Any,
    plan_agent: Any,
    settings: dict[str, Any] | None = None,
    session: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return MIRA's mutable semantic state for one application session."""
    return {
        "planning": False,
        "mode_transition": None,
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
    if planning != mode.get("planning", False):
        mode["mode_transition"] = {"mode": "plan" if planning else "act"}
    mode["planning"] = planning
    mode["planning_stage"] = PLANNING_STAGE_PLAN_RESEARCH
    if planning:
        mode["plan_thread_id"] = plan_thread_id(session)


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
            normalized_item = {
                "name": name,
                "path": path,
                "source": source,
                "replaces": str(item.get("replaces") or ""),
            }
            if item.get("description"):
                normalized_item["description"] = str(item["description"])
            normalized.append(normalized_item)
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
                "skill_count": len(getattr(state, "skills", ()) or ()),
            }
        )
    return {
        "servers": tuple(servers),
        "issues": tuple(str(issue) for issue in (getattr(manager, "issues", ()) or ())),
        "capabilities": {
            "tools": sum(server["tool_count"] for server in servers),
            "prompts": sum(server["prompt_count"] for server in servers),
            "resources": sum(server["resource_count"] for server in servers),
            "skills": sum(server["skill_count"] for server in servers),
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
