"""Textual application shell for MIRA."""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.exceptions import ContextOverflowError
from textual import on
from textual.actions import SkipAction
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.widgets import Button, ListView, Static

from agent.compaction import compact_after_turn
from agent.context_overflow import context_notice_rendered, pop_context_overflow_notice
from agent.planning.criteria import SuccessCriteriaService
from agent.planning.policy import (
    GOAL_FINALIZATION_POLICY,
    PLAN_FINALIZATION_POLICY,
    PLANNING_STAGE_GOAL_FINALIZE,
    PLANNING_STAGE_GOAL_RESEARCH,
    PLANNING_STAGE_PLAN_FINALIZE,
    PLANNING_STAGE_PLAN_RESEARCH,
)
from config.runtime import LaunchOptions, RuntimeSnapshot, build_runtime_snapshot
from config.settings import (
    EXECUTE_TOOL,
    git_protection_enabled,
    load_settings,
    rubric_enabled,
    rubric_max_iterations,
    save_settings,
    tool_enabled,
)
from runtime.diagnostics import get_diagnostics_logger
from runtime.error_report import clear_error_reports, write_error_report
from runtime.runner import FORMAL_CONSTRUCTION_CANCELLED
from runtime.trace_stream import TraceStream
from session.dashboard import ensure_dashboard, normalize_dashboard, update_duration
from session.context import append_event, sync_deepagents_compaction
from session.goals import (
    RESUMABLE_GOAL_STATUSES,
    clear_current_goal,
    current_goal,
    goal_artifact,
    pause_current_goal,
    replace_current_goal,
    start_goal_attempt,
)
from session.plans import (
    RESUMABLE_PLAN_STATUSES,
    clear_current_plan,
    current_plan,
    pause_current_plan,
    plan_artifact,
    plan_artifact_text,
    replace_current_plan,
    start_plan_attempt,
)
from session.recorder import update_goal_event_status, update_plan_event_status
from ui.interrupts import (
    ASK_USER_OPEN_OPTION,
    action_choices,
    action_preview,
    action_requests,
    action_title,
    ask_user_options,
    ask_user_question,
    ask_user_request,
    goal_title_request,
    plan_request,
    prepare_plan_request,
    prepare_goal_request,
)
from ui.repl import (
    available_tools,
    handle_command,
    initial_mode,
    plan_command_prompt,
    plan_revision_text,
    plan_thread_id,
    refresh_agent_specs,
    goal_revision_text,
    run_user_turn,
)
from ui.runtime_snapshot import runtime_report
from ui.widgets import (
    AutocompleteInput,
    ChatLog,
    PromptBox,
    PromptPanel,
    SessionHistory,
    SettingsPanel,
    StatusBar,
    TelemetryBar,
    SubagentsPanel,
    IssuesScreen,
    MCPPanelScreen,
)
from ui.widgets.mcp_panel import mcp_summary_symbol
from ui.widgets.chat_log import DEFAULT_TOOL_OUTPUT_CHARS
from ui.widgets.session_history import SessionItem
from ui.windows_clipboard import set_windows_clipboard
from ui.windows_input import driver_class_for_platform
from ui.windows_scrollbars import configure_scrollbars_for_platform

Bootstrap = Callable[[Path, str | None, bool, dict[str, Any] | None, Any | None], Awaitable[dict[str, Any]]]
GitGuard = Callable[[Path, Any], Any]
DESTRUCTIVE_HISTORY_COMMANDS = {"/clear-chat", "/clear-all-chats", "/clear-errors", "/clear-prompts"}
STARTUP_RECOVERY_COMMANDS = {
    "/settings",
    "/models",
    "/issues",
    "/runtime",
    "/reload-runtime",
    "/exit",
    "/quit",
}
ARTIFACT_STATUS_LABELS = {
    "proposed": "Draft",
    "active": "Active",
    "paused": "Paused",
    "max_iterations_reached": "Limit",
    "completed": "Done",
}
OPERATIONAL_STATES = {"starting", "ready", "running", "cancelling", "error"}
DESTRUCTIVE_CONFIRM_HINT = "Press O to confirm, C or Esc to cancel."
DESTRUCTIVE_CONFIRM_CHOICES = [("o", "OK (o)"), ("c", "Cancel (c)")]


class MiraApp(App[None]):
    """Textual-first interactive MIRA UI."""

    CSS_PATH = "styles/mira.tcss"
    manages_subagent_animation = True
    BINDINGS = [
        Binding("ctrl+c", "copy", "Copy", priority=True),
        Binding("alt+q", "interrupt_or_quit", "Cancel/Quit", priority=True),
        Binding("ctrl+l", "clear_log", "Clear"),
        Binding("escape", "focus_prompt", "Prompt"),
    ]

    def __init__(
        self,
        *,
        workspace: Path | None = None,
        resume: bool = False,
        session_id: str | None = None,
        config: dict[str, Any] | None = None,
        launch_options: LaunchOptions | None = None,
        bootstrap: Bootstrap | None = None,
        ensure_git_repository: GitGuard | None = None,
        prebuilt: dict[str, Any] | None = None,
        tool_output_chars: int = DEFAULT_TOOL_OUTPUT_CHARS,
    ) -> None:
        configure_scrollbars_for_platform(sys.platform)
        super().__init__(driver_class=driver_class_for_platform(sys.platform))
        self.workspace = workspace.expanduser().resolve() if workspace is not None else Path.cwd()
        self.resume = resume
        self.session_id = session_id
        self.config = config
        self.launch_options = launch_options or LaunchOptions()
        self.bootstrap = bootstrap
        self.ensure_git_repository = ensure_git_repository
        self.prebuilt = prebuilt
        self.tool_output_chars = int(tool_output_chars)
        self.history_path = self.workspace / ".mira" / "history.txt"
        self.persist_prompt_history = prebuilt is None

        self.agent: Any = None
        self.plan_agent: Any = None
        self.store: Any = None
        self.session: dict[str, Any] = {"id": "", "workspace": str(self.workspace), "turns": 0}
        self.model_name = ""
        self.context_limit_tokens: int | None = None
        self.context_limit_source = "unknown"
        self.runtime_snapshot: RuntimeSnapshot | None = None
        self.checkpointer: Any = None
        self.mode: dict[str, Any] = {"planning": False}
        self.ready = False
        self.busy = False
        self.status_state = "starting"
        self.turn_worker: Any | None = None
        self.confirming_interrupt = False
        self._waiting_task: Any | None = None
        self._waiting_generation = 0
        self._waiting_delay_seconds = 0.8
        self._waiting_label = "working..."
        self._main_stream_active = False
        self._settings_panel: SettingsPanel | None = None
        self.mcp_manager: Any | None = None
        self._mcp_spinner = 0
        self._reload_in_progress = False
        self.tool_failures: list[Any] = []
        self.issues: list[Any] = []
        self.agent_unavailable_message = "Main model is not configured. Run /models."
        self.trace = TraceStream.disabled(output_chars=self.tool_output_chars)
        self._subagent_live_active = False

    def compose(self) -> ComposeResult:
        """Compose the Textual layout."""
        with Horizontal(id="app-shell"):
            with Vertical(id="session-sidebar"):
                with Horizontal(id="session-sidebar-header"):
                    yield Static("Chat History", id="session-sidebar-title")
                    yield Button("+ New", id="new-chat")
                yield SessionHistory(id="sessions")
            with Vertical(id="main-panel"):
                with Horizontal(id="status-row"):
                    yield StatusBar(id="status")
                    yield Button("Artifact", id="artifact-status-button")
                    yield Button("MCP 0/0", id="mcp-status-button")
                    yield Button("Issues 0", id="issues-button")
                yield ChatLog(tool_output_chars=self.tool_output_chars, id="chat-log")
                yield PromptPanel()
                yield SubagentsPanel(id="subagents-panel")
                yield AutocompleteInput(
                    tool_provider=self._active_autocomplete_tools,
                    subagent_provider=self._active_autocomplete_subagents,
                )
                yield TelemetryBar(id="telemetry-row")

    def on_mount(self) -> None:
        """Start app initialization."""
        self.set_interval(1.0, self._tick_status)
        self.set_interval(0.12, self._tick_animations)
        self._set_status(state="starting")
        self._sync_sidebar_visibility()
        self.query_one(PromptBox).disabled = True
        if self.prebuilt is not None:
            self._install_state(self.prebuilt)
            self._notify_startup_issues()
            return
        self.query_one(ChatLog).startup_loading(workspace=str(self.workspace), state="starting...")
        self.call_after_refresh(self._start_startup_worker)

    def _start_startup_worker(self) -> None:
        """Start bootstrap after the first visible frame has rendered."""
        self.run_worker(self._startup(), name="startup", exclusive=True)

    async def _startup(self) -> None:
        """Run Git safety checks and build agents inside the TUI."""
        try:
            if self.ensure_git_repository is not None:
                self.startup_progress("checking workspace...")
                if not await self.ensure_git_repository(self.workspace, self):
                    self.exit()
                    return

            if self.bootstrap is None:
                raise RuntimeError("MIRA bootstrap function was not provided")

            self.startup_progress("loading model metadata...")
            state = await self.bootstrap(
                self.workspace,
                self.session_id,
                self.resume,
                self.config,
                self,
            )
            self._install_state(state)
            self._notify_startup_issues()
        except Exception as exc:
            error_path = self._write_error_report(exc, source="tui.startup")
            self.system_message(f"startup error: {exc}\nerror report: {error_path}", kind="error")
            self.busy = False
            self._set_status(state="error")
            prompt = self.query_one(PromptBox)
            prompt.disabled = False
            self.action_focus_prompt()

    async def _retry_startup(self) -> None:
        """Retry the complete bootstrap after an initial startup failure."""
        prompt = self.query_one(PromptBox)
        self.busy = True
        self._main_stream_active = False
        prompt.disabled = True
        self._set_status(state="running")
        await self._startup()

    def _install_state(self, state: dict[str, Any]) -> None:
        """Install bootstrapped agents and session state into the app."""
        self.agent = state["agent"]
        self.plan_agent = state["plan_agent"]
        self.config = state["config"]
        self.store = state["store"]
        self.session = state["session"]
        self.model_name = str(state.get("model_name") or "")
        self.context_limit_tokens = state.get("context_limit_tokens")
        self.context_limit_source = str(state.get("context_limit_source") or "unknown")
        self.checkpointer = state.get("checkpointer")
        self.mcp_manager = state.get("mcp_manager") or self.mcp_manager
        if self.mcp_manager is not None:
            self.mcp_manager.set_change_handler(self._mcp_registry_changed)
        self.tool_failures = list(state.get("tool_failures") or getattr(self.agent, "mira_tool_failures", []))
        self.issues = list(state.get("issues") or [])
        self.agent_unavailable_message = str(
            state.get("agent_unavailable_message") or "Main model is not configured. Run /models."
        )
        if not self.issues and self.tool_failures:
            from agent.resources.tool_failures import tool_failure_issues

            self.issues = tool_failure_issues(self.tool_failures)
        self.runtime_snapshot = build_runtime_snapshot(
            self.config or {},
            self.launch_options,
            model_name=self.model_name,
        )
        self.mode = initial_mode(
            self.agent,
            self.plan_agent,
            (self.config or {}).get("settings"),
            self.session,
        )
        if state.get("resource_metadata"):
            self.mode["resources"] = state["resource_metadata"]
        self.ready = True
        self.busy = False
        self._main_stream_active = False
        prompt = self.query_one(PromptBox)
        prompt.disabled = False
        prompt.set_history(read_prompt_history(self.history_path))
        autocomplete = self.query_one(AutocompleteInput)
        autocomplete.set_project_backend(
            getattr(self.agent, "mira_project_backend", None) or state.get("project_backend")
        )
        autocomplete.set_mcp_manager(self.mcp_manager)
        ensure_dashboard(
            self.session,
            model_name=self.model_name,
            context_limit_tokens=self.context_limit_tokens,
            context_limit_source=self.context_limit_source,
        )

        chat = self.query_one(ChatLog)
        self.query_one(SubagentsPanel).reset()
        chat.clear_log()
        chat.startup(
            model_name=self.model_name,
            session_id=self.session["id"],
            workspace=str(self.session["workspace"]),
        )
        chat.restore_session(self.session)
        self._refresh_sessions()
        self._set_status(state="ready")
        self._sync_issues()
        self.action_focus_prompt()

    def _write_error_report(
        self,
        exc: BaseException,
        *,
        source: str,
        context: dict[str, Any] | None = None,
    ) -> Path:
        """Write a report for an exception handled by the TUI."""
        report_context = {
            "workspace": str(self.workspace),
            "mode": self._mode_label(),
            "model": self.model_name,
        }
        if context:
            report_context.update(context)
        error_path = write_error_report(
            exc,
            workspace=self.workspace,
            source=source,
            session_id=str(self.session.get("id") or "") or None,
            context=report_context,
        )
        get_diagnostics_logger().exception("%s failed; error report: %s", source, error_path)
        return error_path

    @on(PromptBox.Submitted)
    async def submit_prompt(self, event: PromptBox.Submitted) -> None:
        """Handle submitted prompt text."""
        text = event.value.strip()
        prompt = self.query_one(PromptBox)
        prompt.value = ""
        if not text or self.busy:
            if text in DESTRUCTIVE_HISTORY_COMMANDS and self.busy:
                self.system_message("finish the current turn before clearing history", kind="warning")
            self.action_focus_prompt()
            return
        if not self.ready and not (
            self.status_state == "error" and text in STARTUP_RECOVERY_COMMANDS
        ):
            self.system_message(
                "MIRA has not finished starting. Use /settings, /models, /issues, "
                "/runtime, or /reload-runtime to recover.",
                kind="warning",
            )
            self.action_focus_prompt()
            return

        self.query_one(ChatLog).resume_tail_following()
        self._record_prompt_history(text)
        if text == "/runtime":
            self._render_runtime_snapshot()
            self.action_focus_prompt()
            return

        self.query_one(SubagentsPanel).prepare_turn()
        if text in DESTRUCTIVE_HISTORY_COMMANDS:
            self.run_worker(self._run_history_command(text), name="history-command", exclusive=False)
            return

        if text == "/settings":
            self.run_worker(self._run_settings_command("general"), name="settings-command", exclusive=False)
            return

        if text == "/models":
            self.run_worker(self._run_settings_command("models"), name="models-command", exclusive=False)
            return

        if text == "/issues":
            self._open_issues()
            self.action_focus_prompt()
            return

        if text == "/reload":
            self.run_worker(self._run_reload_command(), name="reload-command", exclusive=False)
            return

        if text == "/reload-runtime":
            if not self.ready:
                self.turn_worker = self.run_worker(
                    self._retry_startup(),
                    name="startup-retry",
                    exclusive=True,
                )
                return
            self.run_worker(
                self._run_reload_runtime_command(),
                name="reload-runtime-command",
                exclusive=False,
            )
            return

        if text == "/new-chat":
            self.run_worker(self._run_new_chat(), name="new-chat", exclusive=True)
            return

        if text == "/mcp":
            self.open_mcp_panel()
            self.action_focus_prompt()
            return

        if text == "/prompts":
            self.run_worker(self._run_prompts_command(), name="prompts-command", exclusive=False)
            return

        prepared = None
        if self.mcp_manager is not None:
            if text.startswith("/mcp__"):
                await self.mcp_manager.discover_prompts()
            try:
                prepared = await self.mcp_manager.prompt_registry.resolve(text)
            except ValueError as error:
                self.system_message(str(error), kind="warning")
                self.action_focus_prompt()
                return

        plan_prompt = plan_command_prompt(text)
        if plan_prompt is not None:
            await handle_command("/plan", self, self.session, self.model_name, self.mode)
            if not plan_prompt:
                self._set_ready_unless_error()
                self.action_focus_prompt()
                return
            # `/plan <prompt>` is exactly `/plan` followed by the same normal
            # Plan-mode message. Quoting inside the suffix is preserved.
            text = plan_prompt

        if text == "/goal-show":
            await self.show_goal()
            self.action_focus_prompt()
            return

        if text == "/goal-clear":
            self.clear_goal()
            self.action_focus_prompt()
            return

        if text == "/goal-resume":
            await self.resume_goal()
            return

        if text == "/goal" or text.startswith("/goal "):
            objective = text.removeprefix("/goal").strip()
            if not objective:
                self.system_message("usage: /goal <prompt>", kind="muted")
                self.action_focus_prompt()
                return
            if not self._model_agents_available():
                return
            self.busy = True
            self._main_stream_active = False
            self._set_status(state="running")
            prompt.disabled = True
            self.turn_worker = self.run_worker(
                self._run_goal_command(objective),
                name="goal-command",
                exclusive=True,
            )
            return

        if text == "/compact":
            if not self._model_agents_available():
                return
            self.busy = True
            self._main_stream_active = False
            self._set_status(state="running")
            prompt.disabled = True
            self.turn_worker = self.run_worker(
                self._run_compact_command(),
                name="compact-command",
                exclusive=True,
            )
            return

        if prepared is None and await handle_command(text, self, self.session, self.model_name, self.mode):
            if not self.busy:
                self._set_ready_unless_error()
            if text in {"/exit", "/quit"}:
                self.exit()
            elif not self.busy:
                self.action_focus_prompt()
            return

        if not self._model_agents_available():
            return

        self.busy = True
        self._main_stream_active = False
        self._set_status(state="running")
        prompt.disabled = True
        attachments = self.mcp_manager.attachments_from_text(text) if self.mcp_manager is not None else []
        self.turn_worker = self.run_worker(
            self._run_turn(
                text,
                prepared_messages=prepared.messages if prepared is not None else None,
                display_text=prepared.display_text if prepared is not None else None,
                attachments=attachments,
            ),
            name="turn",
            exclusive=True,
        )

    async def _run_turn(
        self,
        text: str,
        *,
        prepared_messages: list[Any] | None = None,
        display_text: str | None = None,
        attachments: list[dict[str, str]] | None = None,
    ) -> None:
        """Run one agent turn and restore prompt focus when done."""
        try:
            await self._refresh_model_metadata()
            turn_kwargs: dict[str, Any] = {}
            if display_text is not None:
                turn_kwargs["display_text"] = display_text
            if prepared_messages is not None:
                turn_kwargs["prepared_messages"] = prepared_messages
            if attachments:
                turn_kwargs["attachments"] = attachments
            await run_user_turn(
                agent=self.agent,
                plan_agent=self.plan_agent,
                renderer=self,
                store=self.store,
                session=self.session,
                mode=self.mode,
                text=text,
                **turn_kwargs,
                model_name=self.model_name,
                context_limit_tokens=self.context_limit_tokens,
                context_limit_source=self.context_limit_source,
            )
            self._refresh_sessions()
            self._set_status(state="ready")
        except asyncio.CancelledError:
            if self.mode.get("executing_goal"):
                pause_current_goal(self.session)
                self.mode["current_goal"] = current_goal(self.session)
                self.store.save(self.session)
                self.mode["executing_goal"] = False
                self._resolve_goal_attempt_status()
            if self.mode.get("executing_plan"):
                pause_current_plan(self.session)
                self.mode["current_plan"] = current_plan(self.session)
                self.store.save(self.session)
                self.mode["executing_plan"] = False
            self.mode["plan_staging"] = None
            self.mode["plan_revision"] = None
            self.mode["planning_stage"] = PLANNING_STAGE_PLAN_RESEARCH
            self.finish_turn(cancelled=True)
            self.system_message("turn cancelled", kind="warning")
            self._set_status(state="ready")
            raise
        except ContextOverflowError as exc:
            if self.mode.get("executing_goal"):
                pause_current_goal(self.session)
                self.mode["current_goal"] = current_goal(self.session)
                self.store.save(self.session)
                self.mode["executing_goal"] = False
                self._resolve_goal_attempt_status()
            if self.mode.get("executing_plan"):
                pause_current_plan(self.session)
                self.mode["current_plan"] = current_plan(self.session)
                self.store.save(self.session)
                self.mode["executing_plan"] = False
            self.mode["plan_staging"] = None
            self.mode["plan_revision"] = None
            self.mode["planning_stage"] = PLANNING_STAGE_PLAN_RESEARCH
            self.finish_turn(cancelled=True, tool_status="interrupted")
            if not context_notice_rendered(exc):
                self.system_message(pop_context_overflow_notice(exc), kind="info")
            self._set_status(state="ready")
        except Exception as exc:
            if self.mode.get("executing_goal"):
                pause_current_goal(self.session)
                self.mode["current_goal"] = current_goal(self.session)
                self.store.save(self.session)
                self.mode["executing_goal"] = False
                self._resolve_goal_attempt_status()
            if self.mode.get("executing_plan"):
                pause_current_plan(self.session)
                self.mode["current_plan"] = current_plan(self.session)
                self.store.save(self.session)
                self.mode["executing_plan"] = False
            self.mode["plan_staging"] = None
            self.mode["plan_revision"] = None
            self.mode["planning_stage"] = PLANNING_STAGE_PLAN_RESEARCH
            self.finish_turn(cancelled=True, tool_status="interrupted")
            error_path = self._write_error_report(exc, source="tui.turn")
            self.system_message(f"error: {exc}\nerror report: {error_path}", kind="error")
            self._set_status(state="error")
        finally:
            self.mode["goal_staging"] = None
            self.mode["goal_revision"] = None
            self.turn_worker = None
            self.busy = False
            self.finish_turn()
            prompt = self.query_one(PromptBox)
            prompt.disabled = False
            self.action_focus_prompt()

    async def _run_compact_command(self) -> None:
        """Compact the current action or planning conversation."""
        if self.mode["planning"]:
            active_agent = self.plan_agent
            thread_id = self.mode["plan_thread_id"]
        else:
            active_agent = self.agent
            thread_id = self.session["id"]

        self.compaction_started()
        try:
            result = await compact_after_turn(active_agent, thread_id)
            if result.compacted:
                if await sync_deepagents_compaction(self.session, active_agent, thread_id):
                    self.store.save(self.session)
                    self._refresh_sessions()
                self.compaction_finished("context compacted", resume_waiting=False)
            elif result.reason in {"no_messages", "nothing_to_compact"}:
                self.compaction_finished("nothing to compact", success=False, resume_waiting=False)
            else:
                self.compaction_finished(
                    f"context compaction unavailable: {result.reason}",
                    success=False,
                    resume_waiting=False,
                )
            self._set_status(state="ready")
        except asyncio.CancelledError:
            self.compaction_finished("context compaction cancelled", success=False, resume_waiting=False)
            self._set_status(state="ready")
            raise
        except Exception as exc:
            self.compaction_finished("context compaction failed", success=False, resume_waiting=False)
            error_path = self._write_error_report(exc, source="tui.compact")
            self.system_message(f"compaction error: {exc}\nerror report: {error_path}", kind="error")
            self._set_status(state="error")
        finally:
            self.turn_worker = None
            self.busy = False
            prompt = self.query_one(PromptBox)
            prompt.disabled = False
            self.action_focus_prompt()

    @on(Button.Pressed, ".plan-action")
    def press_plan_action(self, event: Button.Pressed) -> None:
        """Handle structured plan bubble actions."""
        event.stop()
        button_id = event.button.id or ""
        for action in ("implement", "revise", "close", "clear"):
            prefix = f"plan-{action}-"
            if button_id.startswith(prefix):
                self.run_worker(
                    self._handle_plan_action(action, button_id[len(prefix):]),
                    name=f"plan-{action}",
                    exclusive=False,
                )
                return

    @on(Button.Pressed, ".goal-action")
    def press_goal_action(self, event: Button.Pressed) -> None:
        """Handle Goal review actions."""
        event.stop()
        button_id = event.button.id or ""
        for action in ("implement", "revise", "close", "clear"):
            prefix = f"goal-{action}-"
            if button_id.startswith(prefix):
                self.run_worker(
                    self._handle_goal_action(action, button_id[len(prefix):]),
                    name=f"goal-{action}",
                    exclusive=False,
                )
                return

    @on(Button.Pressed, "#new-chat")
    def press_new_chat(self, event: Button.Pressed) -> None:
        """Start a fresh saved chat session from the sidebar action."""
        event.stop()
        self.run_worker(self._run_new_chat(), name="new-chat", exclusive=True)

    async def _run_new_chat(self) -> None:
        """Create and switch to a fresh session outside event handlers."""
        try:
            previous_session = self.session
            if self._handle_new_chat() and self.session is not previous_session:
                self._set_status(state="ready")
        except Exception as exc:
            self.system_message(f"new chat error: {exc}", kind="error")
            self._set_status(state="error")
        finally:
            self.action_focus_prompt()

    def _handle_new_chat(self) -> bool:
        """Create a fresh saved session while preserving the current one."""
        if self.busy:
            self.system_message("finish the current turn before starting a new chat", kind="warning")
            return True
        if self.store is None:
            self.system_message("new chat needs the normal session store", kind="warning")
            return True

        load = getattr(self.store, "load", None)
        if callable(load):
            session = load(None, resume=False, workspace=self.workspace)
        else:
            new = getattr(self.store, "new", None)
            if not callable(new):
                self.system_message("new chat needs the normal session store", kind="warning")
                return True
            session = new(session_id=None, workspace=self.workspace)
            save = getattr(self.store, "save", None)
            if callable(save):
                save(session)

        self._switch_to_session(session)
        self.system_message("started new chat", kind="info")
        return True

    def _switch_to_session(self, session: dict[str, Any]) -> None:
        """Install a new active session without rebuilding agents."""
        self.session = session
        self.mode = initial_mode(
            self.agent,
            self.plan_agent,
            (self.config or {}).get("settings"),
            self.session,
        )
        self.query_one(SubagentsPanel).reset()
        ensure_dashboard(
            self.session,
            model_name=self.model_name,
            context_limit_tokens=self.context_limit_tokens,
            context_limit_source=self.context_limit_source,
        )
        self._render_current_session()

    async def _run_prompts_command(self) -> None:
        """List the exact shared prompt registry after lazy MCP discovery."""
        if self.mcp_manager is None:
            self.system_message("No prompts loaded.", kind="muted")
            return
        await self.mcp_manager.discover_prompts()
        specs = sorted(self.mcp_manager.prompt_registry.specs.values(), key=lambda item: item.command.casefold())
        if not specs:
            self.system_message("No local or MCP prompts loaded.", kind="muted")
        else:
            lines = ["Prompts", *(f"{spec.usage}    {spec.description}" for spec in specs)]
            self.system_message("\n".join(lines), kind="info")
        self.action_focus_prompt()

    async def _handle_plan_action(self, action: str, plan_id: str) -> None:
        """Resolve the active structured plan."""
        plan = current_plan(self.session)
        if plan is None or str(plan.get("id") or "") != plan_id:
            self.system_message("that plan is no longer active", kind="warning")
            return
        if self.busy:
            self.system_message("finish the current turn before resolving the plan", kind="warning")
            return

        if action == "close":
            self.query_one(ChatLog).resolve_plan(plan_id, "closed")
            update_plan_event_status(self.session, plan_id, "closed")
            self.store.save(self.session)
            self.system_message(f"closed plan \"{plan_title(plan)}\"", kind="muted")
            self.action_focus_prompt()
            return

        if action == "clear":
            self.clear_plan()
            self.action_focus_prompt()
            return

        if action in {"revise", "implement"} and not self._model_agents_available():
            return

        if action == "revise":
            feedback = await self._prompt_text("Revise Plan", "What should MIRA change about this plan?")
            feedback = (feedback or "").strip()
            if not feedback:
                self.system_message(f"kept plan \"{plan_title(plan)}\" active", kind="muted")
                self.action_focus_prompt()
                return
            self.mode["planning"] = True
            self.mode["planning_stage"] = PLANNING_STAGE_PLAN_RESEARCH
            self.mode["plan_thread_id"] = plan_thread_id(self.session)
            self.mode["plan_revision"] = {"previous_plan": plan, "feedback": feedback}
            self.system_message(f"revising plan \"{plan_title(plan)}\"", kind="status")
            self.busy = True
            self._main_stream_active = False
            self._set_status(state="running")
            self.query_one(PromptBox).disabled = True
            self.turn_worker = self.run_worker(
                self._run_turn_for_plan_revision(plan, feedback),
                name="plan-revision",
                exclusive=True,
            )
            return

        if action == "implement":
            started = start_plan_attempt(self.session)
            if started is None:
                self.system_message("no current Plan", kind="muted")
                return
            self.mode["current_plan"] = started
            self.mode["executing_plan"] = True
            self.query_one(ChatLog).resolve_plan(plan_id, "active")
            self.store.save(self.session)
            self.mode["planning"] = False
            self.system_message(f"implementing plan \"{plan_title(plan)}\"", kind="status")
            self.busy = True
            self._main_stream_active = False
            self._set_status(state="running")
            self.query_one(PromptBox).disabled = True
            self.turn_worker = self.run_worker(
                self._run_turn_for_plan(plan),
                name="plan-implementation",
                exclusive=True,
            )

    async def _run_goal_command(self, objective: str) -> None:
        """Run one explicit Goal request through the read-only Goal agent."""
        try:
            if not await self._confirm_formal_replacement("Goal"):
                self.system_message("kept the current formal work", kind="muted")
                return
            await self._refresh_model_metadata()
            self.mode["plan_runs"] = self.mode.get("plan_runs", 0) + 1
            self.mode["goal_staging"] = {
                "authoritative_objective": objective,
                "objective": objective,
                "context_and_constraints": "",
                "research_evidence": "",
                "success_criteria": "",
                "stage": PLANNING_STAGE_GOAL_RESEARCH,
                "thread_id": plan_thread_id(self.session, self.mode["plan_runs"]),
                "replacement_confirmed": True,
            }
            await run_user_turn(
                agent=self.agent,
                plan_agent=self.plan_agent,
                renderer=self,
                store=self.store,
                session=self.session,
                mode=self.mode,
                text=objective,
                display_text=f"/goal {objective}",
                record_user=True,
                model_name=self.model_name,
                context_limit_tokens=self.context_limit_tokens,
                context_limit_source=self.context_limit_source,
            )
            self._refresh_sessions()
            self._set_status(state="ready")
        except asyncio.CancelledError:
            self.system_message("Goal generation cancelled", kind="warning")
            raise
        except Exception as exc:
            error_path = self._write_error_report(exc, source="tui.goal")
            self.system_message(f"error: {exc}\nerror report: {error_path}", kind="error")
            self._set_status(state="error")
        finally:
            self.mode["goal_staging"] = None
            self.mode["goal_revision"] = None
            self.waiting_finished()
            self._waiting_label = "working..."
            self.turn_worker = None
            self.busy = False
            if self.status_state != "error":
                self._set_status(state="ready")
            prompt = self.query_one(PromptBox)
            prompt.disabled = False
            self.action_focus_prompt()

    async def _handle_goal_action(self, action: str, goal_id: str) -> None:
        """Resolve an action from the current Goal bubble."""
        value = current_goal(self.session)
        if value is None or str(value.get("id") or "") != goal_id:
            self.system_message("that Goal is no longer current", kind="warning")
            return
        if self.busy:
            self.system_message("finish the current turn before resolving the Goal", kind="warning")
            return

        if action == "close":
            self.query_one(ChatLog).resolve_goal(goal_id, "closed")
            update_goal_event_status(self.session, goal_id, "closed")
            self.store.save(self.session)
            self.system_message(f'closed Goal "{goal_title(value)}"', kind="muted")
            self.action_focus_prompt()
            return

        if action == "clear":
            self.clear_goal()
            self.action_focus_prompt()
            return

        if action in {"revise", "implement"} and not self._model_agents_available():
            return

        if action == "revise":
            feedback = await self._prompt_text("Revise Goal", "What should MIRA change about this Goal?")
            feedback = (feedback or "").strip()
            if not feedback:
                self.system_message(f'kept Goal "{goal_title(value)}"', kind="muted")
                self.action_focus_prompt()
                return
            self.busy = True
            self._main_stream_active = False
            self._set_status(state="running")
            self.query_one(PromptBox).disabled = True
            self.turn_worker = self.run_worker(
                self._run_goal_revision(value, feedback),
                name="goal-revision",
                exclusive=True,
            )
            return

        if action == "implement":
            started = start_goal_attempt(self.session)
            if started is None:
                self.system_message("no current Goal", kind="muted")
                return
            self.mode["current_goal"] = started
            self.mode["executing_goal"] = True
            self.query_one(ChatLog).resolve_goal(goal_id, "active")
            self.store.save(self.session)
            self.mode["planning"] = False
            self.mode["planning_stage"] = None
            self.system_message(f'implementing Goal "{goal_title(started)}"', kind="status")
            self.busy = True
            self._main_stream_active = False
            self._set_status(state="running")
            self.query_one(PromptBox).disabled = True
            self.turn_worker = self.run_worker(
                self._run_turn_for_goal(started),
                name="goal-implementation",
                exclusive=True,
            )

    async def _run_turn_for_goal(self, value: dict[str, Any]) -> None:
        """Run the normal action agent for one explicit Goal attempt."""
        try:
            await self._refresh_model_metadata()
            await run_user_turn(
                agent=self.agent,
                plan_agent=self.plan_agent,
                renderer=self,
                store=self.store,
                session=self.session,
                mode=self.mode,
                text="Work toward the retained Goal.",
                display_text=f"Implement Goal: {goal_title(value)}",
                record_user=False,
                model_name=self.model_name,
                context_limit_tokens=self.context_limit_tokens,
                context_limit_source=self.context_limit_source,
            )
            self._resolve_goal_attempt_status()
            self._refresh_sessions()
            self._set_status(state="ready")
        except asyncio.CancelledError:
            pause_current_goal(self.session)
            self.mode["current_goal"] = current_goal(self.session)
            self.mode["executing_goal"] = False
            self.store.save(self.session)
            self._resolve_goal_attempt_status()
            self.finish_turn(cancelled=True)
            self.system_message("turn cancelled", kind="warning")
            raise
        except Exception as exc:
            pause_current_goal(self.session)
            self.mode["current_goal"] = current_goal(self.session)
            self.mode["executing_goal"] = False
            self.store.save(self.session)
            self._resolve_goal_attempt_status()
            self.finish_turn(cancelled=True, tool_status="interrupted")
            error_path = self._write_error_report(exc, source="tui.goal_turn", context={"goal": goal_title(value)})
            self.system_message(f"error: {exc}\nerror report: {error_path}", kind="error")
            self._set_status(state="error")
        finally:
            self.turn_worker = None
            self.busy = False
            self.finish_turn()
            if self.status_state != "error":
                self._set_status(state="ready")
            self.query_one(PromptBox).disabled = False
            self.action_focus_prompt()

    async def _run_goal_revision(self, value: dict[str, Any], feedback: str) -> None:
        """Run feedback through the same read-only Goal pipeline."""
        try:
            await self._refresh_model_metadata()
            self.mode["plan_runs"] = self.mode.get("plan_runs", 0) + 1
            thread_id = plan_thread_id(self.session, self.mode["plan_runs"])
            self.mode["goal_revision"] = {"previous_goal": value, "feedback": feedback}
            self.mode["goal_staging"] = {
                "authoritative_objective": str(value.get("objective") or ""),
                "objective": str(value.get("objective") or ""),
                "context_and_constraints": "",
                "research_evidence": "",
                "success_criteria": "",
                "stage": PLANNING_STAGE_GOAL_RESEARCH,
                "thread_id": thread_id,
                "replacement_confirmed": True,
            }
            await run_user_turn(
                agent=self.agent,
                plan_agent=self.plan_agent,
                renderer=self,
                store=self.store,
                session=self.session,
                mode=self.mode,
                text=goal_revision_text(value, feedback),
                display_text=f"Revise Goal: {feedback}",
                record_user=True,
                model_name=self.model_name,
                context_limit_tokens=self.context_limit_tokens,
                context_limit_source=self.context_limit_source,
            )
            self._refresh_sessions()
            self._set_status(state="ready")
        except asyncio.CancelledError:
            self.finish_turn(cancelled=True)
            self.system_message("revision cancelled", kind="warning")
            raise
        except Exception as exc:
            self.finish_turn(cancelled=True, tool_status="interrupted")
            error_path = self._write_error_report(exc, source="tui.goal_revision")
            self.system_message(f"error: {exc}\nerror report: {error_path}", kind="error")
            self._set_status(state="error")
        finally:
            self.mode["goal_staging"] = None
            self.mode["goal_revision"] = None
            self.turn_worker = None
            self.busy = False
            self.finish_turn()
            if self.status_state != "error":
                self._set_status(state="ready")
            self.query_one(PromptBox).disabled = False
            self.action_focus_prompt()

    async def _run_turn_for_plan(self, plan: dict[str, Any]) -> None:
        """Run action mode from an approved structured plan."""
        try:
            await self._refresh_model_metadata()
            await run_user_turn(
                agent=self.agent,
                plan_agent=self.plan_agent,
                renderer=self,
                store=self.store,
                session=self.session,
                mode=self.mode,
                text="Implement the approved plan.",
                display_text=f"Implement plan: {plan_title(plan)}",
                record_user=False,
                model_name=self.model_name,
                context_limit_tokens=self.context_limit_tokens,
                context_limit_source=self.context_limit_source,
            )
            self._refresh_sessions()
            self._set_status(state="ready")
        except asyncio.CancelledError:
            pause_current_plan(self.session)
            self.mode["current_plan"] = current_plan(self.session)
            self.mode["executing_plan"] = False
            self.store.save(self.session)
            self.finish_turn(cancelled=True)
            self.system_message("turn cancelled", kind="warning")
            self._set_status(state="ready")
            raise
        except Exception as exc:
            pause_current_plan(self.session)
            self.mode["current_plan"] = current_plan(self.session)
            self.mode["executing_plan"] = False
            self.store.save(self.session)
            self.finish_turn(cancelled=True, tool_status="interrupted")
            error_path = self._write_error_report(
                exc,
                source="tui.plan_turn",
                context={"plan": plan_title(plan)},
            )
            self.system_message(f"error: {exc}\nerror report: {error_path}", kind="error")
            self._set_status(state="error")
        finally:
            self.turn_worker = None
            self.busy = False
            self.finish_turn()
            prompt = self.query_one(PromptBox)
            prompt.disabled = False
            self.action_focus_prompt()

    async def _run_turn_for_plan_revision(self, plan: dict[str, Any], feedback: str) -> None:
        """Run planning mode with the current plan and revision feedback."""
        try:
            await self._refresh_model_metadata()
            await run_user_turn(
                agent=self.agent,
                plan_agent=self.plan_agent,
                renderer=self,
                store=self.store,
                session=self.session,
                mode=self.mode,
                text=plan_revision_text(plan, feedback),
                display_text=f"Revise plan: {feedback}",
                record_user=True,
                model_name=self.model_name,
                context_limit_tokens=self.context_limit_tokens,
                context_limit_source=self.context_limit_source,
            )
            self._refresh_sessions()
            self._set_status(state="ready")
        except asyncio.CancelledError:
            self.mode["plan_staging"] = None
            self.mode["plan_revision"] = None
            self.mode["planning_stage"] = PLANNING_STAGE_PLAN_RESEARCH
            self.finish_turn(cancelled=True)
            self.system_message("turn cancelled", kind="warning")
            self._set_status(state="ready")
            raise
        except Exception as exc:
            self.mode["plan_staging"] = None
            self.mode["plan_revision"] = None
            self.mode["planning_stage"] = PLANNING_STAGE_PLAN_RESEARCH
            self.finish_turn(cancelled=True, tool_status="interrupted")
            error_path = self._write_error_report(
                exc,
                source="tui.plan_revision",
                context={"plan": plan_title(plan)},
            )
            self.system_message(f"error: {exc}\nerror report: {error_path}", kind="error")
            self._set_status(state="error")
        finally:
            self.turn_worker = None
            self.busy = False
            self.finish_turn()
            prompt = self.query_one(PromptBox)
            prompt.disabled = False
            self.action_focus_prompt()

    def _next_goal_id(self) -> str:
        """Return a session-local stable Goal id without colliding after reload."""
        existing = {
            str(event.get("goal", {}).get("id") or "")
            for event in self.session.get("events", [])
            if isinstance(event, dict) and isinstance(event.get("goal"), dict)
        }
        counter = int(self.mode.get("plan_counter") or 0)
        while True:
            counter += 1
            goal_id = f"goal-{counter}"
            if goal_id not in existing:
                self.mode["plan_counter"] = counter
                return goal_id

    def _next_plan_id(self) -> str:
        """Return a session-local stable Plan id without colliding after reload."""
        existing = {
            str(event.get("plan", {}).get("id") or "")
            for event in self.session.get("events", [])
            if isinstance(event, dict) and isinstance(event.get("plan"), dict)
        }
        counter = int(self.mode.get("plan_counter") or 0)
        while True:
            counter += 1
            plan_id = f"plan-{counter}"
            if plan_id not in existing:
                self.mode["plan_counter"] = counter
                return plan_id

    def _resolve_goal_attempt_status(self) -> None:
        """Reflect a finished Goal attempt in its execution bubble."""
        value = current_goal(self.session)
        if value is None or value.get("status") in {"proposed", "active"}:
            return
        self.query_one(ChatLog).resolve_goal(
            str(value.get("id") or ""),
            str(value.get("status") or "completed"),
        )

    def action_interrupt_or_quit(self) -> None:
        """Confirm before cancelling a turn or quitting the app."""
        if self.confirming_interrupt:
            return
        if self.query_one(PromptPanel).active:
            self._cancel_turn()
            return
        self.run_worker(self._confirm_interrupt_or_quit(), name="confirm-interrupt", exclusive=False)

    async def _confirm_interrupt_or_quit(self) -> None:
        """Ask for confirmation before handling the cancel/quit shortcut."""
        self.confirming_interrupt = True
        try:
            if self.busy and self.turn_worker is not None:
                answer = await self._prompt_choice(
                    "Cancel Turn?",
                    "MIRA is still working. Cancel this turn?",
                    [("y", "Yes (y)"), ("n", "No (n)")],
                )
                if answer == "y" and self.busy and self.turn_worker is not None:
                    self._cancel_turn()
                return

            answer = await self._prompt_choice(
                "Exit MIRA?",
                "No cancellable turn is running. Exit MIRA?",
                [("y", "Yes (y)"), ("n", "No (n)")],
            )
            if answer == "y":
                self.exit()
        finally:
            self.confirming_interrupt = False

    def _cancel_turn(self) -> None:
        """Cancel the active turn worker."""
        if self.busy and self.turn_worker is not None:
            self.finish_turn(cancelled=True)
            self.turn_worker.cancel()
            self._set_status(state="cancelling")

    def action_clear_log(self) -> None:
        """Clear chat and tool output."""
        self.clear_log()
        self._set_ready_unless_error()

    def action_copy(self) -> None:
        """Copy screen text first, then the focused widget selection."""
        selected_text = self.screen.get_selected_text()
        if selected_text:
            self.copy_to_clipboard(selected_text)
            return

        focused = self.focused
        copy_action = getattr(focused, "action_copy", None)
        if copy_action is None:
            return
        try:
            copy_action()
        except SkipAction:
            return

    def copy_to_clipboard(self, text: str) -> None:
        """Copy text with a native Windows path and Textual elsewhere."""
        if sys.platform != "win32":
            super().copy_to_clipboard(text)
            return

        self._clipboard = text
        try:
            set_windows_clipboard(text)
        except OSError as error:
            get_diagnostics_logger().warning("Windows clipboard copy failed: %s", error)

    def action_focus_prompt(self) -> None:
        """Focus the prompt input."""
        if not self.is_mounted:
            return
        self._sync_artifact_button()
        try:
            self.query_one(PromptBox).focus()
        except NoMatches:
            return

    def _active_autocomplete_tools(self) -> list[dict[str, str]]:
        """Return the same live tool projection shown by /tools."""
        return available_tools(self.mode, planning=bool(self.mode.get("planning")))

    def _active_autocomplete_subagents(self) -> list[dict[str, str]]:
        """Return enabled effective subagents for the shared @ surface."""
        resources = self.mode.get("resources") or {}
        items = resources.get("subagents") or []
        from config.settings import subagent_enabled

        return [
            item
            for item in items
            if item.get("name") and subagent_enabled(self.config or {}, str(item["name"]))
        ]

    def _model_agents_available(self) -> bool:
        """Stop at a model-backed boundary without treating configuration as a crash."""
        active = self.plan_agent if self.mode.get("planning") else self.agent
        if active is not None:
            return True
        self.system_message(self.agent_unavailable_message, kind="warning")
        self._set_ready_unless_error()
        self.action_focus_prompt()
        return False

    def show_mira_splash(self) -> None:
        """Append a fresh splash using the current session metadata."""
        self.query_one(ChatLog).append_splash(
            model_name=self.model_name,
            session_id=str(self.session["id"]),
            workspace=str(self.session["workspace"]),
        )

    def user_message(self, text: str, *, planning: bool = False, created_at: str = "") -> None:
        """Write a submitted user message to the chat log."""
        self.trace.user_message(text, planning=planning)
        self.query_one(ChatLog).timestamped_user_message(text, planning=planning, created_at=created_at)

    def system_message(self, text: str, *, kind: str = "system", created_at: str = "") -> None:
        """Write a presentation-only message to the trace and chat log."""
        self.trace.system_message(text, kind=kind)
        self.query_one(ChatLog).system_message(text, kind=kind, created_at=created_at)

    def command_output(self, renderable: Any) -> None:
        """Write command output to the chat log."""
        self.trace.command_output(renderable)
        self.waiting_finished()
        self.query_one(ChatLog).command_output(renderable)

    async def prepare_plan(self, interrupt: Any) -> str:
        """Generate Success Criteria before the forced Plan finalisation call."""
        self.waiting_finished()
        revision = self.mode.get("plan_revision")
        confirm = getattr(self, "_confirm_formal_replacement", None)
        accepted = await confirm("Plan") if not isinstance(revision, dict) and callable(confirm) else True
        if not accepted:
            self.mode["plan_staging"] = None
            self.mode["planning_stage"] = PLANNING_STAGE_PLAN_RESEARCH
            self.system_message("kept the current formal work", kind="muted")
            return FORMAL_CONSTRUCTION_CANCELLED
        request = prepare_plan_request(interrupt)
        objective = request["objective"]
        context = request["context_and_constraints"] or "No additional constraints."
        if not objective:
            raise RuntimeError("prepare_plan requires the authoritative user objective")
        self.waiting_started("Generating success criteria", immediate=True, elapsed=True)
        service = SuccessCriteriaService(self.config or {})
        if isinstance(revision, dict) and isinstance(revision.get("previous_plan"), dict):
            previous = revision["previous_plan"]
            criteria = await service.revise(
                objective,
                str(previous.get("success_criteria") or ""),
                str(revision.get("feedback") or ""),
                context,
            )
        else:
            criteria = await service.generate(objective, context)
        self.mode["plan_staging"] = {
            "objective": objective,
            "context_and_constraints": context,
            "success_criteria": criteria,
        }
        self.mode["planning_stage"] = PLANNING_STAGE_PLAN_FINALIZE
        self.waiting_started("Creating plan", immediate=True, elapsed=True)
        revision_context = ""
        if isinstance(revision, dict) and isinstance(revision.get("previous_plan"), dict):
            revision_context = (
                "\n\nThe revision must be a complete replacement.\n"
                f"<previous_plan>\n{plan_artifact_text(revision['previous_plan'])}\n</previous_plan>\n"
                f"<user_feedback>\n{revision.get('feedback') or ''}\n</user_feedback>"
            )
        return (
            f"{PLAN_FINALIZATION_POLICY}\n\n"
            f"<objective>\n{objective}\n</objective>\n\n"
            f"<context_and_constraints>\n{context}\n</context_and_constraints>\n\n"
            f"<success_criteria>\n{criteria}\n</success_criteria>"
            f"{revision_context}"
        )

    async def finalize_plan(self, interrupt: Any) -> str:
        """Persist and render one criteria-first PlanArtifact."""
        self.waiting_finished()
        payload = plan_request(interrupt)
        staging = self.mode.get("plan_staging")
        if not isinstance(staging, dict):
            raise RuntimeError("finalize_plan requires staged Plan context and Success Criteria")
        plan_id = self._next_plan_id()
        artifact = plan_artifact(
            plan_id=plan_id,
            title=str(payload.get("title") or "Plan"),
            objective=str(staging.get("objective") or ""),
            context_and_constraints=str(staging.get("context_and_constraints") or ""),
            key_changes=list(payload.get("key_changes") or []),
            test_plan=list(payload.get("test_plan") or []),
            assumptions=list(payload.get("assumptions") or []),
            success_criteria=str(staging.get("success_criteria") or ""),
            rubric_enabled=bool(self.mode.get("rubric_enabled")),
            rubric_iterations=int(self.mode.get("rubric_max_iterations") or 3),
        )

        previous_plan = current_plan(self.session)
        previous_goal = current_goal(self.session)
        if previous_plan is not None:
            self.query_one(ChatLog).resolve_plan(str(previous_plan["id"]), "superseded")
        if previous_goal is not None:
            self.query_one(ChatLog).resolve_goal(str(previous_goal["id"]), "superseded")

        artifact = replace_current_plan(self.session, artifact)
        self.mode["current_plan"] = artifact
        self.mode["current_goal"] = None
        self.mode["plan_staging"] = None
        self.mode["plan_revision"] = None
        self.mode["planning_stage"] = PLANNING_STAGE_PLAN_RESEARCH
        self._sync_artifact_button()
        event = append_event(
            self.session,
            {"type": "plan", "mode": "planning", "plan": artifact, "status": "proposed"},
        )
        self.store.save(self.session)
        self.query_one(ChatLog).present_plan(
            artifact,
            active=True,
            status="proposed",
            created_at=str(event.get("created_at") or ""),
        )
        return "Plan and Success Criteria presented for user review."

    async def show_plan(self, interrupt: Any = None) -> str:  # noqa: ARG002
        """Render the exact current Plan without invoking or paraphrasing through a model."""
        value = current_plan(self.session)
        if value is None:
            if current_goal(self.session) is not None:
                message = "There is no current Plan. The current formal work is a Goal.\nUse /goal-show to display it."
                self.system_message(message, kind="info")
                return message
            self.system_message("no current Plan", kind="muted")
            return "No current Plan."
        self.mode["current_plan"] = value
        self.query_one(ChatLog).present_plan(value, active=True, status=value["status"])
        return "Current Plan rendered."

    def clear_plan(self) -> None:
        """Clear current_plan without deleting historical transcript events."""
        value = clear_current_plan(self.session)
        self.mode["current_plan"] = None
        if value is None:
            if current_goal(self.session) is not None:
                self.system_message("There is no current Plan. Use /goal-clear to clear the current Goal.", kind="info")
                return
            self.system_message("no current Plan", kind="muted")
            return
        self.query_one(ChatLog).resolve_plan(str(value["id"]), "cleared")
        self.store.save(self.session)
        self._sync_artifact_button()
        self.system_message(f'cleared current Plan "{plan_title(value)}"', kind="muted")

    async def resume_plan(self) -> None:
        """Resume any incomplete current Plan immediately in Act mode."""
        if not self._model_agents_available():
            return
        value = current_plan(self.session)
        if value is None:
            self.system_message("no current Plan", kind="muted")
            return
        if value["status"] == "completed":
            self.system_message(
                "the current Plan is completed; use /plan-show or ask MIRA to show/run it again",
                kind="info",
            )
            return
        if value["status"] not in RESUMABLE_PLAN_STATUSES:
            self.system_message("the current Plan is not resumable", kind="warning")
            return
        started = start_plan_attempt(self.session)
        if started is None:
            return
        self.mode["current_plan"] = started
        self.mode["executing_plan"] = True
        self.mode["planning"] = False
        self.store.save(self.session)
        self.query_one(ChatLog).resolve_plan(str(started["id"]), "active")
        self.system_message(f'resuming Plan "{plan_title(started)}"', kind="status")
        self.busy = True
        self._main_stream_active = False
        self._set_status(state="running")
        self.query_one(PromptBox).disabled = True
        self.turn_worker = self.run_worker(
            self._run_turn_for_plan(started),
            name="plan-resume",
            exclusive=True,
        )

    async def prepare_goal(self, interrupt: Any) -> str:
        """Generate shared Success Criteria before forced Goal finalisation."""
        self.waiting_finished()
        staging = self.mode.get("goal_staging")
        if not isinstance(staging, dict):
            raise RuntimeError("Goal staging context is unavailable")
        request = prepare_goal_request(interrupt)
        revision = self.mode.get("goal_revision")
        authoritative_request = str(staging.get("authoritative_objective") or "")
        objective = request["objective"] or authoritative_request
        if not objective:
            raise RuntimeError("prepare_goal requires the authoritative user objective")
        context = request["context_and_constraints"]
        evidence = request["research_evidence"]
        research_context = "\n\n".join(value for value in (context, evidence) if value)
        self.waiting_started("Generating success criteria", immediate=True, elapsed=True)
        service = SuccessCriteriaService(self.config or {})
        previous = revision.get("previous_goal") if isinstance(revision, dict) else None
        if isinstance(previous, dict):
            criteria = await service.revise(
                objective,
                str(previous.get("success_criteria") or ""),
                str(revision.get("feedback") or ""),
                research_context,
            )
        else:
            criteria = await service.generate(
                objective,
                research_context,
                authoritative_request=authoritative_request,
            )
        staging.update(
            {
                "objective": objective,
                "context_and_constraints": context,
                "research_evidence": evidence,
                "success_criteria": criteria,
                "stage": PLANNING_STAGE_GOAL_FINALIZE,
            }
        )
        self.mode["planning_stage"] = PLANNING_STAGE_GOAL_FINALIZE
        self.waiting_started("Finalizing Goal", immediate=True, elapsed=True)
        revision_context = ""
        if isinstance(revision, dict):
            revision_context = (
                f"\n\n<user_feedback>\n{str(revision.get('feedback') or '')}\n</user_feedback>"
            )
        return (
            f"{GOAL_FINALIZATION_POLICY}\n\n"
            f"<authoritative_request>\n{authoritative_request}\n</authoritative_request>\n\n"
            f"<objective>\n{objective}\n</objective>\n\n"
            f"<success_criteria>\n{criteria}\n</success_criteria>"
            f"{revision_context}"
        )

    async def finalize_goal(self, interrupt: Any) -> str:
        """Persist and render one criteria-only GoalArtifact."""
        self.waiting_finished()
        staging = self.mode.get("goal_staging")
        if not isinstance(staging, dict) or not staging.get("success_criteria"):
            raise RuntimeError("finalize_goal requires staged Objective and Success Criteria")
        artifact = goal_artifact(
            goal_id=self._next_goal_id(),
            title=goal_title_request(interrupt),
            objective=str(staging.get("objective") or ""),
            success_criteria=str(staging.get("success_criteria") or ""),
            rubric_enabled=bool(self.mode.get("rubric_enabled")),
            rubric_iterations=int(self.mode.get("rubric_max_iterations") or 3),
        )
        previous_plan = current_plan(self.session)
        previous_goal = current_goal(self.session)
        if previous_plan is not None:
            self.query_one(ChatLog).resolve_plan(str(previous_plan["id"]), "superseded")
        if previous_goal is not None:
            self.query_one(ChatLog).resolve_goal(str(previous_goal["id"]), "superseded")
        artifact = replace_current_goal(self.session, artifact)
        self.mode["current_plan"] = None
        self.mode["current_goal"] = artifact
        self.mode["goal_staging"] = None
        self.mode["goal_revision"] = None
        self.mode["planning_stage"] = PLANNING_STAGE_PLAN_RESEARCH
        self._sync_artifact_button()
        event = append_event(
            self.session,
            {"type": "goal", "mode": "planning" if self.mode.get("planning") else "action", "goal": artifact, "status": "proposed"},
        )
        self.store.save(self.session)
        self.query_one(ChatLog).present_goal(
            artifact,
            active=True,
            status="proposed",
            created_at=str(event.get("created_at") or ""),
        )
        return "Goal and Success Criteria presented for user review."

    async def show_goal(self, interrupt: Any = None) -> str:  # noqa: ARG002
        """Render the exact current Goal without a model call."""
        value = current_goal(self.session)
        if value is None:
            if current_plan(self.session) is not None:
                message = "There is no current Goal. The current formal work is a Plan.\nUse /plan-show to display it."
                self.system_message(message, kind="info")
                return message
            self.system_message("no current Goal", kind="muted")
            return "No current Goal."
        self.mode["current_goal"] = value
        self.query_one(ChatLog).present_goal(value, active=True, status=value["status"])
        return "Current Goal rendered."

    def clear_goal(self) -> None:
        """Clear current_goal without deleting historical transcript events."""
        value = clear_current_goal(self.session)
        self.mode["current_goal"] = None
        if value is None:
            if current_plan(self.session) is not None:
                self.system_message("There is no current Goal. Use /plan-clear to clear the current Plan.", kind="info")
                return
            self.system_message("no current Goal", kind="muted")
            return
        self.query_one(ChatLog).resolve_goal(str(value["id"]), "cleared")
        self.store.save(self.session)
        self._sync_artifact_button()
        self.system_message(f'cleared current Goal "{goal_title(value)}"', kind="muted")

    async def resume_goal(self) -> None:
        """Resume any incomplete current Goal immediately in Act mode."""
        if not self._model_agents_available():
            return
        value = current_goal(self.session)
        if value is None:
            if current_plan(self.session) is not None:
                self.system_message("There is no current Goal. Use /plan-resume to resume the current Plan.", kind="info")
                return
            self.system_message("no current Goal", kind="muted")
            return
        if value["status"] == "completed":
            self.system_message(
                "This Goal is already completed.\nUse /goal-show to reopen it and select Implement to run it again.",
                kind="info",
            )
            return
        if value["status"] not in RESUMABLE_GOAL_STATUSES:
            self.system_message("the current Goal is not resumable", kind="warning")
            return
        started = start_goal_attempt(self.session)
        if started is None:
            return
        self.mode["current_goal"] = started
        self.mode["executing_goal"] = True
        self.mode["planning"] = False
        self.store.save(self.session)
        self.query_one(ChatLog).resolve_goal(str(started["id"]), "active")
        self.system_message(f'resuming Goal "{goal_title(started)}"', kind="status")
        self.busy = True
        self._main_stream_active = False
        self._set_status(state="running")
        self.query_one(PromptBox).disabled = True
        self.turn_worker = self.run_worker(
            self._run_turn_for_goal(started),
            name="goal-resume",
            exclusive=True,
        )

    async def _confirm_formal_replacement(self, new_kind: str) -> bool:
        """Confirm replacement of one incomplete current Plan or Goal."""
        plan = current_plan(self.session)
        goal = current_goal(self.session)
        current_kind = "Plan" if plan is not None else "Goal" if goal is not None else ""
        current = plan or goal
        if current is None or current.get("status") == "completed":
            return True
        answer = await self._prompt_choice(
            f"Replace Current {current_kind}?",
            (
                f"A current {current_kind} is still incomplete.\n\n"
                f"If the new {new_kind} is successfully created, it will replace "
                f"the current {current_kind}."
            ),
            [
                ("replace", f"Replace Current {current_kind}"),
                ("keep", f"Keep Current {current_kind}"),
            ],
            vertical=True,
        )
        return answer == "replace"

    def compaction_started(self) -> None:
        """Show that DeepAgents is compacting conversation context."""
        self.trace.compaction_started()
        self._main_stream_active = False
        self.waiting_finished()
        notice = pop_context_overflow_notice()
        if notice:
            self.query_one(ChatLog).system_message(notice, kind="info")
        self.query_one(ChatLog).compaction_started()
        self._set_status(state="running")

    def compaction_finished(
        self,
        message: str = "context compacted",
        *,
        success: bool = True,
        resume_waiting: bool = True,
    ) -> None:
        """Show that DeepAgents has finished compacting context."""
        self.trace.compaction_finished()
        self.query_one(ChatLog).compaction_finished(message, success=success)
        self._set_status(state="running")
        if resume_waiting:
            self._rearm_waiting_if_busy()

    def clear_log(self) -> None:
        """Clear chat output."""
        self.query_one(ChatLog).clear_log()
        self.query_one(SubagentsPanel).reset()

    async def _run_history_command(self, text: str) -> None:
        """Run a destructive history command outside the submit event handler."""
        try:
            await self._handle_history_command(text)
            self._set_ready_unless_error()
        except Exception as exc:
            self.system_message(f"clear history error: {exc}", kind="error")
            self._set_status(state="error")
        finally:
            self.action_focus_prompt()

    async def _handle_history_command(self, text: str) -> bool:
        """Handle destructive history slash commands with confirmation."""
        if text not in DESTRUCTIVE_HISTORY_COMMANDS:
            return False
        if self.busy:
            self.system_message("finish the current turn before clearing history", kind="warning")
            return True

        if text == "/clear-chat":
            answer = await self._prompt_choice(
                "Clear Current Chat?",
                "Clear the saved transcript for this chat? Older chats and prompt history will be kept.\n\n"
                + DESTRUCTIVE_CONFIRM_HINT,
                DESTRUCTIVE_CONFIRM_CHOICES,
            )
            if answer != "o":
                self.system_message("clear chat cancelled", kind="muted")
                return True
            self._clear_current_chat()
            self.system_message("current chat history cleared", kind="info")
            return True

        if text == "/clear-all-chats":
            answer = await self._prompt_choice(
                "Clear All Chats?",
                "Delete all saved chat sessions and compaction archives for this workspace? Prompt history is kept.\n\n"
                + DESTRUCTIVE_CONFIRM_HINT,
                DESTRUCTIVE_CONFIRM_CHOICES,
            )
            if answer != "o":
                self.system_message("clear all chats cancelled", kind="muted")
                return True
            sessions, compactions = self._clear_all_chats()
            session_suffix = "s" if sessions != 1 else ""
            compaction_suffix = "s" if compactions != 1 else ""
            self.system_message(
                f"cleared {sessions} saved chat session{session_suffix} and "
                f"{compactions} compaction file{compaction_suffix}",
                kind="info",
            )
            return True

        if text == "/clear-errors":
            answer = await self._prompt_choice(
                "Clear Error Reports?",
                "Delete saved error reports under .mira/_errors for this workspace? "
                "Saved chats and prompt history will be kept.\n\n"
                + DESTRUCTIVE_CONFIRM_HINT,
                DESTRUCTIVE_CONFIRM_CHOICES,
            )
            if answer != "o":
                self.system_message("clear error reports cancelled", kind="muted")
                return True
            reports = clear_error_reports(self.workspace)
            suffix = "" if reports == 1 else "s"
            self.system_message(f"cleared {reports} error report file{suffix}", kind="info")
            return True

        if text == "/clear-prompts":
            answer = await self._prompt_choice(
                "Clear Prompt History?",
                "Clear prompt up/down history from .mira/history.txt? Saved chat sessions will be kept.\n\n"
                + DESTRUCTIVE_CONFIRM_HINT,
                DESTRUCTIVE_CONFIRM_CHOICES,
            )
            if answer != "o":
                self.system_message("clear prompt history cancelled", kind="muted")
                return True
            self._clear_prompt_history()
            self.system_message("prompt history cleared", kind="info")
            return True

        return False

    async def _run_settings_command(self, initial_tab: str = "general") -> None:
        """Run the interactive settings menu."""
        try:
            mounted = self._handle_settings_command(initial_tab)
            if mounted is not True:
                await mounted
            self._set_ready_unless_error()
        except Exception as exc:
            self.system_message(f"settings error: {exc}", kind="error")
            self._set_status(state="error")

    async def _run_reload_command(self) -> bool:
        """Reload configuration/resources and rebuild agents without restarting MCP."""
        return await self._run_reload(full_runtime=False)

    async def _run_reload_runtime_command(self) -> bool:
        """Reload the full runtime, including MCP connections."""
        return await self._run_reload(full_runtime=True)

    async def _run_reload(self, *, full_runtime: bool) -> bool:
        """Run one reload scope with shared status and error handling."""
        if self.query_one(PromptPanel).active:
            if isinstance(self.screen, MCPPanelScreen):
                self.screen.dismiss()
            self.system_message("answer the current prompt before reloading", kind="warning")
            return False

        if self._reload_in_progress:
            self.system_message("reload already in progress", kind="warning")
            return False

        if self.busy:
            await self._handle_reload_command(full_runtime=full_runtime)
            return False

        prompt = self.query_one(PromptBox)
        prompt_was_disabled = prompt.disabled
        self._reload_in_progress = True
        prompt.disabled = True
        self._set_status(state="running")
        try:
            reloaded = await self._handle_reload_command(full_runtime=full_runtime)
            if reloaded:
                self._set_status(state="ready")
            return reloaded
        except Exception as exc:
            from config.llm import ConfigError

            if isinstance(exc, ConfigError):
                self.system_message(str(exc), kind="warning")
            else:
                error_path = self._write_error_report(exc, source="tui.reload")
                self.system_message(f"reload error: {exc}\nerror report: {error_path}", kind="error")
            self._set_status(state="error")
            return False
        finally:
            self._reload_in_progress = False
            if self.is_mounted:
                prompt.disabled = prompt_was_disabled
                if not prompt.disabled:
                    self.action_focus_prompt()

    async def _handle_reload_command(self, *, full_runtime: bool = False) -> bool:
        """Reload config, resources, and agents, optionally restarting MCP."""
        if self.busy:
            self.system_message("finish the current turn before reloading agents", kind="warning")
            return True

        if full_runtime:
            await self._reload_runtime()
        else:
            await self._reload_agents()
        self._notify_explicit_reload()
        self.system_message("runtime reloaded" if full_runtime else "agent reloaded", kind="info")
        return True

    async def _reload_agents(self) -> RuntimeSnapshot:
        """Reload configuration/resources and agents while preserving live MCP."""
        return await self._reload_runtime_state(reload_mcp=False)

    async def _reload_runtime(self) -> RuntimeSnapshot:
        """Reload configuration/resources, MCP, visible state, and agents."""
        return await self._reload_runtime_state(reload_mcp=True)

    async def _reload_runtime_state(self, *, reload_mcp: bool) -> RuntimeSnapshot:
        """Build and install refreshed runtime state for the requested scope."""
        from agent.llm import active_model_issues, get_llm, get_model_name, model_unavailable_message
        from agent.resources import build_resources
        from agent.resources.subagents import subagent_model_issues
        from config.metadata import ModelMetadata, infer_model_metadata
        from config.runtime import load_effective_config

        config = load_effective_config(
            self.workspace,
            self.launch_options,
            override_dotenv=True,
        )
        if self.mcp_manager is not None:
            if reload_mcp:
                await self.mcp_manager.reload()
            else:
                self.mcp_manager.prompt_registry.reload_local()
        resources = build_resources(
            self.workspace,
            settings=config.get("settings"),
            config=None,
        )
        if resources.subagent_discovery.complete and config.get("settings_valid", True):
            from config.settings import prune_subagent_settings, save_settings

            current_settings = config.get("settings") or {}
            pruned = prune_subagent_settings(
                current_settings,
                {item.name for item in resources.subagent_discovery.items},
            )
            if pruned != current_settings and save_settings(self.workspace, pruned):
                config["settings"] = pruned
        assignment_issues = [
            *active_model_issues(config),
            *subagent_model_issues(resources.subagent_discovery, config),
        ]
        issues = [
            *(config.get("issues") or []),
            *(self.mcp_manager.issues if self.mcp_manager is not None else []),
            *(self.mcp_manager.prompt_registry.issues if self.mcp_manager is not None else []),
            *resources.issues,
            *assignment_issues,
        ]
        metadata = ModelMetadata(
            context_tokens=int((config.get("settings") or {}).get("models", {}).get("context_limit_tokens", 32768)),
            context_source="settings.models.context_limit_tokens",
        )
        agent = None
        plan_agent = None
        if config.get("settings_valid", True) and not assignment_issues:
            inspect_model = get_llm(config, metadata=ModelMetadata())
            metadata = await infer_model_metadata(config, model=inspect_model)
            from agent.resources import configure_subagents

            action_resources = configure_subagents(resources, config)
            agent, plan_agent = self._build_agent_pair(
                config=config,
                metadata=metadata,
                action_resources=action_resources,
            )
        model_name = get_model_name(config)
        mode_updates = self._agent_mode_updates(agent, plan_agent, config)
        mode_updates["resources"] = resources.metadata
        snapshot = build_runtime_snapshot(
            config,
            self.launch_options,
            model_name=model_name,
        )

        self.config = config
        if self._settings_panel is not None and self._settings_panel.is_mounted:
            self._settings_panel.refresh_tracing_registry(config["tracing_registry"])
        self.model_name = model_name
        self.context_limit_tokens = metadata.context_tokens
        self.context_limit_source = metadata.context_source
        self.agent = agent
        self.plan_agent = plan_agent
        self.agent_unavailable_message = model_unavailable_message(config) if agent is None else ""
        self.query_one(AutocompleteInput).set_project_backend(
            getattr(agent, "mira_project_backend", None) or resources.project_backend
        )
        self.tool_failures = resources.tool_failures
        self.issues = issues
        self.mode.update(mode_updates)
        self.runtime_snapshot = snapshot
        ensure_dashboard(
            self.session,
            model_name=self.model_name,
            context_limit_tokens=self.context_limit_tokens,
            context_limit_source=self.context_limit_source,
        )
        if self.store is not None:
            self.store.save(self.session)
        self._refresh_startup_splash()
        self._set_status(state=self.status_state)
        self._sync_issues()
        return snapshot

    def _render_runtime_snapshot(
        self,
        snapshot: RuntimeSnapshot | None = None,
    ) -> None:
        """Render the current sanitized runtime and connection state."""
        snapshot = snapshot or self.runtime_snapshot
        if snapshot is None:
            self.system_message("runtime state is not available", kind="warning")
            return
        self.command_output(runtime_report(snapshot))

    def _refresh_startup_splash(self) -> None:
        """Refresh the visible startup metadata block after runtime changes."""
        if not self.is_mounted:
            return
        self.query_one(ChatLog).startup(
            model_name=self.model_name,
            session_id=self.session["id"],
            workspace=str(self.session["workspace"]),
        )

    def _handle_settings_command(self, initial_tab: str = "general") -> Any:
        """Mount the interactive settings panel."""
        from config.tracing import TracingRegistry, load_tracing_registry

        if self.busy:
            self.system_message("finish the current turn before changing settings", kind="warning")
            return True

        settings = load_settings(self.workspace)
        if self._settings_panel is not None and self._settings_panel.is_mounted:
            self._settings_panel.remove()
        tracing_registry = (self.config or {}).get("tracing_registry")
        if not isinstance(tracing_registry, TracingRegistry):
            tracing_registry = load_tracing_registry(self.workspace)
        panel = SettingsPanel(
            settings,
            tool_metadata=self._settings_tool_metadata(),
            apply_change=self._apply_settings,
            reload_runtime=self._run_reload_runtime_command,
            close_panel=self._close_settings_panel,
            mcp_manager=self.mcp_manager,
            model_registry=(self.config or {}).get("model_registry"),
            tracing_registry=tracing_registry,
            subagent_metadata=((self.mode.get("resources") or {}).get("subagents") or []),
            initial_tab=initial_tab,
        )
        self._settings_panel = panel
        return self.mount(panel)

    async def _apply_settings(self, settings: dict[str, Any]) -> tuple[bool, str]:
        """Persist settings and apply any needed runtime changes."""
        old_settings = (self.config or {}).get("settings") or load_settings(self.workspace)
        if old_settings == settings:
            return True, "settings unchanged"
        if await self._execute_enable_cancelled(old_settings, settings):
            return False, "execute remains disabled"
        old_git_enabled = git_protection_enabled(old_settings)
        new_git_enabled = git_protection_enabled(settings)
        if not save_settings(self.workspace, settings):
            return False, "could not save .mira/settings.yml"

        old_config = self.config
        self.config = dict(old_config or {})
        self.config["settings"] = settings
        if not self.ready:
            return True, "settings saved; MIRA is not ready; Reload Runtime required"
        if new_git_enabled != old_git_enabled:
            if new_git_enabled:
                return await self._ensure_git_after_enabling()
            return True, "git protection disabled; no reload required"
        if old_settings.get("tracing") != settings.get("tracing"):
            return True, "tracing settings saved; Reload Runtime required"

        try:
            if old_settings.get("models") != settings.get("models"):
                await self._rebuild_model_agents(old_settings)
            else:
                await self._rebuild_agents()
        except Exception as error:
            from config.llm import ConfigError

            restored = save_settings(self.workspace, old_settings)
            self.config = old_config
            if isinstance(error, ConfigError):
                message = f"settings not saved: {error}"
            else:
                error_path = self._write_error_report(error, source="tui.settings")
                message = f"settings not saved; agent rebuild failed: {error}; error report: {error_path}"
            if not restored:
                message += "; could not restore .mira/settings.yml"
            return False, message
        self._set_status(state="ready")
        return True, "settings saved; agents rebuilt; no reload required"

    async def _execute_enable_cancelled(self, old_settings: dict[str, Any], new_settings: dict[str, Any]) -> bool:
        """Confirm before switching the agent to LocalShellBackend."""
        if tool_enabled(old_settings, EXECUTE_TOOL) or not tool_enabled(new_settings, EXECUTE_TOOL):
            return False

        panel = self._settings_panel
        if panel is not None and panel.is_mounted:
            panel.display = False
        try:
            answer = await self._prompt_choice(
                "Enable Execute?",
                "Enabling execute switches MIRA to LocalShellBackend.\n"
                "The agent can run shell commands directly on this machine with your user permissions.\n"
                "Shell commands are not sandboxed and can access paths outside the workspace.\n"
                "MIRA passes only a small OS shell environment allowlist, not your full environment or API keys.\n\n"
                "Continue?",
                [("y", "Enable (y)"), ("n", "Cancel (n)")],
            )
        finally:
            if panel is not None and panel.is_mounted:
                panel.display = True
                panel.focus()
        return answer != "y"

    async def _ensure_git_after_enabling(self) -> tuple[bool, str]:
        """Initialize Git if protection was enabled for an unprotected workspace."""
        from cli.git_guard import init_git_repository, is_git_worktree

        if is_git_worktree(self.workspace):
            return True, "git protection enabled; no reload required"
        if init_git_repository(self.workspace):
            return True, "git protection enabled; repository initialized; no reload required"
        return True, "git protection enabled; Git was not initialized; no reload required"

    def _settings_tool_metadata(self) -> list[dict[str, str]]:
        """Return loaded tool metadata for the settings panel."""
        resources = self.mode.get("resources") if isinstance(self.mode, dict) else None
        tools = resources.get("tools") if isinstance(resources, dict) else None
        return tools if isinstance(tools, list) else []

    def _close_settings_panel(self) -> None:
        """Forget the closed settings panel and return focus to the prompt."""
        self._settings_panel = None
        self.action_focus_prompt()

    def _clear_current_chat(self) -> None:
        """Reset the active persisted transcript while keeping the same session id."""
        self.session["title"] = "Untitled session"
        self.session["turns"] = 0
        self.session["events"] = []
        self.session["dashboard"] = normalize_dashboard(None)
        ensure_dashboard(
            self.session,
            model_name=self.model_name,
            context_limit_tokens=self.context_limit_tokens,
            context_limit_source=self.context_limit_source,
        )
        if self.store is not None:
            self.store.save(self.session)
        self._render_current_session()

    def _clear_all_chats(self) -> tuple[int, int]:
        """Delete saved session files and keep the active session usable."""
        sessions = 0
        compactions = 0
        if self.store is not None:
            clear_all = getattr(self.store, "clear_all", None)
            if callable(clear_all):
                sessions = int(clear_all())
            clear_compactions = getattr(self.store, "clear_compactions", None)
            if callable(clear_compactions):
                compactions = int(clear_compactions())
        self._clear_current_chat()
        return sessions, compactions

    def _clear_prompt_history(self) -> None:
        """Clear prompt history on disk and in memory."""
        try:
            if self.history_path.exists():
                self.history_path.write_text("", encoding="utf-8")
            else:
                self.history_path.parent.mkdir(parents=True, exist_ok=True)
        finally:
            self.query_one(PromptBox).set_history([])

    def _render_current_session(self) -> None:
        """Rebuild visible chat output from the active session."""
        chat = self.query_one(ChatLog)
        self.query_one(SubagentsPanel).reset()
        chat.clear_log()
        chat.startup(
            model_name=self.model_name,
            session_id=self.session["id"],
            workspace=str(self.session["workspace"]),
        )
        chat.restore_session(self.session)
        self._refresh_sessions()

    def reasoning_delta(self, delta: str, *, created_at: str = "") -> None:
        """Render streamed reasoning text."""
        self.trace.reasoning_delta(delta)
        self.waiting_finished()
        self._mark_main_stream_active()
        self.query_one(ChatLog).reasoning_delta(delta, created_at=created_at)

    def discard_reasoning(self) -> None:
        """Remove streamed reasoning that was later classified as internal."""
        self.trace.discard_reasoning()
        self.query_one(ChatLog).discard_reasoning()

    def correction(self, event: dict[str, Any], *, created_at: str = "") -> None:
        """Render a deterministic correction in trace and TUI history."""
        self.trace.correction(event)
        self.waiting_finished()
        self.query_one(ChatLog).correction(event, created_at=created_at)

    def text_delta(self, delta: str, *, created_at: str = "") -> None:
        """Render streamed assistant text."""
        self.trace.assistant_delta(delta)
        self.waiting_finished()
        self._mark_main_stream_active()
        self.query_one(ChatLog).text_delta(delta, created_at=created_at)

    def model_stream_finished(self) -> None:
        """Re-arm waiting UI after streamed model text/reasoning goes quiet."""
        self.query_one(ChatLog).finish_stream_phase()
        self._finish_main_stream_activity()
        self._rearm_waiting_if_busy()

    def tool_call_delta(self, name: str, args: Any, call_id: str = "") -> None:
        """Render a live draft of streamed tool-call input."""
        self.waiting_finished()
        self._mark_main_stream_active()
        self.query_one(ChatLog).tool_call_delta(name, args, call_id=call_id)
        self._set_status(state="running")

    def delegation_delta(self, calls: list[dict[str, Any]]) -> None:
        """Render a live draft of streamed task delegation input."""
        self.waiting_finished()
        self._mark_main_stream_active()
        if self._subagent_panel_is_live():
            self._set_status(state="running")
            return
        self.query_one(ChatLog).delegation_delta(calls)
        self._set_status(state="running")

    def tool_call(self, name: str, args: Any, call_id: str = "", *, created_at: str = "") -> None:
        """Render a tool call in transcript order."""
        self.trace.tool_call(name, args)
        self._finish_main_stream_activity()
        self.waiting_finished()
        self.query_one(ChatLog).tool_call(name, args, call_id=call_id, created_at=created_at)
        self._rearm_waiting_if_busy()

    def tool_call_updated(self, name: str, args: Any, call_id: str = "", *, created_at: str = "") -> None:
        """Update an edited call without creating another transcript block."""
        self.trace.tool_call_updated(name, args)
        self.query_one(ChatLog).tool_call_updated(
            name,
            args,
            call_id=call_id,
            created_at=created_at,
        )

    def tool_call_approval_resolved(self, name: str, call_id: str = "") -> None:
        """Advance idless approval matching without changing visible content."""
        self.query_one(ChatLog).tool_call_approval_resolved(name, call_id=call_id)

    def tool_result(
        self,
        name: str,
        result: str,
        call_id: str = "",
        *,
        created_at: str = "",
        duration_ms: int | None = None,
    ) -> None:
        """Render a tool result in transcript order."""
        self.trace.tool_result(name, result)
        self._finish_main_stream_activity()
        self.waiting_finished()
        self.query_one(ChatLog).tool_result(
            name,
            result,
            call_id=call_id,
            created_at=created_at,
            duration_ms=duration_ms,
        )
        self._rearm_waiting_if_busy()

    def completed_tool_result(
        self,
        name: str,
        result: str,
        call_id: str = "",
        *,
        created_at: str = "",
        duration_ms: int | None = None,
    ) -> None:
        """Update a finished ordinary tool without closing active model output."""
        self.trace.completed_tool_result(name, result)
        self._finish_eval_group(name, call_id, failed=False)
        self.query_one(ChatLog).completed_tool_result(
            name,
            result,
            call_id=call_id,
            created_at=created_at,
            duration_ms=duration_ms,
        )
        self._rearm_waiting_if_busy()

    def completed_tool_error(
        self,
        name: str,
        error: str,
        call_id: str = "",
        *,
        created_at: str = "",
        duration_ms: int | None = None,
    ) -> None:
        """Update a failed ordinary tool without closing active model output."""
        self.trace.completed_tool_error(name, error)
        self._finish_eval_group(name, call_id, failed=True)
        self.query_one(ChatLog).completed_tool_error(
            name,
            error,
            call_id=call_id,
            created_at=created_at,
            duration_ms=duration_ms,
        )
        self._rearm_waiting_if_busy()

    def tool_call_stopped(
        self,
        name: str,
        call_id: str = "",
        *,
        status: str = "cancelled",
        duration_ms: int | None = None,
    ) -> None:
        """Freeze an invoked tool that ended with its parent turn."""
        self.query_one(ChatLog).tool_call_stopped(
            name,
            call_id=call_id,
            status=status,
            duration_ms=duration_ms,
        )

    def recovered_tool_result(
        self,
        name: str,
        result: str,
        call_id: str = "",
        *,
        created_at: str = "",
        duration_ms: int | None = None,
    ) -> None:
        """Render a late-discovered tool result in session transcript order."""
        self.trace.recovered_tool_result(name, result)
        self._finish_main_stream_activity()
        self.waiting_finished()
        self._finish_eval_group(name, call_id, failed=False)
        self.query_one(ChatLog).tool_result(
            name,
            result,
            call_id=call_id,
            created_at=created_at,
            duration_ms=duration_ms,
        )
        self._rearm_waiting_if_busy()

    def recovered_tool_error(
        self,
        name: str,
        error: str,
        call_id: str = "",
        *,
        created_at: str = "",
        duration_ms: int | None = None,
    ) -> None:
        """Render a late-discovered failed tool result in session transcript order."""
        self.trace.recovered_tool_error(name, error)
        self._finish_main_stream_activity()
        self.waiting_finished()
        self._finish_eval_group(name, call_id, failed=True)
        self.query_one(ChatLog).tool_error(
            name,
            error,
            call_id=call_id,
            created_at=created_at,
            duration_ms=duration_ms,
        )
        self._rearm_waiting_if_busy()

    def _finish_eval_group(self, name: str, call_id: str, *, failed: bool) -> None:
        """Reconcile eval subagent rows from the parent tool lifecycle."""
        if name == "eval" and call_id:
            self.query_one(SubagentsPanel).finish_eval_group(call_id, failed=failed)

    def delegation_started(self, calls: list[dict[str, Any]], *, created_at: str = "") -> None:
        """Render task delegation summary."""
        self.trace.delegation_started(calls)
        self._finish_main_stream_activity()
        self.waiting_finished()
        if self._subagent_panel_is_live():
            self._set_status(state="running")
            self._rearm_waiting_if_busy()
            return
        self.query_one(ChatLog).delegation_started(calls, created_at=created_at)
        self._rearm_waiting_if_busy()

    def start_subagent_live(self) -> None:
        """Prepare subagent display."""
        self._subagent_live_active = True
        self.query_one(ChatLog).start_subagent_live()

    def _subagent_panel_is_live(self) -> bool:
        """Return whether live subagent work should own task delegation display."""
        try:
            panel = self.query_one(SubagentsPanel)
        except NoMatches:
            return self._subagent_live_active
        return self._subagent_live_active or panel.has_running_subagents()

    def stop_subagent_live(self) -> None:
        """Finalize subagent display."""
        self.query_one(ChatLog).stop_subagent_live()
        self._subagent_live_active = False

    def subagents_cancelled(self) -> None:
        """Mark active subagent display as cancelled."""
        self.query_one(ChatLog).subagents_cancelled()
        self.query_one(SubagentsPanel).cancel_running()
        self._subagent_live_active = False

    def tick_subagents(self) -> None:
        """Advance subagent status animation."""
        self.query_one(ChatLog).tick_subagents()
        self.query_one(SubagentsPanel).tick()

    def subagent_label(self, subagent: Any) -> str:
        """Return a stable display label for a subagent."""
        return self.query_one(ChatLog).subagent_label(subagent)

    def subagent_started(
        self,
        subagent: str,
        task_input: str = "",
        *,
        origin: str = "",
        eval_id: str = "",
        row_id: str = "",
        model: str = "",
        created_at: str = "",
    ) -> None:
        """Render a subagent start."""
        self.trace.subagent_started(subagent, task_input)
        self._finish_main_stream_activity()
        self.waiting_finished()
        self.query_one(SubagentsPanel).start_subagent(
            subagent,
            task_input,
            row_id=row_id,
            eval_id=eval_id,
        )
        if not self._subagent_live_active:
            self.query_one(ChatLog).subagent_started(subagent, task_input, origin=origin, created_at=created_at)
        self._rearm_waiting_if_busy()

    def subagent_request_updated(self, subagent: str, task_input: str) -> None:
        """Fill in a subagent request that arrived after the block started."""
        self.waiting_finished()
        self.query_one(SubagentsPanel).update_subagent_request(subagent, task_input)
        if not self._subagent_live_active:
            self.query_one(ChatLog).subagent_request_updated(subagent, task_input)

    def subagent_finished(
        self,
        subagent: str,
        result: str = "",
        *,
        eval_id: str = "",
        row_id: str = "",
        duration_ms: int | None = None,
        created_at: str = "",
    ) -> None:
        """Render a subagent finish."""
        self.trace.subagent_finished(subagent, result)
        self._finish_main_stream_activity()
        self.waiting_finished()
        self.query_one(SubagentsPanel).finish_subagent(
            subagent,
            result,
            row_id=row_id,
            eval_id=eval_id,
            duration_ms=duration_ms,
        )
        if not self._subagent_live_active:
            self.query_one(ChatLog).subagent_finished(subagent, result, created_at=created_at)
        self._rearm_waiting_if_busy()

    def subagent_cancelled(
        self,
        subagent: str,
        result: str = "",
        *,
        eval_id: str = "",
        row_id: str = "",
        duration_ms: int | None = None,
        created_at: str = "",
    ) -> None:
        """Render a subagent cancellation."""
        self.trace.subagent_cancelled(subagent, result)
        self._finish_main_stream_activity()
        self.waiting_finished()
        self.query_one(SubagentsPanel).finish_subagent(
            subagent,
            result,
            row_id=row_id,
            eval_id=eval_id,
            duration_ms=duration_ms,
            status="CANCELLED",
        )
        if not self._subagent_live_active:
            self.query_one(ChatLog).subagent_cancelled(subagent, result, created_at=created_at)

    def eval_subagent_started(
        self,
        subagent: str,
        task_input: str = "",
        *,
        eval_id: str = "",
        row_id: str = "",
        model: str = "",
        label: str = "",
    ) -> None:
        """Render an eval-created subagent only in the live panel."""
        self._finish_main_stream_activity()
        self.waiting_finished()
        self.query_one(SubagentsPanel).start_subagent(
            subagent,
            task_input,
            row_id=row_id,
            eval_id=eval_id,
            label=label,
        )

    def eval_subagent_finished(
        self,
        subagent: str,
        result: str = "",
        *,
        eval_id: str = "",
        row_id: str = "",
        duration_ms: int | None = None,
    ) -> None:
        """Mark an eval-created subagent done only in the live panel."""
        self._finish_main_stream_activity()
        self.waiting_finished()
        self.query_one(SubagentsPanel).finish_subagent(
            subagent,
            result,
            row_id=row_id,
            eval_id=eval_id,
            duration_ms=duration_ms,
        )

    def eval_subagent_cancelled(
        self,
        subagent: str,
        result: str = "",
        *,
        eval_id: str = "",
        row_id: str = "",
        duration_ms: int | None = None,
    ) -> None:
        """Mark an eval-created subagent error/cancelled only in the live panel."""
        self._finish_main_stream_activity()
        self.waiting_finished()
        self.query_one(SubagentsPanel).finish_subagent(
            subagent,
            result,
            row_id=row_id,
            eval_id=eval_id,
            duration_ms=duration_ms,
            status="ERROR" if result else "CANCELLED",
        )

    def rubric_evaluation_started(
        self,
        run_id: str,
        pass_number: int,
        max_iterations: int,
        *,
        grader_model: str = "",
    ) -> None:
        """Show live rubric activity in the TUI and trace sidecar."""
        self._finish_main_stream_activity()
        self.waiting_finished()
        self.trace.rubric_evaluation_started(
            run_id,
            pass_number,
            max_iterations,
            grader_model=grader_model,
        )
        self.query_one(ChatLog).rubric_evaluation_started(
            run_id,
            pass_number,
            max_iterations,
            grader_model=grader_model,
        )

    def rubric_evaluation_finished(
        self,
        evaluation: dict[str, Any],
        max_iterations: int,
        *,
        created_at: str = "",
    ) -> None:
        """Render a completed rubric evaluation."""
        self.trace.rubric_evaluation_finished(evaluation, max_iterations)
        self.query_one(ChatLog).rubric_evaluation_finished(
            evaluation,
            max_iterations,
            created_at=created_at,
        )

    def rubric_evaluations_cancelled(self) -> None:
        """Stop transient rubric activity after an interrupted invocation."""
        self.query_one(ChatLog).rubric_evaluations_cancelled()

    def rubric_evaluation_status(
        self,
        run_id: str,
        pass_number: int,
        status: str,
        max_iterations: int,
    ) -> None:
        """Reconcile a rubric result with the completed agent state."""
        self.trace.rubric_evaluation_status(run_id, pass_number, status, max_iterations)
        self.query_one(ChatLog).rubric_evaluation_status(
            run_id,
            pass_number,
            status,
            max_iterations,
        )

    def finish_main(self) -> None:
        """Close streamed chat blocks after a top-level turn."""
        self.trace.flush_all()
        self._finish_main_stream_activity()
        self.waiting_finished()
        self.query_one(ChatLog).finish_main()

    def finish_turn(self, *, cancelled: bool = False, tool_status: str = "cancelled") -> None:
        """Close live turn widgets without clearing visible transcript history."""
        self.trace.flush_all()
        self._finish_main_stream_activity()
        self.waiting_finished()
        self._waiting_label = "working..."
        self.query_one(ChatLog).finish_turn(cancelled=cancelled, tool_status=tool_status)
        if cancelled:
            self.query_one(SubagentsPanel).cancel_running()
            self._subagent_live_active = False

    def usage_updated(self) -> None:
        """Refresh the status bar after token usage is committed."""
        self._set_status(state=self.status_state)

    async def ask_approvals(self, interrupts: list[Any]) -> list[dict[str, Any]]:
        """Ask the user to approve, edit, or reject interrupted actions."""
        self.waiting_finished()
        decisions: list[dict[str, Any]] = []
        for interrupt in interrupts:
            for index, action in enumerate(action_requests(interrupt)):
                answer = await self._prompt_choice(
                    action_title(action),
                    action_preview(action),
                    action_choices(interrupt, action, index),
                )
                if answer == "e":
                    decisions.append(await self.edit_decision(action))
                elif answer == "r":
                    decisions.append({"type": "reject"})
                else:
                    decisions.append({"type": "approve"})
        return decisions

    async def ask_user(self, interrupt: Any) -> str:
        """Ask the user for a concrete next-step choice from an ask_user interrupt."""
        self.waiting_finished()
        request = ask_user_request(interrupt)
        question = ask_user_question(request)
        options = ask_user_options(request)
        choices = [(str(index), f"{index} {option}") for index, option in enumerate(options, start=1)]
        answer = str(await self._prompt_choice("Question", question, choices, vertical=True) or "")
        selected = options[int(answer) - 1] if answer.isdigit() and 0 < int(answer) <= len(options) else options[-1]
        if selected != ASK_USER_OPEN_OPTION:
            result = selected
        else:
            response = await self._prompt_text("Question", ASK_USER_OPEN_OPTION)
            result = (response or "").strip() or ASK_USER_OPEN_OPTION
        return result

    async def ask_create_git_repo(self, message: str) -> bool:
        """Ask whether MIRA should initialize Git for the workspace."""
        answer = await self._prompt_choice("Git", message, [("y", "Yes (y)"), ("n", "No (n)")])
        return answer == "y"

    async def ask_continue_without_git(self, message: str) -> bool:
        """Ask whether startup should continue without Git protection."""
        answer = await self._prompt_choice("Git", message, [("c", "Continue (c)"), ("e", "Exit (e)")])
        return answer == "c"

    async def edit_decision(self, action: Any) -> dict[str, Any]:
        """Prompt for edited JSON args and return a LangGraph decision."""
        if not isinstance(action, dict):
            return {"type": "reject"}

        edited_text = await self._prompt_json("Edited Args", json.dumps(action.get("args", {}), indent=2))
        if edited_text is None:
            return {"type": "reject"}

        try:
            edited_args = json.loads(edited_text)
        except json.JSONDecodeError:
            self.system_message("invalid JSON; rejecting action", kind="warning")
            return {"type": "reject"}

        if not isinstance(edited_args, dict):
            self.system_message("edited args must be a JSON object; rejecting action", kind="warning")
            return {"type": "reject"}

        return {
            "type": "edit",
            "edited_action": {
                "name": action.get("name", "tool"),
                "args": edited_args,
            },
        }

    async def _prompt_choice(
        self,
        title: str,
        message: str,
        choices: list[tuple[str, str]],
        *,
        vertical: bool = False,
    ) -> str | None:
        """Show a choice prompt in the main window."""
        return await self._with_prompt_lock(self.query_one(PromptPanel).choose(title, message, choices, vertical=vertical))

    async def _prompt_text(self, title: str, message: str) -> str | None:
        """Show a text prompt in the main window."""
        return await self._with_prompt_lock(self.query_one(PromptPanel).ask_text(title, message))

    async def _prompt_json(self, title: str, text: str) -> str | None:
        """Show a JSON editor prompt in the main window."""
        return await self._with_prompt_lock(self.query_one(PromptPanel).edit_json(title, text))

    async def _with_prompt_lock(self, prompt_waiter: Any) -> str | None:
        """Disable the prompt box while an in-window prompt is active."""
        prompt = self.query_one(PromptBox)
        was_disabled = prompt.disabled
        prompt.disabled = True
        try:
            return await prompt_waiter
        finally:
            prompt.disabled = was_disabled
            if self.is_mounted and self.ready and not self.busy and not prompt.disabled:
                self.action_focus_prompt()

    def waiting_started(
        self,
        label: str | None = None,
        *,
        immediate: bool = False,
        elapsed: bool | None = None,
    ) -> None:
        """Arm the transient working indicator while the turn is silent."""
        if label is not None:
            self._waiting_label = label
        if elapsed is None:
            elapsed = True
        self._waiting_generation += 1
        self._cancel_waiting_task()
        if not self.is_mounted or not self.busy or self._main_stream_active:
            return
        try:
            if self.query_one(PromptPanel).active:
                return
        except NoMatches:
            return
        if immediate:
            self.query_one(ChatLog).show_waiting(self._waiting_label, elapsed=elapsed)
            return
        generation = self._waiting_generation
        self._waiting_task = self.run_worker(
            self._show_waiting_after_delay(generation, elapsed=elapsed),
            name=f"waiting-{generation}",
            exclusive=False,
        )

    def waiting_finished(self) -> None:
        """Hide the transient working indicator and cancel pending timers."""
        self._cancel_waiting_task()
        if self.is_mounted:
            self.query_one(ChatLog).hide_waiting()

    async def _show_waiting_after_delay(
        self,
        generation: int,
        *,
        elapsed: bool | None = None,
    ) -> None:
        """Show working only if the current wait survives the grace period."""
        try:
            await asyncio.sleep(self._waiting_delay_seconds)
        except asyncio.CancelledError:
            return
        if generation != self._waiting_generation or not self.busy or not self.is_mounted or self._main_stream_active:
            return
        self.query_one(ChatLog).show_waiting(self._waiting_label, elapsed=elapsed)

    def _cancel_waiting_task(self) -> None:
        """Cancel the pending delayed thinking task if one exists."""
        task = self._waiting_task
        if task is None:
            return
        task.cancel()
        self._waiting_task = None

    def _mark_main_stream_active(self) -> None:
        """Track visible streaming activity for the current model message."""
        self._main_stream_active = True

    def _finish_main_stream_activity(self) -> None:
        """Stop suppressing waiting UI for the current model stream."""
        self._main_stream_active = False

    def _rearm_waiting_if_busy(self) -> None:
        """Start the silent-wait timer again after a visible runtime event."""
        if (
            self.busy
            and not self._main_stream_active
            and not self.query_one(ChatLog).has_live_tools()
        ):
            self.waiting_started()

    def startup_progress(self, state: str) -> None:
        """Update startup details without changing the operational lifecycle."""
        self.trace.startup(state)
        if not self.is_mounted:
            return
        try:
            self.query_one(ChatLog).startup_progress(state)
        except NoMatches:
            return

    def _set_status(self, *, state: str) -> None:
        """Update the status bar if it has been mounted."""
        if state not in OPERATIONAL_STATES:
            state = "running" if self.busy else "starting" if not self.ready else "ready"
        self.status_state = state
        if not self.is_mounted:
            return
        ensure_dashboard(
            self.session,
            model_name=self.model_name,
            context_limit_tokens=self.context_limit_tokens,
            context_limit_source=self.context_limit_source,
        )
        self.query_one(StatusBar).set_state(
            mode=self._mode_label(),
            model_name=self.model_name or "loading",
            state=state,
            dashboard=self.session.get("dashboard"),
            turns=int(self.session.get("turns") or 0),
        )
        self.query_one(TelemetryBar).set_state(
            model_name=self.model_name or "loading",
            dashboard=self.session.get("dashboard"),
            turns=int(self.session.get("turns") or 0),
        )
        self._sync_artifact_button()
        self._sync_mcp_button()
        self._sync_issues_button()

    def _set_ready_unless_error(self) -> None:
        """Return passive work to Ready without clearing a sticky Error."""
        if not self.busy and self.status_state != "error":
            self._set_status(state="ready")

    @on(Button.Pressed, "#mcp-status-button")
    def press_mcp_status(self, event: Button.Pressed) -> None:
        event.stop()
        self.open_mcp_panel()

    @on(Button.Pressed, "#model-settings-button")
    def press_model_settings(self, event: Button.Pressed) -> None:
        """Open the shared Settings panel directly on Models."""
        event.stop()
        self.run_worker(self._run_settings_command("models"), name="models-button", exclusive=False)

    def open_mcp_panel(self) -> None:
        """Shared button and slash-command pathway for the MCP panel."""
        if self.mcp_manager is None or not self.mcp_manager.show_status:
            self.notify("No configured MCP servers.", title="MCP")
            return
        if isinstance(self.screen, MCPPanelScreen):
            return
        self.push_screen(MCPPanelScreen(self.mcp_manager, self._run_reload_runtime_command))

    @on(Button.Pressed, "#artifact-status-button")
    async def press_artifact_status(self, event: Button.Pressed) -> None:
        """Reopen the exact retained Goal or Plan from the header."""
        event.stop()
        if self.busy:
            return
        if current_plan(self.session) is not None:
            await self.show_plan()
        elif current_goal(self.session) is not None:
            await self.show_goal()
        self.action_focus_prompt()

    def _sync_artifact_button(self) -> None:
        """Project the canonical retained artifact into one header control."""
        try:
            button = self.query_one("#artifact-status-button", Button)
        except NoMatches:
            return
        plan = current_plan(self.session)
        goal = current_goal(self.session)
        artifact = plan or goal
        if artifact is None:
            button.disabled = True
            button.remove_class("plan-artifact", "goal-artifact")
            self._sync_header_control_visibility()
            return
        kind = "Plan" if plan is not None else "Goal"
        state = ARTIFACT_STATUS_LABELS.get(str(artifact.get("status") or ""), "Draft")
        button.label = f"{kind} {state}"
        button.disabled = self.busy
        button.set_class(plan is not None, "plan-artifact")
        button.set_class(goal is not None, "goal-artifact")
        self._sync_header_control_visibility()

    def _sync_mcp_button(self) -> None:
        try:
            button = self.query_one("#mcp-status-button", Button)
        except NoMatches:
            return
        visible = self.mcp_manager is not None and self.mcp_manager.show_status
        if visible:
            symbol = mcp_summary_symbol(self.mcp_manager.servers.values(), spinner=self._mcp_spinner)
            button.label = f"{symbol} MCP {self.mcp_manager.usable_count}/{self.mcp_manager.configured_count}"
        self._sync_header_control_visibility()

    @on(Button.Pressed, "#issues-button")
    def press_issues(self, event: Button.Pressed) -> None:
        """Open current project tool failures without recording chat history."""
        event.stop()
        self._open_issues()

    def _open_issues(self) -> None:
        if not self.issues:
            self.notify("No unresolved issues.", title="Issues")
            return
        if isinstance(self.screen, IssuesScreen):
            return
        self.push_screen(IssuesScreen(self.issues))

    def _sync_issues(self) -> None:
        """Refresh the persistent Issues entry point and any open modal."""
        if not self.is_mounted:
            return
        self._sync_issues_button()
        if isinstance(self.screen, IssuesScreen):
            self.screen.dismiss()

    def _notify_explicit_reload(self) -> None:
        """Show one consolidated current-Issues toast after reload."""
        count = len(self.issues)
        if not count:
            return
        self.notify(
            f"{count} current issue{'s' if count != 1 else ''}.\nOpen Issues or run /issues.",
            title="Reload completed",
            severity="warning",
        )

    def _notify_startup_issues(self) -> None:
        """Show one grouped startup warning when custom tools are unavailable."""
        count = len(self.issues)
        if not count:
            return
        self.notify(
            f"{count} current issue{'s' if count != 1 else ''}.\n"
            "Open Issues or run /issues.",
            title="Configuration issues",
            severity="warning",
        )

    def _sync_issues_button(self) -> None:
        if not self.is_mounted:
            return
        try:
            button = self.query_one("#issues-button", Button)
        except NoMatches:
            return
        count = len(self.issues)
        button.label = f"Issues {count}"
        self._sync_header_control_visibility()

    def _sync_header_control_visibility(self) -> None:
        """Yield narrow header width to status, then artifact, MCP, and Issues."""
        if not self.is_mounted:
            return
        try:
            sidebar = self.query_one("#session-sidebar")
            controls = (
                (
                    self.query_one("#artifact-status-button", Button),
                    current_plan(self.session) is not None or current_goal(self.session) is not None,
                ),
                (
                    self.query_one("#mcp-status-button", Button),
                    self.mcp_manager is not None and self.mcp_manager.show_status,
                ),
                (self.query_one("#issues-button", Button), bool(self.issues)),
            )
        except NoMatches:
            return
        has_controls = any(wanted for _button, wanted in controls)
        sidebar.display = self.size.width >= (90 if has_controls else 72)
        main_width = self.size.width - 42 if sidebar.display else self.size.width
        slots = max(0, (main_width - 40) // 13)
        for button, wanted in controls:
            button.display = bool(wanted and slots > 0)
            if wanted and slots > 0:
                slots -= 1

    @on(ListView.Selected, "#sessions")
    def select_session(self, event: ListView.Selected) -> None:
        """Resume the selected session."""
        item = event.item
        if not isinstance(item, SessionItem):
            return
        if item.session_id == str(self.session.get("id") or ""):
            return
        if self.busy:
            self.system_message("finish the current turn before switching sessions", kind="warning")
            return
        self.run_worker(self._load_session(item.session_id), name="load-session", exclusive=True)

    async def _load_session(self, session_id: str) -> None:
        """Bootstrap and install a selected session."""
        if self.bootstrap is None:
            self.system_message("session switching needs the normal bootstrap path", kind="warning")
            return

        prompt = self.query_one(PromptBox)
        prompt.disabled = True
        self.busy = True
        self._set_status(state="running")
        try:
            state = await self.bootstrap(
                self.workspace,
                session_id,
                True,
                self.config,
                self,
            )
            self._install_state(state)
        except Exception as exc:
            error_path = self._write_error_report(
                exc,
                source="tui.session_load",
                context={"requested_session_id": session_id},
            )
            self.system_message(f"session load error: {exc}\nerror report: {error_path}", kind="error")
            self._set_status(state="error")
        finally:
            self.busy = False
            prompt.disabled = False
            self.action_focus_prompt()

    async def _rebuild_model_agents(self, old_settings: dict[str, Any]) -> None:
        """Apply Models settings without reloading configuration or MCP servers."""
        if self.config is None or self.checkpointer is None:
            return

        from agent.llm import get_llm, get_model_name, resolve_model_profile
        from config.llm import ConfigError
        from config.metadata import ModelMetadata, infer_model_metadata
        from config.settings import MAIN_MODEL, context_limit_tokens, model_assignment

        metadata = self._current_model_metadata()
        main_changed = model_assignment(old_settings, MAIN_MODEL) != model_assignment(
            self.config,
            MAIN_MODEL,
        )
        context_changed = context_limit_tokens(old_settings) != context_limit_tokens(self.config)
        if main_changed or context_changed:
            try:
                resolve_model_profile(self.config)
            except ConfigError:
                metadata = ModelMetadata(
                    context_limit_tokens(self.config),
                    "settings.models.context_limit_tokens",
                )
            else:
                inspect_model = get_llm(self.config, metadata=ModelMetadata())
                metadata = await infer_model_metadata(self.config, model=inspect_model)

        await self._rebuild_agents(metadata=metadata)

        self.model_name = get_model_name(self.config)
        self.context_limit_tokens = metadata.context_tokens
        self.context_limit_source = metadata.context_source
        self.runtime_snapshot = build_runtime_snapshot(
            self.config,
            self.launch_options,
            model_name=self.model_name,
        )
        ensure_dashboard(
            self.session,
            model_name=self.model_name,
            context_limit_tokens=self.context_limit_tokens,
            context_limit_source=self.context_limit_source,
        )
        if self.store is not None:
            self.store.save(self.session)
        self._refresh_startup_splash()
        self._set_status(state=self.status_state)

    def _current_model_metadata(self) -> Any:
        """Return the metadata currently bound to the active agent pair."""
        from config.metadata import ModelMetadata
        from config.settings import context_limit_tokens

        configured_limit = context_limit_tokens(self.config)
        return ModelMetadata(
            context_tokens=self.context_limit_tokens or configured_limit,
            context_source=(
                self.context_limit_source
                if self.context_limit_tokens
                else "settings.models.context_limit_tokens"
            ),
        )

    async def _refresh_model_metadata(self) -> None:
        """Refresh model metadata and rebuild agents when context changes."""
        if self.config is None or self.checkpointer is None:
            return
        from agent.llm import resolve_model_profile
        from config.llm import ConfigError

        try:
            resolve_model_profile(self.config)
        except ConfigError:
            return

        from agent.llm import get_llm
        from config.metadata import ModelMetadata, infer_model_metadata

        inspect_model = get_llm(self.config, metadata=ModelMetadata())
        metadata = await infer_model_metadata(self.config, model=inspect_model)
        if not metadata.context_tokens or metadata.context_tokens == self.context_limit_tokens:
            return

        await self._rebuild_agents(metadata=metadata)
        self.context_limit_tokens = metadata.context_tokens
        self.context_limit_source = metadata.context_source
        ensure_dashboard(
            self.session,
            model_name=self.model_name,
            context_limit_tokens=self.context_limit_tokens,
            context_limit_source=self.context_limit_source,
        )
        self.store.save(self.session)
        self._set_status(state=self.status_state)

    async def _rebuild_agents(self, metadata: Any | None = None) -> None:
        """Rebuild action and planning agents after settings or metadata changes."""
        if self.config is None or self.checkpointer is None:
            return

        metadata = metadata or self._current_model_metadata()

        from agent.llm import active_model_issues, model_unavailable_message
        from agent.resources import build_resources, configure_subagents
        from agent.resources.subagents import subagent_model_issues

        resources = build_resources(
            self.workspace,
            settings=self.config.get("settings"),
            config=None,
        )
        assignment_issues = [
            *active_model_issues(self.config),
            *subagent_model_issues(resources.subagent_discovery, self.config),
        ]
        agent = None
        plan_agent = None
        if self.config.get("settings_valid", True) and not assignment_issues:
            action_resources = configure_subagents(resources, self.config)
            agent, plan_agent = self._build_agent_pair(
                config=self.config,
                metadata=metadata,
                action_resources=action_resources,
            )
        mode_updates = self._agent_mode_updates(agent, plan_agent, self.config)
        mode_updates["resources"] = resources.metadata
        self.agent = agent
        self.plan_agent = plan_agent
        self.agent_unavailable_message = model_unavailable_message(self.config) if agent is None else ""
        self.tool_failures = resources.tool_failures
        self.issues = [
            *(self.config.get("issues") or []),
            *(self.mcp_manager.issues if self.mcp_manager is not None else []),
            *(self.mcp_manager.prompt_registry.issues if self.mcp_manager is not None else []),
            *resources.issues,
            *assignment_issues,
        ]
        self.mode.update(mode_updates)
        self.query_one(AutocompleteInput).set_project_backend(resources.project_backend)
        self._sync_issues()

    def _build_agent_pair(
        self,
        *,
        config: dict[str, Any],
        metadata: Any | None,
        action_resources: Any | None = None,
    ) -> tuple[Any, Any]:
        """Build both agents without replacing the active pair."""
        from agent.factory import build_agent, build_plan_agent
        from agent.resources import build_resources

        action_resources = action_resources or build_resources(
            self.workspace,
            settings=config.get("settings"),
            config=config,
        )
        plan_resources = build_resources(
            self.workspace,
            create_examples=False,
            settings=config.get("settings"),
            enable_execute=False,
            config=config,
            subagent_discovery=action_resources.subagent_discovery,
        )

        agent = build_agent(
            config=config,
            workspace=self.workspace,
            checkpointer=self.checkpointer,
            metadata=metadata,
            mcp_manager=self.mcp_manager,
            resources=action_resources,
        )
        plan_agent = build_plan_agent(
            config=config,
            workspace=self.workspace,
            checkpointer=self.checkpointer,
            metadata=metadata,
            mcp_manager=self.mcp_manager,
            resources=plan_resources,
        )
        return agent, plan_agent

    async def _mcp_registry_changed(self) -> None:
        """Install one rebuilt Act/Plan pair after a manager registry transition."""
        if self.config is not None:
            self.config["settings"] = load_settings(self.workspace)
        await self._rebuild_agents()
        if self.is_mounted:
            autocomplete = self.query_one(AutocompleteInput)
            autocomplete.set_mcp_manager(self.mcp_manager)
            self._set_status(state=self.status_state)
            if isinstance(self.screen, MCPPanelScreen):
                await self.screen.refresh_from_manager()

    async def approve_mcp_server(self, state: Any, preview: str) -> str:
        """Use MIRA's existing modal choice interaction for server trust."""
        answer = await self._prompt_choice(
            f"Allow MCP Server: {state.name}",
            preview,
            [("a", "Allow (a)"), ("d", "Deny (d)"), ("l", "Always allow (l)")],
        )
        return {"a": "allow", "l": "always_allow"}.get(answer, "deny")

    async def on_unmount(self) -> None:
        """Close every MCP runtime when the Textual app exits."""
        if self.mcp_manager is not None:
            await self.mcp_manager.shutdown()

    def _agent_mode_updates(
        self,
        agent: Any,
        plan_agent: Any,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Prepare mode projections before installing a rebuilt agent pair."""
        candidate = dict(self.mode)
        refresh_agent_specs(candidate, agent, plan_agent)
        candidate["rubric_enabled"] = rubric_enabled(config.get("settings"))
        candidate["rubric_max_iterations"] = rubric_max_iterations(config.get("settings"))
        return {
            key: candidate[key]
            for key in (
                "action_tools",
                "planning_tools",
                "resources",
                "rubric_enabled",
                "rubric_max_iterations",
            )
        }

    def _refresh_sessions(self) -> None:
        """Reload the session list if the store is available."""
        if self.store is None or not self.is_mounted:
            return
        self.query_one(SessionHistory).refresh_sessions(self.store, current_id=str(self.session.get("id") or ""))

    def _tick_status(self) -> None:
        """Refresh the clock and dashboard line."""
        if not self.ready or not self.is_mounted:
            return
        update_duration(self.session)
        try:
            self._set_status(state=self.status_state)
        except NoMatches:
            return

    def _tick_animations(self) -> None:
        """Advance lightweight chat animations."""
        if not self.is_mounted:
            return
        try:
            chat = self.query_one(ChatLog)
        except NoMatches:
            return
        chat.tick_waiting()
        chat.tick_startup()
        chat.tick_subagents()
        chat.tick_compaction()
        chat.tick_rubrics()
        chat.tick_tools()
        try:
            self.query_one(StatusBar).tick()
        except NoMatches:
            pass
        if self.mcp_manager is not None and any(state.transient for state in self.mcp_manager.servers.values()):
            self._mcp_spinner += 1
            self._sync_mcp_button()
        try:
            self.query_one(SubagentsPanel).tick()
        except NoMatches:
            return

    def on_resize(self) -> None:
        """Keep the main panel usable in very narrow terminals."""
        self._sync_sidebar_visibility()

    def _sync_sidebar_visibility(self) -> None:
        """Hide history before the fixed sidebar can squeeze chat to zero width."""
        if not self.is_mounted:
            return
        self._sync_header_control_visibility()

    def _mode_label(self) -> str:
        """Return the compact mode label shown in the status line."""
        return "PLAN" if self.mode.get("planning") else "ACT"

    def _record_prompt_history(self, text: str) -> None:
        """Remember submitted prompt text and persist it for normal app runs."""
        self.query_one(PromptBox).remember(text)
        if not self.persist_prompt_history:
            return
        try:
            append_prompt_history(self.history_path, text)
        except OSError as exc:
            self.system_message(f"could not update prompt history: {exc}", kind="warning")


def read_prompt_history(path: Path) -> list[str]:
    """Read MIRA prompt history entries from a prompt-toolkit-style file."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []

    entries: list[str] = []
    current: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            if current:
                entries.append("\n".join(current).strip())
                current = []
            continue
        if stripped.startswith("+"):
            current.append(stripped[1:])
    if current:
        entries.append("\n".join(current).strip())
    return entries


def append_prompt_history(path: Path, text: str) -> None:
    """Append one submitted prompt to the workspace history file."""
    entry = text.strip()
    if not entry:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat(sep=" ", timespec="microseconds")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n# {timestamp}\n")
        for line in entry.splitlines():
            handle.write(f"+{line}\n")


def plan_title(plan: dict[str, Any]) -> str:
    """Return a compact plan title for status text."""
    return str(plan.get("title") or "Implementation Plan")


def goal_title(goal: dict[str, Any]) -> str:
    """Return a compact Goal title for status text."""
    return str(goal.get("title") or "Goal")
