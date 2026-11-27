"""Textual application shell for MIRA."""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Awaitable, Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

from rich.pretty import pretty_repr
from textual import on
from textual.actions import SkipAction
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.events import Key
from textual.reactive import reactive
from textual.widgets import Button, ContentSwitcher, DirectoryTree, ListView, Static

from mira.api import LaunchOptions, RuntimeSnapshot
from mira.settings import (
    EXECUTE_TOOL,
    tool_enabled,
    read_configuration,
    configure_workspace,
    configure_git,
    preview_tracing,
)
from mira.api import report_error as write_error_report
from mira import MiraApplication, MiraSession
from mira.api import ContextOverflowError, MessageEvent, RuntimeEvent, ToolEvent
from ui.shared.trace import TraceStream
from ui.shared.interrupts import (
    ASK_USER_OPEN_OPTION,
    action_choices,
    action_preview,
    action_requests,
    action_title,
    ask_user_options,
    ask_user_question,
    ask_user_request,
    mcp_elicitation_preview,
    mcp_elicitation_request,
)
from ui.textual.artifact_review import PendingArtifactReview
from ui.textual.adapter import TextualFrontend
from ui.textual.commands.dispatcher import handle_command
from ui.textual.runtime_report import prompts_table, runtime_report, workflows_table
from ui.textual.widgets import (
    AutocompleteInput,
    ChatLog,
    ContextReportScreen,
    ContextStatus,
    Inspector,
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
from ui.textual.widgets.subagent_panel import SubagentSelected
from ui.textual.widgets.rubric_bubble import RubricInspectionSelected
from ui.textual.widgets.tool_bubble import SubagentHistoryAnchor
from ui.textual.widgets.workflow_bubble import (
    WorkflowAgentSelected,
    WorkflowFinalStateSelected,
    WorkflowNodeSelected,
)
from ui.textual.widgets.mcp_panel import mcp_summary_symbol
from ui.textual.widgets.chat_log import DEFAULT_TOOL_OUTPUT_CHARS
from ui.textual.widgets.session_history import (
    SessionActionMenuScreen,
    SessionItem,
    session_display_title,
)
from ui.textual.platform.windows.clipboard import set_windows_clipboard
from ui.textual.platform.windows.file_picker import choose_local_files
from ui.textual.platform.windows.input import driver_class_for_platform
from ui.textual.platform.windows.scrollbars import configure_scrollbars_for_platform

from ui.textual.widgets.workspace import Workspace
from ui.textual.widgets.workspace_controls import DragDivider, ModelAliasPopup
from ui.textual.widgets.workspace_surfaces import ChatSurface, ComposerActions

DESTRUCTIVE_HISTORY_COMMANDS = {
    "/clear-chat",
    "/clear-all-chats",
    "/clear-uploads",
    "/clear-all-uploads",
    "/clear-errors",
    "/clear-prompts",
}
BUSY_MUTATING_COMMANDS = {
    "/act",
    "/clear",
    "/compact",
    "/goal-clear",
    "/goal-resume",
    "/models",
    "/new-chat",
    "/plan",
    "/plan-clear",
    "/plan-resume",
    "/reload",
    "/reload-runtime",
    "/settings",
    *DESTRUCTIVE_HISTORY_COMMANDS,
}
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

    CSS_PATH = ["styles/mira.tcss", "styles/workspace.tcss"]
    manages_subagent_animation = True
    busy = reactive(False)
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
        self.prebuilt = prebuilt
        self.tool_output_chars = int(tool_output_chars)
        self.history_path = self.workspace / ".mira" / "history.txt"
        self.persist_prompt_history = prebuilt is None
        self.local_file_uploads_enabled = sys.platform == "win32"
        self._last_file_picker_directory = self.workspace
        self._file_ingestion_active = False

        self.application: MiraApplication | None = None
        self.core_session: MiraSession | None = None
        self.session: dict[str, Any] = {"id": "", "workspace": str(self.workspace), "turns": 0}
        self.model_name = ""
        self.context_limit_tokens: int | None = None
        self.context_limit_source = "unknown"
        self.runtime_snapshot: RuntimeSnapshot | None = None
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
        self._foreground_interrupt_lock = asyncio.Lock()
        self._foreground_interrupt_owner: asyncio.Task[Any] | None = None
        self._artifact_action_disabled: dict[Button, bool] = {}
        self.history_collapsed = True
        self.sidebar_kind = "history"
        self.sidebar_width = 42
        self.effective_model_alias = ""
        self._selection_before_drag = None
        self.mcp_manager: Any | None = None
        self._mcp_spinner = 0
        self._mcp_activity_snapshots: dict[str, dict[str, Any]] = {}
        self._mcp_activity_revealed: set[str] = set()
        self._mcp_activity_tasks: dict[str, asyncio.Task[None]] = {}
        self._mcp_reveal_delay_seconds = 0.75
        self._reload_in_progress = False
        self.tool_failures: list[Any] = []
        self.issues: list[Any] = []
        self.agent_unavailable_message = "Main model is not configured. Run /models."
        self.trace = TraceStream.disabled(output_chars=self.tool_output_chars)
        self._subagent_live_active = False
        self._pending_artifact_review: PendingArtifactReview | None = None

    def compose(self) -> ComposeResult:
        """Compose the Textual layout."""
        with Horizontal(id="status-row"):
            yield StatusBar(id="status")
            yield Button("Artifact", id="artifact-status-button")
            yield Button("MCP 0/0", id="mcp-status-button")
            yield Button("Issues 0", id="issues-button")
        with Horizontal(id="app-shell"):
            with Vertical(id="navigation-rail"):
                yield Button("💬", id="rail-chat", tooltip="Chat History")
                yield Button(DirectoryTree.ICON_NODE.strip(), id="rail-files", tooltip="File Explorer")
                yield Static("", id="rail-space")
                yield Button("🔧", id="rail-settings", tooltip="Settings · General")
            yield Static("", id="rail-panel-gap")
            with Vertical(id="session-sidebar"):
                with Horizontal(id="session-sidebar-header"):
                    yield Static("HISTORY", id="session-sidebar-title")
                    yield Button("+ New", id="new-chat")
                    yield Button("×", id="history-collapse", tooltip="Collapse sidebar")
                with ContentSwitcher(initial="history-panel", id="sidebar-content"):
                    with Vertical(id="history-panel"):
                        yield SessionHistory(id="sessions")
                    with Vertical(id="files-panel"):
                        yield DirectoryTree(self.workspace, id="files-tree")
            yield DragDivider("sidebar", id="side-divider")
            with Vertical(id="main-panel"):
                chat = ChatSurface(
                    ChatLog(tool_output_chars=self.tool_output_chars,
                            subagent_runs_provider=self._current_subagent_runs, id="chat-log"),
                    PromptPanel(),
                    SubagentsPanel(id="subagents-panel"),
                    AutocompleteInput(
                        compact=True,
                        tool_provider=self._active_autocomplete_tools,
                        subagent_provider=self._active_autocomplete_subagents,
                        skill_registry_provider=self._skill_registry,
                        workflow_registry_provider=self._workflow_registry,
                    ),
                )
                yield Workspace(chat, Inspector(lambda key: self.core_session.inspection(key) if self.core_session else None,
                                tool_output_chars=self.tool_output_chars, embedded=True, id="inspector"))
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
            if self.core_session is not None and self.core_session.snapshot(include_transcript=False).pending != "none":
                self.run_worker(self._recover_open_session(), name="recover-session")
            return
        self.query_one(ChatLog).startup_loading(workspace=str(self.workspace), state="starting...")
        self.call_after_refresh(self._start_startup_worker)

    def _start_startup_worker(self) -> None:
        """Start bootstrap after the first visible frame has rendered."""
        self.run_worker(self._startup(), name="startup", exclusive=True)

    async def _startup(self) -> None:
        application = None
        try:
            self.startup_progress("loading model metadata...")
            application = await MiraApplication.start(workspace=self.workspace, frontend=TextualFrontend(self), config=self.config, launch_options=self.launch_options)
            core_session = await application.open_session(self.session_id, resume=self.resume)
            self._install_state({"application": application, "core_session": core_session})
            self._notify_startup_issues()
            await self._recover_open_session()
        except Exception as exc:
            if application is not None and self.application is not application:
                with suppress(Exception):
                    await application.shutdown()
            error_path = self._write_error_report(exc, source="tui.startup")
            self.system_message(f"startup error: {exc}\nerror report: {error_path}", kind="error")
            self.busy = False
            self._set_status(state="error")
            self.query_one(PromptBox).disabled = False
            self.action_focus_prompt()

    async def _retry_startup(self) -> None:
        """Retry the complete bootstrap after an initial startup failure."""
        prompt = self.query_one(PromptBox)
        self.busy = True
        self._main_stream_active = False
        prompt.disabled = True
        self._set_status(state="running")
        await self._startup()

    def _refresh_runtime_view(self) -> None:
        self.config = self.application.configuration()
        self.runtime_snapshot = self.application.runtime()
        self.mcp_manager = self.application.mcp
        self.issues = list(self.config["issues"])
        self.tool_failures = list(self.config["tool_failures"])
        self.agent_unavailable_message = self.config["agent_unavailable_message"]
        self._refresh_session_view()

    def _install_state(self, state: dict[str, Any]) -> None:
        self.application = state["application"]
        self.core_session = state["core_session"]
        self._refresh_runtime_view()
        self._refresh_session_view(full=True)
        self.ready = True
        self.busy = False
        self._main_stream_active = False
        prompt = self.query_one(PromptBox)
        prompt.disabled = False
        prompt.set_history(self.application.prompt_history())
        autocomplete = self.query_one(AutocompleteInput)
        autocomplete.set_mcp_manager(self.application.mcp)
        autocomplete.set_file_provider(self.application.list_files)
        chat = self.query_one(ChatLog)
        self._restore_chat_viewport()
        self.query_one(SubagentsPanel).reset()
        chat.clear_log(preserve_startup_activity=True)
        chat.startup(model_name=self.model_name, session_id=self.session["id"], workspace=self.session["workspace"])
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
        return error_path

    @on(PromptBox.Submitted)
    async def submit_prompt(self, event: PromptBox.Submitted) -> None:
        """Handle submitted prompt text."""
        text = event.value.strip()
        self._refresh_session_view()
        prompt = self.query_one(PromptBox)
        prompt.value = ""
        if not text:
            self.action_focus_prompt()
            return
        if self.busy:
            command = text.partition(" ")[0]
            if (
                command in BUSY_MUTATING_COMMANDS
                or command == "/goal"
                or command.startswith("/workflow__")
            ):
                self._notify_busy_action("using that command")
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
        if text == "/workflow-demo":
            self.busy = True
            self._main_stream_active = False
            self._set_status(state="running")
            prompt.disabled = True
            self.turn_worker = self.run_worker(
                self._run_workflow_demo(),
                name="workflow-demo",
                exclusive=True,
            )
            return
        if text == "/workflow-demo-phase2":
            self.busy = True
            self._main_stream_active = False
            self._set_status(state="running")
            prompt.disabled = True
            self.turn_worker = self.run_worker(
                self._run_workflow_demo_phase2(),
                name="workflow-demo-phase2",
                exclusive=True,
            )
            return

        if text == "/workflows":
            self._run_workflows_command()
            self.action_focus_prompt()
            return

        if text.startswith("/workflow__"):
            try:
                resolved = self.application.workflow_invocation(text)
            except ValueError as error:
                self.system_message(str(error), kind="warning")
                self.action_focus_prompt()
                return
            if resolved is None:
                self.system_message("unknown workflow command", kind="warning")
                self.action_focus_prompt()
                return
            spec, inputs = resolved
            self.busy = True
            self._main_stream_active = False
            self._set_status(state="running")
            prompt.disabled = True
            self.turn_worker = self.run_worker(
                self._run_registered_workflow(spec, inputs, text),
                name=f"workflow-{spec.name}",
                exclusive=True,
            )
            return

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
        if text.startswith(("/skill__", "/mcp__", "/prompt__")):
            try:
                prepared = await self.application.prepare_input(text)
            except ValueError as error:
                self.system_message(str(error), kind="warning")
                self.action_focus_prompt()
                return

        plan_prompt = text[len("/plan"):].strip() if text == "/plan" or text.startswith("/plan ") else None
        if text == "/continue":
            if (
                self.core_session is None
                or self.core_session.snapshot(include_transcript=False).pending != "incomplete"
            ):
                self.system_message("no interrupted operation to continue", kind="muted")
            else:
                self.system_message(
                    "Continuing the saved operation; an external side effect may repeat.",
                    kind="warning",
                )
                self.busy = True
                self._main_stream_active = False
                self._set_status(state="running")
                prompt.disabled = True
                self.turn_worker = self.run_worker(
                    self._run_incomplete_continuation(), name="continue", exclusive=True
                )
                return
            self.action_focus_prompt()
            return
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
            await self.clear_goal()
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
                pinned_skills=getattr(prepared, "pinned_skills", None),
                attachments=attachments,
            ),
            name="turn",
            exclusive=True,
        )

    async def _run_incomplete_continuation(self) -> None:
        await self._run_operation(self.core_session.continue_incomplete())

    async def _run_operation(self, operation: Awaitable[Any], *, source: str = "tui.turn", error_label: str = "error") -> None:
        previous_goal = self.session.get("current_goal") or {}
        previous_goal_state = (previous_goal.get("id"), previous_goal.get("status"))
        try:
            await operation
            self._set_status(state="ready")
        except asyncio.CancelledError:
            self.finish_turn(cancelled=True)
            self.system_message("Goal generation cancelled" if source == "tui.goal" else "turn cancelled", kind="warning")
            self._set_status(state="ready")
            raise
        except ContextOverflowError:
            self.finish_turn(cancelled=True, tool_status="interrupted")
            self._set_status(state="ready")
        except Exception as exc:
            self.finish_turn(cancelled=True, tool_status="interrupted")
            error_path = self._write_error_report(exc, source=source)
            self.system_message(f"{error_label}: {exc}\nerror report: {error_path}", kind="error")
            self._set_status(state="error")
        finally:
            self._refresh_session_view()
            self._refresh_sessions()
            goal = self.session.get("current_goal") or {}
            if (goal.get("id"), goal.get("status")) != previous_goal_state:
                self._resolve_goal_attempt_status()
            self.turn_worker = None
            self.busy = False
            self.finish_turn()
            self.waiting_finished()
            self.query_one(PromptBox).disabled = False
            self.action_focus_prompt()

    async def _run_turn(self, text: str, *, prepared_messages: list[Any] | None = None,
                        display_text: str | None = None, pinned_skills: list[str] | None = None,
                        attachments: list[dict[str, str]] | None = None) -> None:
        options = {key: value for key, value in {
            "prepared_messages": prepared_messages, "display_text": display_text,
            "pinned_skills": pinned_skills, "attachments": attachments,
        }.items() if value is not None}
        await self._run_operation(self.core_session.prompt(text, **options))

    async def _run_compact_command(self) -> None:
        """Compact the current action or planning conversation."""
        self.compaction_started()
        try:
            result = await self.core_session.compact()
            if result.compacted:
                self._refresh_session_view()
                self._refresh_sessions()
                self.compaction_finished("context compacted", resume_waiting=False)
            elif result.reason in {"no_messages", "nothing_to_compact"}:
                self.compaction_finished("nothing to compact", success=False, resume_waiting=False)
            elif result.reason == "archive_failed":
                self.compaction_finished("context compaction failed", success=False, resume_waiting=False)
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

    async def _run_workflow_demo(self) -> None:
        payload = {"events": []}
        self._show_workflow_input(payload)
        await self._run_operation(self.core_session.run_workflow("/workflow-demo", inputs=payload, record_user=False), source="tui.workflow-demo")


    async def _run_workflow_demo_phase2(self) -> None:
        payload = {"messages": [{"role": "user", "content": "Confirm that the Workflow Phase 2 demo is running."}]}
        self._show_workflow_input(payload)
        await self._run_operation(self.core_session.run_workflow("/workflow-demo-phase2", inputs=payload, record_user=False), source="tui.workflow-demo-phase2")


    async def _run_registered_workflow(self, spec: Any, inputs: dict[str, Any], command_text: str) -> None:
        await self._run_operation(self.core_session.run_workflow(command_text), source=f"tui.workflow.{spec.name}", error_label=f"workflow {spec.name} error")

    def _show_workflow_input(self, payload: Any) -> None:
        """Render one process-local Workflow payload as a normal user message."""
        self.query_one(ChatLog).user_message(pretty_repr(payload, expand_all=True))

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

    @on(Button.Pressed, "#history-collapse")
    def press_history_collapse(self, event: Button.Pressed) -> None:
        """Collapse the optional sidebar beside the permanent navigation rail."""
        event.stop()
        previous_focus = self.screen.focused
        self.history_collapsed = True
        self._sync_sidebar_visibility()
        self.call_after_refresh(self._restore_focus_after_history_toggle, previous_focus)

    @on(Button.Pressed, "#rail-chat")
    @on(Button.Pressed, "#rail-files")
    def press_sidebar(self, event: Button.Pressed) -> None:
        event.stop()
        kind = "history" if event.button.id == "rail-chat" else "files"
        previous_focus = self.screen.focused
        self.history_collapsed = not self.history_collapsed if self.sidebar_kind == kind else False
        self.sidebar_kind = kind
        self._sync_sidebar_visibility()
        self.call_after_refresh(self._restore_focus_after_history_toggle, previous_focus)

    @on(Button.Pressed, "#rail-settings")
    def press_workspace_settings(self, event: Button.Pressed) -> None:
        event.stop()
        self.run_worker(self._run_settings_command("general"), name="workspace-settings")

    @on(DirectoryTree.FileSelected, "#files-tree")
    async def preview_workspace_file(self, event: DirectoryTree.FileSelected) -> None:
        event.stop()
        await self.query_one(Workspace).open_file(event.path)

    def _restore_focus_after_history_toggle(self, previous_focus: Any) -> None:
        """Return idle mouse toggles to input without taking interrupt focus."""
        if not self.is_mounted:
            return
        settings = self._settings_panel
        interrupt_owns_focus = bool(
            len(self.screen_stack) > 1
            or self._foreground_interrupt_owner is not None
            or self._pending_artifact_review is not None
            or (settings is not None and settings.display)
            or self.query_one(PromptPanel).active
        )
        if interrupt_owns_focus:
            focus_id = getattr(previous_focus, "id", None)
            focus_target = previous_focus
            if focus_target is not None and not focus_target.is_mounted and focus_id:
                try:
                    focus_target = self.query_one(f"#{focus_id}")
                except NoMatches:
                    focus_target = None
            if (
                focus_target is not None
                and focus_target.is_mounted
                and not focus_target.disabled
            ):
                focus_target.focus()
            if focus_id:
                self.set_timer(0.05, lambda: self._restore_interaction_focus(focus_id))
            return
        prompt = self.query_one(PromptBox)
        if not prompt.is_mounted or prompt.disabled:
            return
        prompt.focus()

    def _restore_interaction_focus(self, focus_id: str) -> None:
        """Reacquire an interaction control if layout refresh recomposed it."""
        if not self.is_mounted:
            return
        settings = self._settings_panel
        if not (
            len(self.screen_stack) > 1
            or self._foreground_interrupt_owner is not None
            or self._pending_artifact_review is not None
            or (settings is not None and settings.display)
            or self.query_one(PromptPanel).active
        ):
            return
        try:
            focus_target = self.query_one(f"#{focus_id}")
        except NoMatches:
            return
        if focus_target.is_mounted and not focus_target.disabled:
            focus_target.focus()

    @on(SessionItem.PinRequested)
    def pin_history_session(self, event: SessionItem.PinRequested) -> None:
        """Toggle pin metadata without treating the row as a session selection."""
        event.stop()
        if self.busy:
            self._notify_busy_action("pinning a chat")
            return
        self._set_history_pin(event.item, not event.item.pinned)
        self.call_after_refresh(self.action_focus_prompt)

    @on(SessionItem.MenuRequested)
    def open_history_session_menu(self, event: SessionItem.MenuRequested) -> None:
        """Open the compact native action menu for one history row."""
        event.stop()
        if self.busy:
            self._notify_busy_action("changing a saved chat")
            return
        self.run_worker(
            self._run_history_session_menu(event.item),
            name="history-session-menu",
            exclusive=False,
        )

    @on(SessionItem.RenameSubmitted)
    def rename_history_session(self, event: SessionItem.RenameSubmitted) -> None:
        event.stop()
        if self.busy:
            self._notify_busy_action("renaming a chat")
            return
        self.application.update_session(event.item.session_id, title=event.value)
        self._refresh_session_view()
        self._refresh_sessions()

    async def _run_history_session_menu(self, item: SessionItem) -> None:
        """Apply the selected action to the exact originating session."""
        row_region = item.region
        action = await self.push_screen_wait(
            SessionActionMenuScreen(
                pinned=item.pinned,
                anchor_x=row_region.right,
                anchor_y=row_region.y,
            )
        )
        if action == "rename" and item.is_mounted:
            item.begin_rename()
            return
        if action == "pin":
            self._set_history_pin(item, not item.pinned)
        elif action == "delete":
            await self._delete_history_session(item)
        self.call_after_refresh(self.action_focus_prompt)

    def _set_history_pin(self, item: SessionItem, pinned: bool) -> None:
        self.application.update_session(item.session_id, pinned=pinned)
        self._refresh_session_view()
        self._refresh_sessions()


    async def _delete_history_session(self, item: SessionItem) -> None:
        """Confirm and delete only the selected session."""
        title = session_display_title(item.record)
        answer = await self._prompt_choice(
            "Delete Chat?",
            f'Delete saved chat "{title}"?\n\n{DESTRUCTIVE_CONFIRM_HINT}',
            DESTRUCTIVE_CONFIRM_CHOICES,
        )
        if answer != "o":
            return
        if self.application is None or not await self.application.delete_session(item.session_id):
            self.notify("The saved chat no longer exists.", title="History")
            self._refresh_sessions()
            return
        if item.session_id == str(self.session.get("id") or ""):
            await self._handle_new_chat()
        else:
            self._refresh_sessions()

    async def _run_new_chat(self) -> None:
        """Create and switch to a fresh session outside event handlers."""
        try:
            previous_session = self.session
            if await self._handle_new_chat() and self.session is not previous_session:
                self._set_status(state="ready")
        except Exception as exc:
            self.system_message(f"new chat error: {exc}", kind="error")
            self._set_status(state="error")
        finally:
            self.action_focus_prompt()

    async def _handle_new_chat(self) -> bool:
        """Create a fresh saved session while preserving the current one."""
        if self.busy:
            self._notify_busy_action("starting a new chat")
            return True
        if self.application is None:
            self.system_message("new chat needs the normal session store", kind="warning")
            return True

        core_session = await self.application.open_session()
        self._activate_core_session(core_session)
        self.system_message("started new chat", kind="info")
        return True

    def _refresh_session_view(self, *, full: bool = False) -> None:
        if self.core_session is None:
            return
        snapshot = self.core_session.snapshot(include_transcript=full)
        previous_events = self.session.get("events", [])
        self.session = {
            **snapshot.history, "id": snapshot.session_id, "workspace": snapshot.workspace,
            "title": snapshot.title, "turns": snapshot.turns, "dashboard": dict(snapshot.dashboard),
            "current_goal": snapshot.current_goal, "current_plan": snapshot.current_plan,
            "events": list(snapshot.transcript) if full else previous_events,
        }
        self.mode = {
            **snapshot.execution, "planning": snapshot.mode == "planning",
            "resources": dict(snapshot.resources), "current_goal": snapshot.current_goal,
            "current_plan": snapshot.current_plan, "action_tools": list(snapshot.tools),
            "planning_tools": list(snapshot.tools), "rubric_enabled": snapshot.rubric["enabled"],
            "rubric_max_iterations": snapshot.rubric["max_iterations"],
        }
        self.model_name = str(snapshot.model["name"])
        self.context_limit_tokens = snapshot.model["context_limit_tokens"]
        self.context_limit_source = str(snapshot.model["context_limit_source"])
        self.effective_model_alias = str(snapshot.model["alias"])
        self.models_available = bool(snapshot.model["available"])

    def _activate_core_session(self, core_session: MiraSession) -> None:
        self.core_session = core_session
        self._refresh_session_view(full=True)
        self.query_one(SubagentsPanel).reset()
        self._render_current_session()

    async def _recover_open_session(self) -> None:
        pending = self.core_session.snapshot(include_transcript=False).pending
        if pending == "none":
            return
        if pending == "incomplete":
            self.system_message("The previous operation stopped before its result was checkpointed. External side effects may already have occurred. Review them before continuing.", kind="warning")
            return
        await self._run_operation(self.core_session.resume_pending())

    async def _run_prompts_command(self) -> None:
        """List the exact shared prompt registry after lazy MCP discovery."""
        if self.mcp_manager is None:
            self.command_output(prompts_table([]))
            self.action_focus_prompt()
            return
        await self.mcp_manager.discover_prompts()
        specs = self.application.commands("prompt")
        self.command_output(prompts_table(specs))
        self.action_focus_prompt()

    def _run_workflows_command(self) -> None:
        """List the exact process-local launchable Workflow registry."""
        specs = self.application.commands("workflow")
        self.command_output(workflows_table(specs))

    def _skill_registry(self) -> tuple[Any, ...]:
        return self.application.commands("skill") if self.application is not None else ()


    def _workflow_registry(self) -> tuple[Any, ...]:
        return self.application.commands("workflow") if self.application is not None else ()


    async def _handle_plan_action(self, action: str, plan_id: str) -> None:
        """Resolve the active structured plan."""
        self._refresh_session_view()
        pending = self._pending_artifact_review
        if pending is not None and pending.matches("plan", plan_id):
            await self._handle_pending_artifact_action(pending, action)
            return
        plan = self.session.get("current_plan")
        if plan is None or str(plan.get("id") or "") != plan_id:
            self.system_message("that plan is no longer active", kind="warning")
            return
        if self.busy:
            self._notify_busy_action("resolving the plan")
            return

        if action == "close":
            self.query_one(ChatLog).dismiss_plan(plan_id)
            await self.core_session.dismiss_artifact("plan")
            self.system_message(f"closed plan \"{plan_title(plan)}\"", kind="muted")
            self.action_focus_prompt()
            return

        if action == "clear":
            await self.clear_plan()
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
            started = await self.core_session.start_artifact("plan")
            if started is None:
                self.system_message("no current Plan", kind="muted")
                return
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
        if not await self._confirm_formal_replacement("Goal"):
            self.system_message("kept the current formal work", kind="muted")
            self.busy = False
            self.turn_worker = None
            self.query_one(PromptBox).disabled = False
            self.action_focus_prompt()
            return
        await self._run_operation(self.core_session.create_goal(objective), source="tui.goal")

    async def _handle_goal_action(self, action: str, goal_id: str) -> None:
        """Resolve an action from the current Goal bubble."""
        self._refresh_session_view()
        pending = self._pending_artifact_review
        if pending is not None and pending.matches("goal", goal_id):
            await self._handle_pending_artifact_action(pending, action)
            return
        value = self.session.get("current_goal")
        if value is None or str(value.get("id") or "") != goal_id:
            self.system_message("that Goal is no longer current", kind="warning")
            return
        if self.busy:
            self._notify_busy_action("resolving the Goal")
            return

        if action == "close":
            self.query_one(ChatLog).dismiss_goal(goal_id)
            await self.core_session.dismiss_artifact("goal")
            self.system_message(f'closed Goal "{goal_title(value)}"', kind="muted")
            self.action_focus_prompt()
            return

        if action == "clear":
            await self.clear_goal()
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
            started = await self.core_session.start_artifact("goal")
            if started is None:
                self.system_message("no current Goal", kind="muted")
                return
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

    async def _handle_pending_artifact_action(
        self,
        review: PendingArtifactReview,
        action: str,
    ) -> None:
        """Resolve only the exact provisional artifact awaiting finalizer HITL."""
        artifact = review.artifact
        title = plan_title(artifact) if review.kind == "plan" else goal_title(artifact)
        if review is not self._pending_artifact_review or review.future.done():
            self.system_message(f"that {review.kind} review is no longer pending", kind="warning")
            return
        if action not in {"implement", "close", "revise", "clear"}:
            self.system_message(f"unknown {review.kind} review action", kind="warning")
            return

        if action in {"revise", "implement"} and not self._model_agents_available():
            return
        values: dict[str, Any] = {}
        if action == "revise":
            label = "Plan" if review.kind == "plan" else "Goal"
            feedback = await self._prompt_text(
                f"Revise {label}",
                f"What should MIRA change about this {label}?",
                foreground_owned=True,
            )
            feedback = (feedback or "").strip()
            if not feedback:
                self.system_message(f'kept {label} "{title}" under review', kind="muted")
                return
            values["feedback"] = feedback
        if action == "implement":
            self.query_one(ChatLog).begin_artifact_implementation(
                review.kind,
                str(artifact.get("id") or ""),
            )
            self.system_message(f'implementing {review.kind} "{title}"', kind="status")
        review.resolve(action, **values)

    async def review_artifact(
        self,
        kind: str,
        interrupt: Any,
        artifact: dict[str, Any],
    ) -> dict[str, Any]:
        """Present a Core-owned provisional artifact and await its UI decision."""
        return await self.foreground_interrupt(
            lambda: self._review_artifact(kind, interrupt, artifact)
        )

    async def _review_artifact(
        self,
        kind: str,
        interrupt: Any,
        artifact: dict[str, Any],
    ) -> dict[str, Any]:
        """Present one serialized artifact review in the foreground."""
        del interrupt
        self.query_one(Workspace).activate("chat")
        self.waiting_finished()
        self.mode["planning_stage"] = (
            "plan_finalize" if kind == "plan" else "goal_finalize"
        )
        artifact_id = str(artifact.get("id") or "")
        created_at = ""
        for event in reversed(self.session.get("events", [])):
            value = event.get(kind) if isinstance(event, dict) else None
            if isinstance(value, dict) and str(value.get("id") or "") == artifact_id:
                created_at = str(event.get("created_at") or "")
                break
        review = PendingArtifactReview.create(
            kind,  # type: ignore[arg-type]
            artifact,
        )
        self._pending_artifact_review = review
        # Reserve the header control slot before the provisional bubble is laid
        # out. Implementing it then swaps review state for retained state
        # without resizing the transcript; resumed artifacts already have this
        # slot, which is why their transition was stable.
        self._sync_header_control_visibility()
        if kind == "plan":
            self.query_one(ChatLog).present_plan(
                artifact,
                active=True,
                status="proposed",
                created_at=created_at,
            )
        else:
            self.query_one(ChatLog).present_goal(
                artifact,
                active=True,
                status="proposed",
                created_at=created_at,
            )
        self.call_after_refresh(self._sync_artifact_actions)
        try:
            return await review.wait()
        finally:
            if self._pending_artifact_review is review:
                self._pending_artifact_review = None
                self.call_after_refresh(self._sync_header_control_visibility)
                self.call_after_refresh(self._sync_artifact_actions)

    def artifact_state_changed(self, event: Any) -> None:
        """Project one authoritative Core artifact transition into chat widgets."""
        self._refresh_session_view()
        artifact = dict(event.artifact or {})
        artifact_id = str(event.artifact_id or artifact.get("id") or "")
        decision = dict(event.decision or {})
        previous_plan_id = str(decision.get("previous_plan_id") or "")
        previous_goal_id = str(decision.get("previous_goal_id") or "")
        if previous_plan_id and previous_plan_id != artifact_id:
            self.query_one(ChatLog).resolve_plan(previous_plan_id, "superseded")
        if previous_goal_id and previous_goal_id != artifact_id:
            self.query_one(ChatLog).resolve_goal(previous_goal_id, "superseded")
        if event.phase == "proposed":
            return
        if event.artifact_type == "plan":
            if event.phase in {"clear", "close"}:
                self.query_one(ChatLog).dismiss_plan(artifact_id)
            else:
                status = {"revise": "rejected", "implement": "active"}.get(
                    event.phase,
                    event.phase,
                )
                self.query_one(ChatLog).resolve_plan(artifact_id, status)
        else:
            if event.phase in {"clear", "close"}:
                self.query_one(ChatLog).dismiss_goal(artifact_id)
            else:
                status = {"revise": "rejected", "implement": "active"}.get(
                    event.phase,
                    event.phase,
                )
                self.query_one(ChatLog).resolve_goal(artifact_id, status)
        self._sync_artifact_button()

    async def _run_turn_for_goal(self, value: dict[str, Any]) -> None:
        await self._run_operation(self.core_session.prompt("Work toward the retained Goal.",
            display_text=f"Implement Goal: {goal_title(value)}", record_user=False), source="tui.goal_turn")

    async def _run_goal_revision(self, value: dict[str, Any], feedback: str) -> None:
        await self._run_operation(self.core_session.revise_artifact("goal", feedback), source="tui.goal_revision")

    async def _run_turn_for_plan(self, plan: dict[str, Any]) -> None:
        await self._run_operation(self.core_session.prompt("Implement the approved plan.",
            display_text=f"Implement plan: {plan_title(plan)}", record_user=False), source="tui.plan_turn")

    async def _run_turn_for_plan_revision(self, plan: dict[str, Any], feedback: str) -> None:
        await self._run_operation(self.core_session.revise_artifact("plan", feedback), source="tui.plan_revision")

    def _resolve_goal_attempt_status(self) -> None:
        """Reflect a finished Goal attempt in its execution bubble."""
        value = self.session.get("current_goal")
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
            if self._pending_artifact_review is not None:
                self._pending_artifact_review.cancel()
            self.finish_turn(cancelled=True)
            if self.core_session is not None:
                self.run_worker(
                    self.core_session.cancel(),
                    name="cancel-core-session",
                    exclusive=False,
                )
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
            self.notify(f"Windows clipboard copy failed: {error}", severity="warning")

    def action_focus_prompt(self) -> None:
        """Focus the prompt input."""
        if not self.is_mounted:
            return
        self._sync_artifact_button()
        try:
            self.query_one(Workspace).activate("chat")
            self.query_one(PromptBox).focus()
        except NoMatches:
            return

    def _active_autocomplete_tools(self) -> list[dict[str, str]]:
        """Return the same live tool projection shown by /tools."""
        return list(self.core_session.snapshot(include_transcript=False).tools) if self.core_session else []

    def _active_autocomplete_subagents(self) -> list[dict[str, str]]:
        """Return enabled effective subagents for the shared @ surface."""
        resources = self.mode.get("resources") or {}
        items = resources.get("subagents") or []
        from mira.settings import subagent_enabled

        return [
            item
            for item in items
            if item.get("name") and subagent_enabled(self.config or {}, str(item["name"]))
        ]

    def _model_agents_available(self) -> bool:
        self._refresh_session_view()
        if self.models_available:
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

    def user_message(
        self, text: str, *, planning: bool = False, created_at: str = "",
        frontend_event: MessageEvent | None = None,
    ) -> None:
        """Write a submitted user message to the chat log."""
        self.trace.user_message(text, planning=planning)
        self.query_one(ChatLog).timestamped_user_message(
            text, planning=planning, created_at=created_at, frontend_event=frontend_event,
        )

    def system_message(self, text: str, *, kind: str = "system", created_at: str = "") -> None:
        """Write a presentation-only message to the trace and chat log."""
        self.trace.system_message(text, kind=kind)
        self.query_one(ChatLog).system_message(text, kind=kind, created_at=created_at)

    def command_output(self, renderable: Any) -> None:
        """Write command output to the chat log."""
        self.trace.command_output(renderable)
        self.waiting_finished()
        self.query_one(ChatLog).command_output(renderable)

    async def show_plan(self, interrupt: Any = None) -> str:  # noqa: ARG002
        """Render the exact current Plan without invoking or paraphrasing through a model."""
        self._refresh_session_view()
        value = self.session.get("current_plan")
        if value is None:
            if self.session.get("current_goal") is not None:
                message = "There is no current Plan. The current formal work is a Goal.\nUse /goal-show to display it."
                self.system_message(message, kind="info")
                return message
            self.system_message("no current Plan", kind="muted")
            return "No current Plan."
        self.mode["current_plan"] = value
        self.query_one(ChatLog).present_plan(value, active=True, status=value["status"])
        return "Current Plan rendered."

    async def clear_plan(self) -> None:
        """Clear current_plan without deleting historical transcript events."""
        value = await self.core_session.clear_artifact("plan")
        if value is None:
            if self.session.get("current_goal") is not None:
                self.system_message("There is no current Plan. Use /goal-clear to clear the current Goal.", kind="info")
                return
            self.system_message("no current Plan", kind="muted")
            return
        self._sync_artifact_button()
        self.system_message(f'cleared current Plan "{plan_title(value)}"', kind="muted")

    async def resume_plan(self) -> None:
        """Resume any incomplete current Plan immediately in Act mode."""
        if not self._model_agents_available():
            return
        value = self.session.get("current_plan")
        if value is None:
            self.system_message("no current Plan", kind="muted")
            return
        if value["status"] == "completed":
            self.system_message(
                "the current Plan is completed; use /plan-show or ask MIRA to show/run it again",
                kind="info",
            )
            return
        if value["status"] not in {"proposed", "active", "paused", "max_iterations_reached"}:
            self.system_message("the current Plan is not resumable", kind="warning")
            return
        started = await self.core_session.start_artifact("plan")
        if started is None:
            return
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

    async def show_goal(self, interrupt: Any = None) -> str:  # noqa: ARG002
        """Render the exact current Goal without a model call."""
        self._refresh_session_view()
        value = self.session.get("current_goal")
        if value is None:
            if self.session.get("current_plan") is not None:
                message = "There is no current Goal. The current formal work is a Plan.\nUse /plan-show to display it."
                self.system_message(message, kind="info")
                return message
            self.system_message("no current Goal", kind="muted")
            return "No current Goal."
        self.mode["current_goal"] = value
        self.query_one(ChatLog).present_goal(value, active=True, status=value["status"])
        return "Current Goal rendered."

    async def clear_goal(self) -> None:
        """Clear current_goal without deleting historical transcript events."""
        value = await self.core_session.clear_artifact("goal")
        if value is None:
            if self.session.get("current_plan") is not None:
                self.system_message("There is no current Goal. Use /plan-clear to clear the current Plan.", kind="info")
                return
            self.system_message("no current Goal", kind="muted")
            return
        self._sync_artifact_button()
        self.system_message(f'cleared current Goal "{goal_title(value)}"', kind="muted")

    async def resume_goal(self) -> None:
        """Resume any incomplete current Goal immediately in Act mode."""
        if not self._model_agents_available():
            return
        value = self.session.get("current_goal")
        if value is None:
            if self.session.get("current_plan") is not None:
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
        if value["status"] not in {"proposed", "active", "paused", "max_iterations_reached"}:
            self.system_message("the current Goal is not resumable", kind="warning")
            return
        started = await self.core_session.start_artifact("goal")
        if started is None:
            return
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
        self._refresh_session_view()
        plan = self.session.get("current_plan")
        goal = self.session.get("current_goal")
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
        self._restore_chat_viewport()
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
            self._notify_busy_action("clearing history")
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
            await self._clear_current_chat()
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
            sessions, compactions = await self._clear_all_chats()
            session_suffix = "s" if sessions != 1 else ""
            compaction_suffix = "s" if compactions != 1 else ""
            self.system_message(
                f"cleared {sessions} saved chat session{session_suffix} and "
                f"{compactions} compaction file{compaction_suffix}",
                kind="info",
            )
            return True

        if text == "/clear-uploads":
            answer = await self._prompt_choice(
                "Clear Current Uploads?",
                "Delete uploaded files for this chat? Uploads from other chats will be kept.\n\n"
                + DESTRUCTIVE_CONFIRM_HINT,
                DESTRUCTIVE_CONFIRM_CHOICES,
            )
            if answer != "o":
                self.system_message("clear uploads cancelled", kind="muted")
                return True
            await self.core_session.clear_uploads()
            self.system_message("current chat uploads cleared", kind="info")
            return True

        if text == "/clear-all-uploads":
            answer = await self._prompt_choice(
                "Clear All Uploads?",
                "Delete all uploaded files under .mira/_uploads for this workspace?\n\n"
                + DESTRUCTIVE_CONFIRM_HINT,
                DESTRUCTIVE_CONFIRM_CHOICES,
            )
            if answer != "o":
                self.system_message("clear all uploads cancelled", kind="muted")
                return True
            await self.application.clear_uploads()
            self.system_message("all uploads cleared", kind="info")
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
            reports = self.application.clear_diagnostics()
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
            self.notify("Reload already in progress.", title="MIRA is busy", severity="warning")
            return False

        if self.busy:
            await self._handle_reload_command(full_runtime=full_runtime)
            return False

        prompt = self.query_one(PromptBox)
        prompt_was_disabled = prompt.disabled
        self._reload_in_progress = True
        self._sync_interactivity()
        prompt.disabled = True
        self._set_status(state="running")
        try:
            reloaded = await self._handle_reload_command(full_runtime=full_runtime)
            if reloaded:
                self._set_status(state="ready")
            return reloaded
        except Exception as exc:
            from mira.api import ConfigError

            if isinstance(exc, ConfigError):
                self.system_message(str(exc), kind="warning")
            else:
                error_path = self._write_error_report(exc, source="tui.reload")
                self.system_message(f"reload error: {exc}\nerror report: {error_path}", kind="error")
            self._set_status(state="error")
            return False
        finally:
            self._reload_in_progress = False
            self._sync_interactivity()
            if self.is_mounted:
                prompt.disabled = prompt_was_disabled
                if not prompt.disabled:
                    self.action_focus_prompt()

    async def _handle_reload_command(self, *, full_runtime: bool = False) -> bool:
        """Reload config, resources, and agents, optionally restarting MCP."""
        if self.busy:
            self._notify_busy_action("reloading agents")
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
        snapshot = await self.application.reload(restart_mcp=reload_mcp)
        self._refresh_runtime_view()
        if self._settings_panel is not None and self._settings_panel.is_mounted:
            self._settings_panel.refresh_tracing_profiles(self.config["tracing_profiles"])
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

    async def open_context_report(self) -> None:
        if self.busy or isinstance(self.screen, ContextReportScreen):
            return
        report = await self.core_session.context_report()
        if not self.busy:
            self.push_screen(ContextReportScreen(report))

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
        self._refresh_session_view()
        if self.busy:
            self._notify_busy_action("changing settings")
            return True

        if self.application is None:
            self.config = read_configuration(self.workspace)
        settings = self.application.settings() if self.application else self.config["settings"]
        if self._settings_panel is not None and self._settings_panel.is_mounted:
            self._settings_panel.remove()
        panel = SettingsPanel(
            settings,
            tool_metadata=self._settings_tool_metadata(),
            apply_change=self._apply_settings,
            reset_change=self._reset_settings,
            configure_git=self._configure_git,
            reload_runtime=self._run_reload_runtime_command,
            close_panel=self._close_settings_panel,
            mcp_manager=self.mcp_manager,
            model_aliases=(self.config or {}).get("model_aliases", ()),
            tracing_profiles=self.config.get("tracing_profiles", ()),
            tracing_preview=self.application.tracing_preview if self.application else lambda preferences: preview_tracing(self.workspace, preferences),
            subagent_metadata=list((self.mode.get("resources") or {}).get("subagents") or ()),
            initial_tab=initial_tab,
        )
        self._settings_panel = panel
        return self.mount(panel)

    async def _reset_settings(self, tab: str, settings: dict[str, Any]) -> tuple[bool, str]:
        from ui.textual.widgets.settings_panel import SETTINGS_TAB_LABELS

        label = SETTINGS_TAB_LABELS[tab]
        answer = await self.foreground_interrupt(lambda: self._prompt_choice(
            f"Reset {label}?",
            f"Restore the default settings for {label}? Other tabs will not be changed.",
            [("r", "Reset"), ("c", "Cancel")],
        ))
        if answer != "r":
            return False, "reset cancelled"
        ok, message = await self._apply_settings(settings, reset_mcp=tab == "mcp")
        if ok and tab == "mcp" and message != "settings unchanged":
            message += "; enabled MCP servers await approval through Connect/Restart"
        return ok, message

    async def _apply_settings(self, settings: dict[str, Any], *, reset_mcp: bool = False) -> tuple[bool, str]:
        if await self._execute_enable_cancelled(self.config.get("settings") or {}, settings):
            return False, "execute remains disabled"
        if self.application is None:
            result = configure_workspace(self.workspace, settings)
            if result[0]:
                self.config = read_configuration(self.workspace)
            return (result[0], "settings saved; MIRA is not ready; Reload Runtime required") if result[0] else result
        result = await self.application.apply_settings(settings, reset_mcp=reset_mcp)
        self._refresh_runtime_view()
        self._refresh_startup_splash()
        self._sync_issues()
        if result[0] and result[1] != "settings unchanged":
            self._set_status(state="ready")
        return result

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

    async def _configure_git(
        self,
        preference: bool | None,
    ) -> tuple[bool, str, dict[str, Any]]:
        """Resolve and persist Git protection without blocking MIRA."""
        settings = (self.config or {}).get("settings") or (self.application.settings() if self.application else read_configuration(self.workspace)["settings"])
        if preference is None:
            answer = await self.foreground_interrupt(
                lambda: self._prompt_choice(
                    "Configure Git Protection?",
                    "Initialize a Git repository for this workspace?",
                    [("y", "Yes (y)"), ("n", "No (n)")],
                )
            )
            if answer is None:
                return False, "Git configuration cancelled", settings
            preference = answer == "y"

        if self.application is None:
            return configure_git(self.workspace, preference)
        result = self.application.configure_git(preference)
        self._refresh_runtime_view()
        self._sync_issues()
        return result

    def _settings_tool_metadata(self) -> list[dict[str, str]]:
        """Return loaded tool metadata for the settings panel."""
        resources = self.mode.get("resources") if isinstance(self.mode, dict) else None
        tools = resources.get("tools") if isinstance(resources, dict) else None
        return list(tools or ())

    def _close_settings_panel(self) -> None:
        """Forget the closed settings panel and return focus to the prompt."""
        self._settings_panel = None
        self.action_focus_prompt()

    async def _clear_current_chat(self) -> None:
        await self.core_session.clear()
        self._refresh_session_view(full=True)
        self._render_current_session()

    async def _clear_all_chats(self) -> tuple[int, int]:
        active_id = self.core_session.id
        counts = await self.application.clear_history()
        self._activate_core_session(await self.application.open_session(active_id))
        return counts


    def _clear_prompt_history(self) -> None:
        self.application.clear_prompt_history()
        self.query_one(PromptBox).set_history([])


    def _render_current_session(self) -> None:
        """Rebuild visible chat output from the active session."""
        chat = self.query_one(ChatLog)
        self._restore_chat_viewport()
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
        """Advance streamed task delegation without adding a transcript bubble."""
        self.waiting_finished()
        self._mark_main_stream_active()
        if self._subagent_panel_is_live():
            self._set_status(state="running")
            return
        self.query_one(ChatLog).finish_main()
        self._set_status(state="running")

    def tool_call(
        self, name: str, args: Any, call_id: str = "", *,
        created_at: str = "", frontend_event: ToolEvent | None = None,
    ) -> None:
        """Render a tool call in transcript order."""
        self.trace.tool_call(name, args)
        self._finish_main_stream_activity()
        self.waiting_finished()
        self.query_one(ChatLog).tool_call(
            name,
            args,
            call_id=call_id,
            created_at=created_at,
            origin_event_id=(
                frontend_event.origin_event_id if frontend_event is not None
                else 0
            ),
        )
        self._rearm_waiting_if_busy()

    def _current_subagent_runs(self) -> Any:
        """Return normalized run state owned by the current persisted session."""
        return self.core_session.inspections() if self.core_session else []

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

    def tool_error(
        self, name: str, error: str, call_id: str = "", *,
        created_at: str = "", duration_ms: int | None = None,
    ) -> None:
        """Render an explicitly failed tool result with its original call."""
        self.trace.recovered_tool_error(name, error)
        self._finish_main_stream_activity()
        self.waiting_finished()
        self.query_one(ChatLog).tool_error(
            name, error, call_id=call_id, created_at=created_at, duration_ms=duration_ms,
        )
        self._rearm_waiting_if_busy()

    def inspection_updated(self, event: RuntimeEvent) -> None:
        """Route public incremental observations only to the selected session."""
        if self.core_session is None or event.session_id != self.core_session.id:
            return
        detail = event.detail
        self.query_one(Inspector).update_inspection(detail["inspection_id"], detail["update"])


    def session_state_changed(self, event: RuntimeEvent) -> None:
        """Project Core session notifications without clearing sticky UI errors."""
        if event.session_id != str(self.session.get("id") or ""):
            return
        state = {"turn_started": "running", "cancelling": "cancelling", "ready": "ready"}.get(event.state)
        if state is not None and not (state == "ready" and self.status_state == "error"):
            self._set_status(state=state)

    def stop_active_tools(self, status: str) -> None:
        self.query_one(ChatLog).stop_live_tools(status)

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
        """Record task delegation without adding a transcript bubble."""
        self.trace.delegation_started(calls)
        self._finish_main_stream_activity()
        self.waiting_finished()
        if self._subagent_panel_is_live():
            self._set_status(state="running")
            self._rearm_waiting_if_busy()
            return
        self.query_one(ChatLog).finish_main()
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

    def workflow_started(
        self,
        workflow_id: str = "",
        workflow_name: str = "",
    ) -> None:
        """Mount one live Workflow execution tree in the transcript."""
        self.query_one(ChatLog).workflow_started(workflow_id, workflow_name)

    def workflow_task_started(
        self,
        task_id: str,
        name: str,
        step: int,
        input_state: Any,
        *,
        workflow_id: str = "",
    ) -> None:
        """Add one native root task beneath its inferred execution batch."""
        view = self.query_one(ChatLog).workflow_task_started(
            workflow_id,
            task_id,
            name,
            step,
            input_state,
        )
        self._refresh_workflow_node_inspector(view)

    def workflow_task_waiting(
        self,
        task_id: str,
        _name: str,
        _step: int,
        *,
        workflow_id: str = "",
    ) -> None:
        view = self.query_one(ChatLog).workflow_task_waiting(workflow_id, task_id)
        self._refresh_workflow_node_inspector(view)

    def workflow_task_resumed(
        self,
        task_id: str,
        _name: str,
        _step: int,
        *,
        workflow_id: str = "",
    ) -> None:
        view = self.query_one(ChatLog).workflow_task_resumed(workflow_id, task_id)
        self._refresh_workflow_node_inspector(view)

    def workflow_agent_started(
        self,
        task_id: str,
        name: str,
        inspection_id: str,
        *,
        task_input: str = "",
        resumed: bool = False,
        workflow_id: str = "",
    ) -> None:
        """Attach one observed agent leaf to its owning native task."""
        self.query_one(ChatLog).workflow_agent_started(
            workflow_id,
            task_id,
            inspection_id,
            name,
            task_input,
            resumed=resumed,
        )
        self._refresh_workflow_node_inspector_by_id(workflow_id, task_id)

    def workflow_agent_waiting(
        self,
        task_id: str,
        _name: str,
        inspection_id: str,
        *,
        workflow_id: str = "",
    ) -> None:
        self.query_one(ChatLog).workflow_agent_waiting(workflow_id, inspection_id)
        self._refresh_workflow_node_inspector_by_id(workflow_id, task_id)

    def workflow_agent_finished(
        self,
        task_id: str,
        _name: str,
        inspection_id: str,
        *,
        status: str = "DONE",
        result: str = "",
        error: str = "",
        workflow_id: str = "",
    ) -> None:
        self.query_one(ChatLog).workflow_agent_finished(
            workflow_id,
            inspection_id,
            status=status,
            result=result,
            error=error,
        )
        self._refresh_workflow_node_inspector_by_id(workflow_id, task_id)

    def workflow_task_finished(
        self,
        task_id: str,
        _name: str,
        _step: int,
        *,
        status: str = "DONE",
        error: str = "",
        result: Any = None,
        result_available: bool = False,
        workflow_id: str = "",
    ) -> None:
        """Freeze one native Workflow tree node at its terminal status."""
        view = self.query_one(ChatLog).workflow_task_finished(
            workflow_id,
            task_id,
            status=status,
            error=error,
            result=result,
            result_available=result_available,
        )
        self._refresh_workflow_node_inspector(view)

    def workflow_finished(
        self,
        workflow_id: str = "",
        *,
        final_state: Any = None,
        final_state_available: bool = False,
    ) -> None:
        """Complete the existing tree and expose its final state footer."""
        self.query_one(ChatLog).workflow_finished(
            workflow_id,
            final_state,
            available=final_state_available,
        )

    def workflow_cancelled(self, workflow_id: str = "", *, error: str = "") -> None:
        """Freeze active Workflow tree rows after cancellation or failure."""
        self.query_one(ChatLog).workflow_cancelled(workflow_id, error)

    def _refresh_workflow_node_inspector(self, view: Any) -> None:
        if view is None:
            return
        inspector = self.query_one(Inspector)
        if inspector.workflow_task_id == view.task_id:
            inspector.open_workflow_node(view)

    def _refresh_workflow_node_inspector_by_id(
        self,
        workflow_id: str,
        task_id: str,
    ) -> None:
        view = self.query_one(ChatLog).workflow_task_view(workflow_id, task_id)
        self._refresh_workflow_node_inspector(view)

    @on(SubagentSelected)
    def open_subagent_inspector(self, event: SubagentSelected) -> None:
        """Replace the chat viewport with the selected live or historical transcript."""
        event.stop()
        inspector = self.query_one(Inspector)
        try:
            if not inspector.open(event.run_id or event.inspection_id):
                raise ValueError("the selected inspection is no longer available")
        except Exception as exc:
            self.system_message(f"subagent history unavailable: {exc}", kind="error")
            return
        self._show_inspector(inspector)

    @on(RubricInspectionSelected)
    def open_rubric_inspector(self, event: RubricInspectionSelected) -> None:
        """Prefer one live Rubric transcript, then its durable snapshot."""
        event.stop()
        inspector = self.query_one(Inspector)
        try:
            if not inspector.open(event.inspection_id):
                raise ValueError("the selected inspection is no longer available")
        except Exception as exc:
            self.system_message(f"Rubric history unavailable: {exc}", kind="error")
            return
        self._show_inspector(inspector)

    @on(WorkflowFinalStateSelected)
    def open_workflow_final_state(self, event: WorkflowFinalStateSelected) -> None:
        """Open one retained Workflow result in the shared Inspector viewport."""
        event.stop()
        inspector = self.query_one(Inspector)
        inspector.open_workflow_state(
            event.workflow_name,
            event.final_state,
            event.snapshot,
        )
        self._show_workflow_inspector(inspector)

    @on(WorkflowNodeSelected)
    def open_workflow_node(self, event: WorkflowNodeSelected) -> None:
        """Open one live Workflow node summary without changing its tree state."""
        event.stop()
        inspector = self.query_one(Inspector)
        inspector.open_workflow_node(event.view)
        self._show_workflow_inspector(inspector)

    @on(WorkflowAgentSelected)
    def open_workflow_agent(self, event: WorkflowAgentSelected) -> None:
        """Open a Workflow-owned agent in the ordinary full Inspector."""
        event.stop()
        inspector = self.query_one(Inspector)
        try:
            if not inspector.open(event.inspection_id):
                raise ValueError("the selected inspection is no longer available")
        except Exception as exc:
            self.system_message(f"Workflow history unavailable: {exc}", kind="error")
            return
        self._show_workflow_inspector(inspector)

    def _show_inspector(self, inspector: Inspector) -> None:
        """Reuse the established viewport and focus transition for live inspection."""
        self.run_worker(self._open_inspector_tab(inspector.query_one("#inspector-log", ChatLog)), name="open-inspector")

    async def _open_inspector_tab(self, focus_target: Any) -> None:
        await self.query_one(Workspace).open_inspector()
        self.call_after_refresh(focus_target.focus)

    def _show_workflow_inspector(self, inspector: Inspector) -> None:
        """Use the retained workspace tab for Workflow inspection."""
        self.run_worker(self._open_inspector_tab(inspector), name="open-inspector")

    @on(SubagentHistoryAnchor.Requested)
    def open_subagent_history(self, event: SubagentHistoryAnchor.Requested) -> None:
        """Reload one owning tool's durable children into the existing panel."""
        event.stop()
        try:
            runs = self.core_session.inspections(origin_event_id=event.origin_event_id)
            if not runs:
                raise ValueError("no persisted subagent runs remain for this tool")
        except Exception as exc:
            self.system_message(f"subagent history unavailable: {exc}", kind="error")
            return
        self.query_one(SubagentsPanel).restore_records(runs)

    @on(Inspector.Closed)
    def close_inspector(self, event: Inspector.Closed) -> None:
        """Restore the normal transcript while preserving surrounding panels."""
        event.stop()
        self._restore_chat_viewport()
        self.call_after_refresh(self.action_focus_prompt)

    def _restore_chat_viewport(self) -> None:
        try:
            workspace = self.query_one(Workspace)
        except NoMatches:
            return
        workspace.inspector.stop_inspection()
        workspace.activate("chat")
        if "inspector" in workspace.views:
            # The subscription was detached above. A newer inspection may be
            # opened before this queued tab removal finishes.
            self.run_worker(workspace.close_view("inspector", stop_inspection=False), name="close-inspector")

    async def on_key(self, event: Key) -> None:
        """Apply Escape to the currently focused transcript or prompt."""
        settings = self._settings_panel
        if (
            settings is not None and settings.is_mounted and settings.display
            and self.screen.focused is None and event.key in {"escape", "q"}
        ):
            event.prevent_default()
            await settings.on_key(event)
            return
        if event.key != "escape":
            return
        try:
            inspector = self.query_one(Inspector)
            prompt = self.query_one(PromptBox)
            workspace = self.query_one(Workspace)
        except NoMatches:
            return
        if "inspector" in workspace.views and inspector.display and (
            inspector.has_focus or inspector.has_focus_within
        ):
            event.stop()
            event.prevent_default()
            self._restore_chat_viewport()
            self.action_focus_prompt()
            return
        if prompt.has_focus and prompt.value:
            event.stop()
            event.prevent_default()
            prompt.value = ""

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
        inspection_id: str = "",
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
            inspection_id=inspection_id,
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
        inspection_id: str = "",
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
            inspection_id=inspection_id,
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
        phase: str = "verifying",
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
            phase=phase,
        )

    def rubric_lifecycle_event(self, event: dict[str, Any]) -> None:
        """Keep nested verifier activity inside the active Rubric bubble."""
        self.query_one(ChatLog).rubric_lifecycle_event(event)

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
        self._rearm_waiting_if_busy()

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
        self._sync_active_agent_pair()
        self.trace.flush_all()
        self._finish_main_stream_activity()
        self.waiting_finished()
        self._waiting_label = "working..."
        self.query_one(ChatLog).finish_turn(cancelled=cancelled, tool_status=tool_status)
        if cancelled:
            self.query_one(SubagentsPanel).cancel_running()
            self._subagent_live_active = False

    def _sync_active_agent_pair(self) -> None:
        self._refresh_session_view()

    def usage_updated(self) -> None:
        """Refresh the status bar after token usage is committed."""
        self._set_status(state=self.status_state)

    async def ask_approvals(self, interrupts: list[Any]) -> list[dict[str, Any]]:
        """Ask the user to approve, edit, or reject interrupted actions."""
        return await self.foreground_interrupt(lambda: self._ask_approvals(interrupts))

    async def _ask_approvals(self, interrupts: list[Any]) -> list[dict[str, Any]]:
        """Resolve one serialized group of tool approvals."""
        self.waiting_finished()
        decisions: list[dict[str, Any]] = []
        for interrupt in interrupts:
            for index, action in enumerate(action_requests(interrupt)):
                while True:
                    answer = await self._prompt_choice(
                        action_title(action),
                        action_preview(action),
                        action_choices(interrupt, action, index),
                    )
                    if answer is None:
                        continue
                    if answer == "e":
                        decision = await self.edit_decision(action)
                        if decision is None:
                            continue
                        decisions.append(decision)
                    elif answer == "r":
                        decisions.append({"type": "reject_once"})
                    elif answer == "l":
                        decisions.append({"type": "allow_always"})
                    else:
                        decisions.append({"type": "allow_once"})
                    break
        return decisions

    async def ask_user(self, interrupt: Any) -> str:
        """Ask the user for a concrete next-step choice from an ask_user interrupt."""
        return await self.foreground_interrupt(lambda: self._ask_user(interrupt))

    async def _ask_user(self, interrupt: Any) -> str:
        """Resolve one serialized ask_user interaction."""
        self.waiting_finished()
        request = ask_user_request(interrupt)
        question = ask_user_question(request)
        options = ask_user_options(request)
        choices = [(str(index), f"{index} {option}") for index, option in enumerate(options, start=1)]
        while True:
            answer = await self._prompt_choice("Question", question, choices, vertical=True)
            if answer is None:
                continue
            selected = (
                options[int(answer) - 1]
                if answer.isdigit() and 0 < int(answer) <= len(options)
                else None
            )
            if selected is None:
                continue
            if selected != ASK_USER_OPEN_OPTION:
                return selected
            response = await self._prompt_text("Question", ASK_USER_OPEN_OPTION)
            if response is None:
                continue
            result = response.strip()
            if result:
                return result

    async def answer_mcp_elicitation(self, interrupt: Any) -> dict[str, Any]:
        """Collect native MCP form or URL answers through the shared prompt panel."""
        return await self.foreground_interrupt(
            lambda: self._answer_mcp_elicitation(interrupt)
        )

    async def _answer_mcp_elicitation(self, interrupt: Any) -> dict[str, Any]:
        """Resolve one serialized MCP elicitation interaction."""
        self.waiting_finished()
        payload = mcp_elicitation_request(interrupt)
        responses: dict[str, dict[str, Any]] = {}
        for request in payload.get("requests", []):
            if not isinstance(request, dict):
                continue
            while True:
                if request.get("mode") == "form":
                    schema = request.get("requested_schema")
                    response = await self._prompt_mcp_form(
                        str(request.get("message") or "MCP input required"),
                        schema if isinstance(schema, dict) else {},
                    )
                    if response is None:
                        continue
                    responses[str(request.get("key") or "")] = response
                    break
                action = await self._prompt_choice(
                    "MCP Input",
                    "\n\n".join(
                        part
                        for part in (
                            str(request.get("message") or "MCP input required"),
                            mcp_elicitation_preview(request),
                        )
                        if part
                    ),
                    [("a", "Accept"), ("d", "Decline"), ("c", "Cancel tool")],
                    vertical=True,
                )
                if action is None:
                    continue
                response: dict[str, Any] = {
                    "action": {"a": "accept", "d": "decline"}.get(action, "cancel")
                }
                responses[str(request.get("key") or "")] = response
                break
        return {"responses": responses}

    async def edit_decision(self, action: Any) -> dict[str, Any] | None:
        """Prompt for edited JSON args and return a LangGraph decision."""
        if not isinstance(action, dict):
            return {"type": "reject_once"}

        edited_text = await self._prompt_json("Edited Args", json.dumps(action.get("args", {}), indent=2))
        if edited_text is None:
            return None

        try:
            edited_args = json.loads(edited_text)
        except json.JSONDecodeError:
            self.system_message("invalid JSON; rejecting action", kind="warning")
            return {"type": "reject_once"}

        if not isinstance(edited_args, dict):
            self.system_message("edited args must be a JSON object; rejecting action", kind="warning")
            return {"type": "reject_once"}

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
        return await self._with_prompt_lock(
            lambda: self.query_one(PromptPanel).choose(
                title,
                message,
                choices,
                vertical=vertical,
            )
        )

    async def _prompt_text(
        self,
        title: str,
        message: str,
        *,
        foreground_owned: bool = False,
    ) -> str | None:
        """Show a text prompt in the main window."""
        if foreground_owned:
            return await self._wait_with_prompt_disabled(
                lambda: self.query_one(PromptPanel).ask_text(title, message)
            )
        return await self._with_prompt_lock(
            lambda: self.query_one(PromptPanel).ask_text(title, message)
        )

    async def _prompt_json(self, title: str, text: str) -> str | None:
        """Show a JSON editor prompt in the main window."""
        return await self._with_prompt_lock(
            lambda: self.query_one(PromptPanel).edit_json(title, text)
        )

    async def _prompt_mcp_form(
        self,
        message: str,
        schema: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Show a schema-driven MCP form in the shared prompt area."""
        return await self._with_prompt_lock(
            lambda: self.query_one(PromptPanel).ask_mcp_form(
                "MCP Input",
                message,
                schema,
            )
        )

    async def _with_prompt_lock(self, prompt_waiter: Callable[[], Awaitable[Any]]) -> Any:
        """Disable the prompt box while an in-window prompt is active."""
        return await self.foreground_interrupt(
            lambda: self._wait_with_prompt_disabled(prompt_waiter)
        )

    async def _wait_with_prompt_disabled(
        self,
        prompt_waiter: Callable[[], Awaitable[Any]],
    ) -> Any:
        """Wait for one prompt while preserving its prior input availability."""
        self.query_one(Workspace).activate("chat")
        prompt = self.query_one(PromptBox)
        was_disabled = prompt.disabled
        prompt.disabled = True
        try:
            return await prompt_waiter()
        finally:
            prompt.disabled = was_disabled
            if self.is_mounted and self.ready and not self.busy and not prompt.disabled:
                self.action_focus_prompt()

    async def foreground_interrupt(self, waiter: Callable[[], Awaitable[Any]]) -> Any:
        """Serialize user-action requests and temporarily suspend foreground UI."""
        task = asyncio.current_task()
        if task is not None and task is self._foreground_interrupt_owner:
            return await waiter()

        async with self._foreground_interrupt_lock:
            self._foreground_interrupt_owner = task
            focused = self.screen.focused if self.is_mounted else None
            settings = self._settings_panel
            settings_visible = bool(
                settings is not None and settings.display
            )
            modal = self.screen if self.is_mounted and len(self.screen_stack) > 1 else None
            if settings_visible:
                settings.display = False
            if modal is not None:
                await self.pop_screen()
            try:
                return await waiter()
            finally:
                try:
                    if modal is not None and self.is_mounted:
                        await self.push_screen(modal)
                    if settings_visible and settings is not None and settings.parent is not None:
                        settings.display = True
                    if focused is not None and focused.is_mounted and not focused.disabled:
                        focused.focus()
                finally:
                    self._foreground_interrupt_owner = None

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

    def mcp_activity(self, phase: str, detail: Any) -> None:
        """Project one manager-owned MCP lifecycle batch into the chat."""
        del phase
        if not isinstance(detail, dict):
            return
        batch_id = str(detail.get("batch_id") or "")
        if not batch_id:
            return
        snapshot = dict(detail)
        self._mcp_activity_snapshots[batch_id] = snapshot
        if batch_id in self._mcp_activity_revealed:
            if self.is_mounted:
                self.query_one(ChatLog).show_mcp_activity(snapshot)
            if snapshot.get("finished"):
                self._mcp_activity_revealed.discard(batch_id)
                self._mcp_activity_snapshots.pop(batch_id, None)
            return
        if snapshot.get("finished"):
            task = self._mcp_activity_tasks.pop(batch_id, None)
            if task is not None:
                task.cancel()
            self._mcp_activity_snapshots.pop(batch_id, None)
            return
        if batch_id not in self._mcp_activity_tasks:
            self._mcp_activity_tasks[batch_id] = asyncio.create_task(
                self._reveal_mcp_activity(batch_id),
                name=f"mcp-activity-{batch_id}",
            )

    async def _reveal_mcp_activity(self, batch_id: str) -> None:
        try:
            await asyncio.sleep(self._mcp_reveal_delay_seconds)
            snapshot = self._mcp_activity_snapshots.get(batch_id)
            if snapshot is None or snapshot.get("finished") or not self.is_mounted:
                return
            self._mcp_activity_revealed.add(batch_id)
            self.query_one(ChatLog).show_mcp_activity(snapshot)
        except asyncio.CancelledError:
            return
        finally:
            self._mcp_activity_tasks.pop(batch_id, None)

    def _set_status(self, *, state: str) -> None:
        """Update the status bar if it has been mounted."""
        if state not in OPERATIONAL_STATES:
            state = "running" if self.busy else "starting" if not self.ready else "ready"
        self.status_state = state
        if not self.is_mounted:
            return
        self._refresh_session_view()
        self.query_one(StatusBar).set_state(
            mode=self._mode_label(),
            model_name=self.model_name or "loading",
            state=state,
            dashboard=self.session.get("dashboard"),
            turns=int(self.session.get("turns") or 0),
            busy=self.busy,
        )
        self.query_one(TelemetryBar).set_state(
            model_name=self.effective_model_alias,
            btw_name="" if "btw" in self.query_one(Workspace).views else None,
            dashboard=self.session.get("dashboard"),
            turns=int(self.session.get("turns") or 0),
        )
        from mira.settings import MAIN_MODEL, model_assignment
        self.query_one("#actions-main", ComposerActions).set_alias(model_assignment(self.config, MAIN_MODEL))
        self._sync_artifact_button()
        self._sync_mcp_button()
        self._sync_issues_button()
        self._sync_interactivity()

    def watch_busy(self) -> None:
        """Keep every state-mutating control aligned with runtime activity."""
        self._sync_interactivity()

    def _sync_interactivity(self) -> None:
        """Central projection of whether runtime/session mutation is available."""
        if not self.is_mounted or not self._screen_stack:
            return
        locked = bool(self.busy or getattr(self, "_reload_in_progress", False))
        for selector in ("#model-trigger-main", "#new-chat"):
            try:
                self.query_one(selector, Button).disabled = locked
            except NoMatches:
                pass
        try:
            self.query_one("#add-file-button", Button).disabled = bool(
                locked
                or not self.local_file_uploads_enabled
                or not self.ready
                or self.core_session is None
                or self._file_ingestion_active
            )
        except NoMatches:
            pass
        try:
            self.query_one(SessionHistory).disabled = locked
        except NoMatches:
            pass
        try:
            self.query_one(ContextStatus).disabled = locked
        except NoMatches:
            pass
        settings_panel = getattr(self, "_settings_panel", None)
        if settings_panel is not None:
            settings_panel.set_runtime_busy(locked)
            if settings_panel.is_mounted:
                self.call_after_refresh(settings_panel.set_runtime_busy, locked)
        self._sync_artifact_button()
        self._sync_artifact_actions()

    def _sync_artifact_actions(self) -> None:
        """Disable retained artifact mutations while keeping active reviews usable."""
        pending = self._pending_artifact_review
        buttons = [*self.query(".plan-action"), *self.query(".goal-action")]
        if not self.busy:
            for button in buttons:
                previous = self._artifact_action_disabled.pop(button, None)
                if previous is not None:
                    button.disabled = previous
            self._artifact_action_disabled = {
                button: disabled
                for button, disabled in self._artifact_action_disabled.items()
                if button.is_mounted
            }
            return
        for button in buttons:
            pending_action = bool(
                pending is not None
                and (button.id or "").endswith(f"-{pending.artifact.get('id', '')}")
            )
            if pending_action:
                previous = self._artifact_action_disabled.pop(button, None)
                if previous is not None:
                    button.disabled = previous
                continue
            self._artifact_action_disabled.setdefault(button, button.disabled)
            button.disabled = True

    def _notify_busy_action(self, action: str) -> None:
        """Report an unavailable mutation without changing the transcript."""
        self.notify(
            f"Finish the current turn before {action}.",
            title="MIRA is busy",
            severity="warning",
        )

    def _set_ready_unless_error(self) -> None:
        """Return passive work to Ready without clearing a sticky Error."""
        if not self.busy and self.status_state != "error":
            self._set_status(state="ready")

    @on(Button.Pressed, "#mcp-status-button")
    def press_mcp_status(self, event: Button.Pressed) -> None:
        event.stop()
        self.open_mcp_panel()

    @on(ContextStatus.Pressed)
    async def press_context_status(self, event: ContextStatus.Pressed) -> None:
        """Open the same Context Report modal used by the slash command."""
        event.stop()
        if self.busy:
            return
        await self.open_context_report()

    @on(Button.Pressed, "#model-trigger-main")
    def press_model_selector(self, event: Button.Pressed) -> None:
        """Choose an alias using the same Settings application/rebuild path."""
        from mira.settings import MAIN_MODEL, model_assignment

        event.stop()
        if self.busy or self._reload_in_progress or not self.ready:
            return
        aliases = tuple((self.config or {}).get("model_aliases", ()))
        if not aliases:
            self.notify("No model aliases are available. Configure profiles with /models.")
            return
        button = event.button
        pane = self.query_one("#primary-pane").region
        self.push_screen(ModelAliasPopup(
            "main", aliases, model_assignment(self.config, MAIN_MODEL) or "",
            (button.region.right, button.region.y), (pane.x, pane.y, pane.right, pane.bottom), tuple(self.size),
        ), self._model_alias_selected)

    def _model_alias_selected(self, alias: str | None) -> None:
        if alias is not None:
            self.run_worker(self._activate_model_alias(alias), name="select-main-model")

    async def _activate_model_alias(self, alias: str) -> None:
        from mira.settings import MAIN_MODEL, set_model_assignment
        if self.busy or self._reload_in_progress or not self.ready:
            return
        self._reload_in_progress = True
        self._sync_interactivity()
        prompt = self.query_one(PromptBox)
        disabled = prompt.disabled
        prompt.disabled = True
        try:
            settings = (self.config or {}).get("settings") or (self.application.settings() if self.application else read_configuration(self.workspace)["settings"])
            ok, message = await self._apply_settings(set_model_assignment(settings, MAIN_MODEL, alias))
            if not ok:
                self.notify(message, severity="error")
        finally:
            self._reload_in_progress = False
            prompt.disabled = disabled
            self._sync_interactivity()
            self._set_status(state=self.status_state)
            self.action_focus_prompt()

    @on(Workspace.ViewsChanged)
    def workspace_views_changed(self, event: Workspace.ViewsChanged) -> None:
        event.stop()
        self._sync_sidebar_visibility()
        self._set_status(state=self.status_state)

    def _record_effective_model(self) -> None:
        self._refresh_session_view()

    @on(Button.Pressed, "#add-file-button")
    def press_add_file(self, event: Button.Pressed) -> None:
        """Open the native multi-file picker without blocking Textual."""
        event.stop()
        if self.busy or self._reload_in_progress or not self.local_file_uploads_enabled or self._file_ingestion_active or not self.ready or self.core_session is None:
            return
        self._file_ingestion_active = True
        self._sync_interactivity()
        self.run_worker(
            self._choose_and_ingest_local_files(),
            name="add-local-files",
            exclusive=False,
        )

    @on(PromptBox.LocalFilesPasted)
    def ingest_pasted_local_files(self, event: PromptBox.LocalFilesPasted) -> None:
        """Route Explorer clipboard files through the shared ingestion path."""
        event.stop()
        if self.busy or self._reload_in_progress or not self.local_file_uploads_enabled or self._file_ingestion_active or not self.ready or self.core_session is None:
            return
        self._file_ingestion_active = True
        self._sync_interactivity()
        self.run_worker(
            self._ingest_local_files_and_restore_focus(event.paths),
            name="paste-local-files",
            exclusive=False,
        )

    async def _choose_and_ingest_local_files(self) -> None:
        """Choose local files, remember the directory, and upload the selection."""
        try:
            paths = await asyncio.to_thread(
                choose_local_files,
                self._last_file_picker_directory,
            )
            if not paths:
                return
            self._last_file_picker_directory = paths[0].parent
            await self._ingest_local_file_paths(paths)
        except Exception as exc:
            self.system_message(f"could not add files: {exc}", kind="warning")
        finally:
            self._finish_file_ingestion()

    async def _ingest_local_files_and_restore_focus(self, paths: list[Path]) -> None:
        """Upload an already selected set of paths and restore composer focus."""
        try:
            await self._ingest_local_file_paths(paths)
        except Exception as exc:
            self.system_message(f"could not add files: {exc}", kind="warning")
        finally:
            self._finish_file_ingestion()

    async def _ingest_local_file_paths(self, paths: list[Path]) -> None:
        """Upload local files through the current project backend and insert refs."""
        references, failures = await self.core_session.upload_files(paths)
        if references:
            self.query_one(PromptBox).insert_file_references(references)
        if failures:
            self.system_message(
                "Could not add " + "; ".join(failures),
                kind="warning",
            )

    def _finish_file_ingestion(self) -> None:
        """Restore local-file controls and prompt focus after one ingestion."""
        self._file_ingestion_active = False
        self._sync_interactivity()
        self.action_focus_prompt()

    def open_mcp_panel(self) -> None:
        """Shared button and slash-command pathway for the MCP panel."""
        if self.mcp_manager is None or not getattr(self.mcp_manager, "show_status", False):
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
        if self.session.get("current_plan") is not None:
            await self.show_plan()
        elif self.session.get("current_goal") is not None:
            await self.show_goal()
        self.action_focus_prompt()

    def _sync_artifact_button(self) -> None:
        """Project the canonical retained artifact into one header control."""
        try:
            button = self.query_one("#artifact-status-button", Button)
        except NoMatches:
            return
        plan = self.session.get("current_plan")
        goal = self.session.get("current_goal")
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
        visible = bool(self.mcp_manager is not None and getattr(self.mcp_manager, "show_status", False))
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
            controls = (
                (
                    self.query_one("#artifact-status-button", Button),
                    self.session.get("current_plan") is not None or self.session.get("current_goal") is not None,
                ),
                (
                    self.query_one("#mcp-status-button", Button),
                    bool(self.mcp_manager is not None and getattr(self.mcp_manager, "show_status", False)),
                ),
                (self.query_one("#issues-button", Button), bool(self.issues)),
            )
        except NoMatches:
            return
        slots = max(0, (self.size.width - 40) // 13)
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
            self._notify_busy_action("switching sessions")
            return
        self.run_worker(self._load_session(item.session_id), name="load-session", exclusive=True)

    async def _load_session(self, session_id: str) -> None:
        """Load and install a selected session without rebuilding shared resources."""
        if self.application is None:
            self.system_message("session switching needs the core application", kind="warning")
            return

        prompt = self.query_one(PromptBox)
        prompt.disabled = True
        self.busy = True
        self._set_status(state="running")
        try:
            core_session = await self.application.open_session(session_id, resume=True)
            self._activate_core_session(core_session)
            await self._recover_open_session()
            self._set_status(state="ready")
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


    async def approve_mcp_server(self, state: Any, preview: str) -> str:
        """Use MIRA's existing modal choice interaction for server trust."""
        return await self.foreground_interrupt(
            lambda: self._approve_mcp_server(state, preview)
        )

    async def _approve_mcp_server(self, state: Any, preview: str) -> str:
        """Resolve one serialized MCP server trust choice."""
        answer = await self._prompt_choice(
            f"Allow MCP Server: {state.name}",
            preview,
            [("a", "Allow once (a)"), ("r", "Reject (r)"), ("l", "Always allow (l)")],
        )
        return {"a": "allow", "l": "always_allow"}.get(answer, "deny")

    async def on_unmount(self) -> None:
        for task in self._mcp_activity_tasks.values():
            task.cancel()
        self._mcp_activity_tasks.clear()
        if self.application is not None:
            await self.application.shutdown(persist_sessions=False)


    def _refresh_sessions(self) -> None:
        if self.application is not None and self.is_mounted:
            self.query_one(SessionHistory).refresh_sessions(self.application.list_sessions(), current_id=str(self.session.get("id") or ""))

    def _tick_status(self) -> None:
        """Refresh the clock and dashboard line."""
        if not self.ready or not self.is_mounted:
            return
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
        chat.tick_mcp_activity()
        chat.tick_subagents()
        chat.tick_compaction()
        chat.tick_rubrics()
        chat.tick_tools()
        chat.tick_workflows()
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

    def on_resize(self, _event: Any = None) -> None:
        if self.is_mounted:
            self._sync_sidebar_visibility()
            self.run_worker(self._resize_workspace(), group="workspace-resize", exclusive=False)

    async def _resize_workspace(self) -> None:
        await self.query_one(Workspace).responsive()
        self._sync_sidebar_visibility()

    def _sync_sidebar_visibility(self) -> None:
        if not self.is_mounted:
            return
        workspace = self.query_one(Workspace)
        show = not self.history_collapsed and self.size.width >= 85
        if workspace.keys("right") and self.size.width < 145:
            show = False
        sidebar = self.query_one("#session-sidebar")
        sidebar.display = show
        available = self.size.width - 2 - 10 - 1 - 28
        reserve = 37 if workspace.keys("right") else 0
        sidebar.styles.width = max(36, min(self.sidebar_width, 66, available - reserve))
        self.query_one("#side-divider").display = show
        self.query_one("#session-sidebar-title", Static).update("HISTORY" if self.sidebar_kind == "history" else "EXPLORER")
        self.query_one("#sidebar-content", ContentSwitcher).current = "history-panel" if self.sidebar_kind == "history" else "files-panel"
        self.query_one("#new-chat").display = self.sidebar_kind == "history"
        for key, kind in (("rail-chat", "history"), ("rail-files", "files")):
            self.query_one("#" + key).set_class(show and self.sidebar_kind == kind, "selected")
        workspace.sync_layout()
        self._sync_header_control_visibility()

    def begin_divider_drag(self) -> None:
        self._selection_before_drag = self.ALLOW_SELECT
        self.ALLOW_SELECT = False
        self.clear_selection()

    def end_divider_drag(self) -> None:
        if self._selection_before_drag is not None:
            self.ALLOW_SELECT = self._selection_before_drag
            self._selection_before_drag = None
        self.clear_selection()

    def drag_width(self, which: str, screen_x: int) -> None:
        workspace = self.query_one(Workspace)
        if which == "sidebar":
            origin = self.query_one("#session-sidebar").region.x
            reserve = workspace.query_one("#right-pane").region.width + 1 if workspace.dock_visible else 0
            capacity = self.query_one("#app-shell").content_region.width - 10 - 1 - reserve - 28
            self.sidebar_width = max(36, min(66, capacity, screen_x - origin))
        else:
            capacity = workspace.region.width - 29
            workspace.right_width = max(36, min(76, capacity, workspace.region.right - screen_x - 1))
        self._sync_sidebar_visibility()

    def _mode_label(self) -> str:
        """Return the compact mode label shown in the status line."""
        return "PLAN" if self.mode.get("planning") else "ACT"

    def _record_prompt_history(self, text: str) -> None:
        """Remember submitted prompt text and persist it for normal app runs."""
        self.query_one(PromptBox).remember(text)
        if not self.persist_prompt_history or self.application is None:
            return
        try:
            self.application.record_prompt(text)
        except OSError as exc:
            self.system_message(f"could not update prompt history: {exc}", kind="warning")


def plan_title(plan: dict[str, Any]) -> str:
    """Return a compact plan title for status text."""
    return str(plan.get("title") or "Implementation Plan")


def goal_title(goal: dict[str, Any]) -> str:
    """Return a compact Goal title for status text."""
    return str(goal.get("title") or "Goal")
