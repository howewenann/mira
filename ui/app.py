"""Textual application shell for MIRA."""

from __future__ import annotations

import asyncio
import json
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

from agent.context_overflow import context_notice_rendered, pop_context_overflow_notice
from config.settings import (
    EXECUTE_TOOL,
    git_protection_enabled,
    load_settings,
    save_settings,
    tool_enabled,
)
from runtime.compaction_filter import is_compaction_reasoning, is_compaction_reasoning_fragment
from session.dashboard import ensure_dashboard, normalize_dashboard, update_duration
from session.context import append_event
from session.recorder import update_plan_event_status
from ui.interrupts import (
    ASK_USER_OPEN_OPTION,
    action_choices,
    action_preview,
    action_requests,
    action_title,
    ask_user_options,
    ask_user_question,
    ask_user_request,
    plan_request,
)
from ui.repl import handle_command, initial_mode, plan_revision_text, plan_thread_id, refresh_agent_specs, run_user_turn
from ui.widgets import ChatLog, PromptBox, PromptPanel, SessionHistory, SettingsPanel, StatusBar
from ui.widgets.chat_log import DEFAULT_TOOL_OUTPUT_CHARS
from ui.widgets.session_history import SessionItem

Bootstrap = Callable[[Path, str | None, bool, dict[str, Any] | None, Any | None], Awaitable[dict[str, Any]]]
GitGuard = Callable[[Path, Any], Any]
DESTRUCTIVE_HISTORY_COMMANDS = {"/clear-chat", "/clear-all-chats", "/clear-prompts"}
DESTRUCTIVE_CONFIRM_HINT = "Press O to confirm, C or Esc to cancel."
DESTRUCTIVE_CONFIRM_CHOICES = [("o", "OK (o)"), ("c", "Cancel (c)")]


class MiraApp(App[None]):
    """Textual-first interactive MIRA UI."""

    CSS_PATH = "styles/mira.tcss"
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
        bootstrap: Bootstrap | None = None,
        ensure_git_repository: GitGuard | None = None,
        prebuilt: dict[str, Any] | None = None,
        tool_output_chars: int = DEFAULT_TOOL_OUTPUT_CHARS,
    ) -> None:
        super().__init__()
        self.workspace = workspace.expanduser().resolve() if workspace is not None else Path.cwd()
        self.resume = resume
        self.session_id = session_id
        self.config = config
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
        self._main_stream_active = False
        self._settings_panel: SettingsPanel | None = None

    def compose(self) -> ComposeResult:
        """Compose the Textual layout."""
        with Horizontal(id="app-shell"):
            with Vertical(id="session-sidebar"):
                yield Static("Chat History", id="session-sidebar-title")
                yield SessionHistory(id="sessions")
            with Vertical(id="main-panel"):
                yield StatusBar(id="status")
                yield ChatLog(tool_output_chars=self.tool_output_chars, id="chat-log")
                yield PromptPanel()
                yield PromptBox()

    def on_mount(self) -> None:
        """Start app initialization."""
        self.set_interval(1.0, self._tick_status)
        self.set_interval(0.12, self._tick_animations)
        self._set_status(state="starting")
        self.query_one(PromptBox).disabled = True
        if self.prebuilt is not None:
            self._install_state(self.prebuilt)
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
                self._set_status(state="checking workspace")
                if not await self.ensure_git_repository(self.workspace, self):
                    self.exit()
                    return

            if self.bootstrap is None:
                raise RuntimeError("MIRA bootstrap function was not provided")

            self.startup_progress("loading model metadata...")
            self._set_status(state="loading")
            state = await self.bootstrap(
                self.workspace,
                self.session_id,
                self.resume,
                self.config,
                self,
            )
            self._install_state(state)
        except Exception as exc:
            self.system_message(f"startup error: {exc}", kind="error")
            self._set_status(state="error")

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
        self.mode = initial_mode(self.agent, self.plan_agent)
        self.ready = True
        self.busy = False
        self._main_stream_active = False
        prompt = self.query_one(PromptBox)
        prompt.disabled = False
        prompt.set_history(read_prompt_history(self.history_path))
        ensure_dashboard(
            self.session,
            model_name=self.model_name,
            context_limit_tokens=self.context_limit_tokens,
            context_limit_source=self.context_limit_source,
        )

        chat = self.query_one(ChatLog)
        chat.clear_log()
        chat.startup(
            model_name=self.model_name,
            session_id=self.session["id"],
            workspace=str(self.session["workspace"]),
        )
        chat.restore_session(self.session)
        self._refresh_sessions()
        self._set_status(state="ready")
        self.action_focus_prompt()

    @on(PromptBox.Submitted)
    async def submit_prompt(self, event: PromptBox.Submitted) -> None:
        """Handle submitted prompt text."""
        text = event.value.strip()
        prompt = self.query_one(PromptBox)
        prompt.value = ""
        if not text or not self.ready or self.busy:
            if text in DESTRUCTIVE_HISTORY_COMMANDS and self.busy:
                self.system_message("finish the current turn before clearing history", kind="warning")
            self.action_focus_prompt()
            return

        self._record_prompt_history(text)
        if text in DESTRUCTIVE_HISTORY_COMMANDS:
            self.run_worker(self._run_history_command(text), name="history-command", exclusive=False)
            return

        if text == "/settings":
            self.run_worker(self._run_settings_command(), name="settings-command", exclusive=False)
            return

        if text == "/reload":
            self.run_worker(self._run_reload_command(), name="reload-command", exclusive=False)
            return

        if await handle_command(text, self, self.session, self.model_name, self.mode):
            self._set_status(state="ready")
            if text in {"/exit", "/quit"}:
                self.exit()
            else:
                self.action_focus_prompt()
            return

        self.busy = True
        self._main_stream_active = False
        self._set_status(state="running")
        prompt.disabled = True
        self.turn_worker = self.run_worker(self._run_turn(text), name="turn", exclusive=True)

    async def _run_turn(self, text: str) -> None:
        """Run one agent turn and restore prompt focus when done."""
        try:
            await self._refresh_model_metadata()
            await run_user_turn(
                agent=self.agent,
                plan_agent=self.plan_agent,
                renderer=self,
                store=self.store,
                session=self.session,
                mode=self.mode,
                text=text,
                model_name=self.model_name,
                context_limit_tokens=self.context_limit_tokens,
                context_limit_source=self.context_limit_source,
            )
            self._refresh_sessions()
            self._set_status(state="ready")
        except asyncio.CancelledError:
            self.subagents_cancelled()
            self.waiting_finished()
            self.system_message("turn cancelled", kind="warning")
            self._set_status(state="ready")
            raise
        except ContextOverflowError as exc:
            self.subagents_cancelled()
            self.waiting_finished()
            if not context_notice_rendered(exc):
                self.system_message(pop_context_overflow_notice(exc), kind="info")
            self._set_status(state="ready")
        except Exception as exc:
            self.subagents_cancelled()
            self.waiting_finished()
            self.system_message(f"error: {exc}", kind="error")
            self._set_status(state="error")
        finally:
            self.turn_worker = None
            self.busy = False
            self.subagents_cancelled()
            self.waiting_finished()
            prompt = self.query_one(PromptBox)
            prompt.disabled = False
            self.action_focus_prompt()

    @on(Button.Pressed, ".plan-action")
    def press_plan_action(self, event: Button.Pressed) -> None:
        """Handle structured plan bubble actions."""
        event.stop()
        button_id = event.button.id or ""
        for action in ("implement", "revise", "discard"):
            prefix = f"plan-{action}-"
            if button_id.startswith(prefix):
                self.run_worker(
                    self._handle_plan_action(action, button_id[len(prefix):]),
                    name=f"plan-{action}",
                    exclusive=False,
                )
                return

    async def _handle_plan_action(self, action: str, plan_id: str) -> None:
        """Resolve the active structured plan."""
        plan = self.mode.get("current_plan")
        if not isinstance(plan, dict) or str(plan.get("id") or "") != plan_id:
            self.system_message("that plan is no longer active", kind="warning")
            return
        if self.busy:
            self.system_message("finish the current turn before resolving the plan", kind="warning")
            return

        if action == "discard":
            self._resolve_current_plan("discarded")
            self.system_message(f"discarded plan \"{plan_title(plan)}\"", kind="muted")
            return

        if action == "revise":
            feedback = await self._prompt_text("Revise Plan", "What should MIRA change about this plan?")
            feedback = (feedback or "").strip()
            if not feedback:
                self.system_message(f"kept plan \"{plan_title(plan)}\" active", kind="muted")
                self.action_focus_prompt()
                return
            self._resolve_current_plan("revision requested")
            self.mode["planning"] = True
            self.mode["plan_runs"] = self.mode.get("plan_runs", 0) + 1
            self.mode["plan_thread_id"] = plan_thread_id(self.session, self.mode["plan_runs"])
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
            self._resolve_current_plan("approved for implementation")
            self.mode["planning"] = False
            self.mode["approved_plan"] = plan
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
        except Exception as exc:
            self.subagents_cancelled()
            self.waiting_finished()
            self.system_message(f"error: {exc}", kind="error")
            self._set_status(state="error")
        finally:
            self.turn_worker = None
            self.busy = False
            self.subagents_cancelled()
            self.waiting_finished()
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
        except Exception as exc:
            self.subagents_cancelled()
            self.waiting_finished()
            self.system_message(f"error: {exc}", kind="error")
            self._set_status(state="error")
        finally:
            self.turn_worker = None
            self.busy = False
            self.subagents_cancelled()
            self.waiting_finished()
            prompt = self.query_one(PromptBox)
            prompt.disabled = False
            self.action_focus_prompt()

    def _resolve_current_plan(self, status: str) -> None:
        """Mark the active plan as resolved and clear live state."""
        plan = self.mode.get("current_plan")
        if not isinstance(plan, dict):
            return
        plan_id = str(plan.get("id") or "")
        self.mode["current_plan"] = None
        self.query_one(ChatLog).resolve_plan(plan_id, status)
        update_plan_event_status(self.session, plan_id, status)
        self.store.save(self.session)

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
                    [("y", "y yes"), ("n", "n no")],
                )
                if answer == "y" and self.busy and self.turn_worker is not None:
                    self._cancel_turn()
                return

            answer = await self._prompt_choice(
                "Exit MIRA?",
                "No cancellable turn is running. Exit MIRA?",
                [("y", "y yes"), ("n", "n no")],
            )
            if answer == "y":
                self.exit()
        finally:
            self.confirming_interrupt = False

    def _cancel_turn(self) -> None:
        """Cancel the active turn worker."""
        if self.busy and self.turn_worker is not None:
            self.subagents_cancelled()
            self.waiting_finished()
            self.turn_worker.cancel()
            self._set_status(state="cancelling")

    def action_clear_log(self) -> None:
        """Clear chat and tool output."""
        self.clear_log()
        self._set_status(state="ready")

    def action_copy(self) -> None:
        """Copy the current widget selection without treating it as interrupt."""
        focused = self.focused
        copy_action = getattr(focused, "action_copy", None)
        if copy_action is None:
            raise SkipAction()
        copy_action()

    def action_focus_prompt(self) -> None:
        """Focus the prompt input."""
        if not self.is_mounted:
            return
        try:
            self.query_one(PromptBox).focus()
        except NoMatches:
            return

    def user_message(self, text: str, *, planning: bool = False, created_at: str = "") -> None:
        """Write a submitted user message to the chat log."""
        self.query_one(ChatLog).timestamped_user_message(text, planning=planning, created_at=created_at)

    def system_message(self, text: str, *, kind: str = "system", created_at: str = "") -> None:
        """Write a command or status message to the chat log."""
        self.waiting_finished()
        self.query_one(ChatLog).system_message(text, kind=kind, created_at=created_at)
        detail = text if kind in {"status", "info", "warning"} else ""
        self._set_status(state="ready" if not self.busy else "running", detail=detail)

    def command_output(self, renderable: Any) -> None:
        """Write command output to the chat log."""
        self.waiting_finished()
        self.query_one(ChatLog).command_output(renderable)

    async def present_plan(self, interrupt: Any) -> str:
        """Render a structured plan for explicit user review."""
        self.waiting_finished()
        plan = plan_request(interrupt)
        self.mode["plan_counter"] = int(self.mode.get("plan_counter") or 0) + 1
        plan = {**plan, "id": f"plan-{self.mode['plan_counter']}"}

        current = self.mode.get("current_plan")
        if isinstance(current, dict):
            current_id = str(current.get("id") or "")
            self.query_one(ChatLog).resolve_plan(current_id, "superseded")
            update_plan_event_status(self.session, current_id, "superseded")

        self.mode["current_plan"] = plan
        event = append_event(self.session, {"type": "plan", "mode": "planning", "plan": plan, "status": "pending"})
        self.store.save(self.session)
        self.query_one(ChatLog).present_plan(
            plan,
            active=True,
            status="pending",
            created_at=str(event.get("created_at") or ""),
        )
        return "Plan presented for user review."

    def compaction_started(self) -> None:
        """Show that DeepAgents is compacting conversation context."""
        self._main_stream_active = False
        self.waiting_finished()
        notice = pop_context_overflow_notice()
        if notice and not is_compaction_notice(notice):
            self.query_one(ChatLog).system_message(notice, kind="info")
        self.query_one(ChatLog).compaction_started()
        self._set_status(state="running", detail="compacting context...")

    def compaction_finished(self) -> None:
        """Show that DeepAgents has finished compacting context."""
        self.query_one(ChatLog).compaction_finished()
        self._set_status(state="running")
        self._rearm_waiting_if_busy()

    def clear_log(self) -> None:
        """Clear chat output."""
        self.query_one(ChatLog).clear_log()

    async def _run_history_command(self, text: str) -> None:
        """Run a destructive history command outside the submit event handler."""
        try:
            await self._handle_history_command(text)
            self._set_status(state="ready")
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

    async def _run_settings_command(self) -> None:
        """Run the interactive settings menu."""
        try:
            self._handle_settings_command()
            self._set_status(state="ready")
        except Exception as exc:
            self.system_message(f"settings error: {exc}", kind="error")
            self._set_status(state="error")

    async def _run_reload_command(self) -> None:
        """Reload project resources and rebuild agents."""
        try:
            if await self._handle_reload_command():
                self._set_status(state="ready")
        except Exception as exc:
            self.system_message(f"reload error: {exc}", kind="error")
            self._set_status(state="error")
        finally:
            self.action_focus_prompt()

    async def _handle_reload_command(self) -> bool:
        """Reload config, UI metadata, and agents from current workspace state."""
        if self.busy:
            self.system_message("finish the current turn before reloading agents", kind="warning")
            return True

        await self._reload_runtime()
        self.system_message("agents reloaded", kind="info")
        return True

    async def _reload_runtime(self) -> None:
        """Reload dotenv/config, model metadata, visible chrome, and agents."""
        from agent.llm import get_llm, get_model_name
        from config.loader import load_config
        from config.metadata import ModelMetadata, infer_model_metadata

        self.config = load_config(self.workspace, override_dotenv=True)
        inspect_model = get_llm(self.config, metadata=ModelMetadata())
        metadata = await infer_model_metadata(self.config, model=inspect_model)
        self.config["llm_inferred_context_tokens"] = metadata.context_tokens
        self.config["llm_context_source"] = metadata.context_source
        self.model_name = get_model_name(self.config)
        self.context_limit_tokens = metadata.context_tokens
        self.context_limit_source = metadata.context_source
        await self._rebuild_agents(metadata=metadata)
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

    def _refresh_startup_splash(self) -> None:
        """Refresh the visible startup metadata block after runtime changes."""
        if not self.is_mounted:
            return
        self.query_one(ChatLog).startup(
            model_name=self.model_name,
            session_id=self.session["id"],
            workspace=str(self.session["workspace"]),
        )

    def _handle_settings_command(self) -> bool:
        """Mount the interactive settings panel."""
        if self.busy:
            self.system_message("finish the current turn before changing settings", kind="warning")
            return True

        settings = load_settings(self.workspace)
        if self._settings_panel is not None and self._settings_panel.is_mounted:
            self._settings_panel.remove()
        panel = SettingsPanel(
            settings,
            tool_metadata=self._settings_tool_metadata(),
            apply_change=self._apply_settings,
            close_panel=self._close_settings_panel,
        )
        self._settings_panel = panel
        self.mount(panel)
        return True

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

        self.config = dict(self.config or {})
        self.config["settings"] = settings
        if new_git_enabled != old_git_enabled:
            if new_git_enabled:
                return await self._ensure_git_after_enabling()
            return True, "git protection disabled"

        await self._rebuild_agents()
        return True, "settings saved; agents rebuilt"

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
            return True, "git protection enabled"
        if init_git_repository(self.workspace):
            return True, "git protection enabled; repository initialized"
        return True, "git protection enabled, but Git was not initialized"

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
        self.waiting_finished()
        self._mark_main_stream_active()
        self.query_one(ChatLog).reasoning_delta(delta, created_at=created_at)

    def discard_reasoning(self) -> None:
        """Remove streamed reasoning that was later classified as internal."""
        self.query_one(ChatLog).discard_reasoning()

    def text_delta(self, delta: str, *, created_at: str = "") -> None:
        """Render streamed assistant text."""
        self.waiting_finished()
        self._mark_main_stream_active()
        self.query_one(ChatLog).text_delta(delta, created_at=created_at)

    def model_activity(self) -> None:
        """Render transient activity for streamed non-text model output."""
        self.waiting_finished()
        self._mark_main_stream_active()
        self.query_one(ChatLog).model_activity()
        self._set_status(state="running", detail="preparing tool call...")

    def model_stream_finished(self) -> None:
        """Re-arm waiting UI after streamed model text/reasoning goes quiet."""
        self._finish_main_stream_activity()
        self._rearm_waiting_if_busy()

    def tool_call_delta(self, name: str, args: Any, call_id: str = "") -> None:
        """Render a live draft of streamed tool-call input."""
        self.waiting_finished()
        self._mark_main_stream_active()
        self.query_one(ChatLog).tool_call_delta(name, args, call_id=call_id)
        self._set_status(state="running", detail="preparing tool call...")

    def delegation_delta(self, calls: list[dict[str, Any]]) -> None:
        """Render a live draft of streamed task delegation input."""
        self.waiting_finished()
        self._mark_main_stream_active()
        self.query_one(ChatLog).delegation_delta(calls)
        self._set_status(state="running", detail="preparing subagent request...")

    def tool_call(self, name: str, args: Any, call_id: str = "", *, created_at: str = "") -> None:
        """Render a tool call in transcript order."""
        self._finish_main_stream_activity()
        self.waiting_finished()
        self.query_one(ChatLog).tool_call(name, args, call_id=call_id, created_at=created_at)
        self._rearm_waiting_if_busy()

    def tool_result(self, name: str, result: str, call_id: str = "", *, created_at: str = "") -> None:
        """Render a tool result in transcript order."""
        self._finish_main_stream_activity()
        self.waiting_finished()
        self.query_one(ChatLog).tool_result(name, result, call_id=call_id, created_at=created_at)
        self._rearm_waiting_if_busy()

    def delegation_started(self, calls: list[dict[str, Any]], *, created_at: str = "") -> None:
        """Render task delegation summary."""
        self._finish_main_stream_activity()
        self.waiting_finished()
        self.query_one(ChatLog).delegation_started(calls, created_at=created_at)
        self._rearm_waiting_if_busy()

    def start_subagent_live(self) -> None:
        """Prepare subagent display."""
        self.query_one(ChatLog).start_subagent_live()

    def stop_subagent_live(self) -> None:
        """Finalize subagent display."""
        self.query_one(ChatLog).stop_subagent_live()

    def subagents_cancelled(self) -> None:
        """Mark active subagent display as cancelled."""
        self.query_one(ChatLog).subagents_cancelled()

    def tick_subagents(self) -> None:
        """Advance subagent status animation."""
        self.query_one(ChatLog).tick_subagents()

    def subagent_label(self, subagent: Any) -> str:
        """Return a stable display label for a subagent."""
        return self.query_one(ChatLog).subagent_label(subagent)

    def subagent_started(
        self,
        subagent: str,
        task_input: str = "",
        *,
        origin: str = "",
        created_at: str = "",
    ) -> None:
        """Render a subagent start."""
        self._finish_main_stream_activity()
        self.waiting_finished()
        self.query_one(ChatLog).subagent_started(subagent, task_input, origin=origin, created_at=created_at)
        self._rearm_waiting_if_busy()

    def subagent_request_updated(self, subagent: str, task_input: str) -> None:
        """Fill in a subagent request that arrived after the block started."""
        self.waiting_finished()
        self.query_one(ChatLog).subagent_request_updated(subagent, task_input)

    def subagent_finished(self, subagent: str, result: str = "", *, created_at: str = "") -> None:
        """Render a subagent finish."""
        self._finish_main_stream_activity()
        self.waiting_finished()
        self.query_one(ChatLog).subagent_finished(subagent, result, created_at=created_at)
        self._rearm_waiting_if_busy()

    def subagent_cancelled(self, subagent: str, result: str = "", *, created_at: str = "") -> None:
        """Render a subagent cancellation."""
        self._finish_main_stream_activity()
        self.waiting_finished()
        self.query_one(ChatLog).subagent_cancelled(subagent, result, created_at=created_at)

    def finish_main(self) -> None:
        """Close streamed chat blocks after a top-level turn."""
        self._finish_main_stream_activity()
        self.waiting_finished()
        self.query_one(ChatLog).finish_main()

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
        answer = str(await self._prompt_choice("Question", question, choices) or "")
        selected = options[int(answer) - 1] if answer.isdigit() and 0 < int(answer) <= len(options) else options[-1]
        if selected != ASK_USER_OPEN_OPTION:
            return selected

        response = await self._prompt_text("Question", ASK_USER_OPEN_OPTION)
        return (response or "").strip() or ASK_USER_OPEN_OPTION

    async def ask_create_git_repo(self, message: str) -> bool:
        """Ask whether MIRA should initialize Git for the workspace."""
        answer = await self._prompt_choice("Git", message, [("y", "y yes"), ("n", "n no")])
        return answer == "y"

    async def ask_continue_without_git(self, message: str) -> bool:
        """Ask whether startup should continue without Git protection."""
        answer = await self._prompt_choice("Git", message, [("c", "c continue"), ("e", "e exit")])
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

    async def _prompt_choice(self, title: str, message: str, choices: list[tuple[str, str]]) -> str | None:
        """Show a choice prompt in the main window."""
        return await self._with_prompt_lock(self.query_one(PromptPanel).choose(title, message, choices))

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

    def waiting_started(self) -> None:
        """Arm the transient working indicator while the turn is silent."""
        self._waiting_generation += 1
        self._cancel_waiting_task()
        if not self.is_mounted or not self.busy or self._main_stream_active:
            return
        try:
            if self.query_one(PromptPanel).active:
                return
        except NoMatches:
            return
        generation = self._waiting_generation
        self._waiting_task = self.run_worker(
            self._show_waiting_after_delay(generation),
            name=f"waiting-{generation}",
            exclusive=False,
        )

    def waiting_finished(self) -> None:
        """Hide the transient working indicator and cancel pending timers."""
        self._cancel_waiting_task()
        if self.is_mounted:
            self.query_one(ChatLog).hide_waiting()

    async def _show_waiting_after_delay(self, generation: int) -> None:
        """Show working only if the current wait survives the grace period."""
        try:
            await asyncio.sleep(self._waiting_delay_seconds)
        except asyncio.CancelledError:
            return
        if generation != self._waiting_generation or not self.busy or not self.is_mounted or self._main_stream_active:
            return
        self.query_one(ChatLog).show_waiting()

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
        if self.busy and not self._main_stream_active:
            self.waiting_started()

    def startup_progress(self, state: str) -> None:
        """Update startup splash and status while bootstrap is running."""
        if not self.is_mounted:
            return
        try:
            self.query_one(ChatLog).startup_progress(state)
        except NoMatches:
            return
        self._set_status(state="loading", detail=state)

    def _set_status(self, *, state: str, detail: str = "") -> None:
        """Update the status bar if it has been mounted."""
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
            detail=detail,
        )

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
        self._set_status(state="loading")
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
            self.system_message(f"session load error: {exc}", kind="error")
            self._set_status(state="error")
        finally:
            self.busy = False
            prompt.disabled = False
            self.action_focus_prompt()

    async def _refresh_model_metadata(self) -> None:
        """Refresh model metadata and rebuild agents when context changes."""
        if self.config is None or self.checkpointer is None:
            return
        if not self.config.get("llm_provider") or not self.config.get("llm_model"):
            return

        from agent.llm import get_llm
        from config.metadata import ModelMetadata, infer_model_metadata

        inspect_model = get_llm(self.config, metadata=ModelMetadata())
        metadata = await infer_model_metadata(self.config, model=inspect_model)
        if not metadata.context_tokens or metadata.context_tokens == self.context_limit_tokens:
            return

        self.config["llm_inferred_context_tokens"] = metadata.context_tokens
        self.config["llm_context_source"] = metadata.context_source
        self.context_limit_tokens = metadata.context_tokens
        self.context_limit_source = metadata.context_source
        await self._rebuild_agents(metadata=metadata)
        ensure_dashboard(
            self.session,
            model_name=self.model_name,
            context_limit_tokens=self.context_limit_tokens,
            context_limit_source=self.context_limit_source,
        )
        self.store.save(self.session)
        self._set_status(state=self.status_state, detail=f"context window: {metadata.context_tokens}")

    async def _rebuild_agents(self, metadata: Any | None = None) -> None:
        """Rebuild action and planning agents after settings or metadata changes."""
        if self.config is None or self.checkpointer is None:
            return

        from agent.factory import build_agent, build_plan_agent

        self.agent = build_agent(
            config=self.config,
            workspace=self.workspace,
            checkpointer=self.checkpointer,
            metadata=metadata,
        )
        self.plan_agent = build_plan_agent(
            config=self.config,
            workspace=self.workspace,
            checkpointer=self.checkpointer,
            metadata=metadata,
        )
        refresh_agent_specs(self.mode, self.agent, self.plan_agent)

    def _refresh_sessions(self) -> None:
        """Reload the session list if the store is available."""
        if self.store is None or not self.is_mounted:
            return
        self.query_one(SessionHistory).refresh_sessions(self.store, current_id=str(self.session.get("id") or ""))

    def _tick_status(self) -> None:
        """Refresh the clock and dashboard line."""
        if not self.ready:
            return
        update_duration(self.session)
        self._set_status(state=self.status_state)

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

    def _mode_label(self) -> str:
        """Return the compact mode label shown in the status line."""
        if self.mode.get("planning"):
            return "Plan"
        if self.mode.get("current_plan"):
            return "Plan Ready"
        return "Action"

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


def is_compaction_notice(text: str) -> bool:
    """Return whether an info notice is really leaked compaction reasoning."""
    return is_compaction_reasoning(text) or is_compaction_reasoning_fragment(text)


def plan_title(plan: dict[str, Any]) -> str:
    """Return a compact plan title for status text."""
    return str(plan.get("title") or "Implementation Plan")
