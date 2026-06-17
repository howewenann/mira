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
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.widgets import ListView, Static

from agent.context_overflow import context_notice_rendered, pop_context_overflow_notice
from session.dashboard import ensure_dashboard, normalize_dashboard, update_duration
from ui.interrupts import (
    ASK_USER_OPEN_OPTION,
    action_choices,
    action_preview,
    action_requests,
    action_title,
    ask_user_options,
    ask_user_question,
    ask_user_request,
    response_message,
)
from ui.repl import handle_command, initial_mode, refresh_agent_specs, run_user_turn
from ui.widgets import ChatLog, PromptBox, PromptPanel, SessionHistory, StatusBar
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
        Binding("ctrl+c", "interrupt_or_quit", "Cancel/Quit", priority=True),
        Binding("ctrl+q", "quit", "Quit"),
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
        self.token_counter: Any | None = None
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
        self.token_counter = state.get("token_counter")
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

        if await handle_command(text, self, self.session, self.model_name, self.mode):
            self._set_status(state="ready")
            if text in {"/exit", "/quit"}:
                self.exit()
            else:
                self.action_focus_prompt()
            return

        self.query_one(ChatLog).user_message(text, planning=bool(self.mode.get("planning")))
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
                token_counter=self.token_counter,
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

    def action_interrupt_or_quit(self) -> None:
        """Confirm before cancelling a turn or quitting the app."""
        if self.confirming_interrupt:
            return
        if self.query_one(PromptPanel).active:
            self._cancel_turn()
            return
        self.run_worker(self._confirm_interrupt_or_quit(), name="confirm-interrupt", exclusive=False)

    async def _confirm_interrupt_or_quit(self) -> None:
        """Ask for confirmation before handling Ctrl+C."""
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

    def action_focus_prompt(self) -> None:
        """Focus the prompt input."""
        if not self.is_mounted:
            return
        try:
            self.query_one(PromptBox).focus()
        except NoMatches:
            return

    def system_message(self, text: str, *, kind: str = "system") -> None:
        """Write a command or status message to the chat log."""
        self.waiting_finished()
        self.query_one(ChatLog).system_message(text, kind=kind)
        detail = text if kind in {"status", "info", "warning"} else ""
        self._set_status(state="ready" if not self.busy else "running", detail=detail)

    def command_output(self, renderable: Any) -> None:
        """Write command output to the chat log."""
        self.waiting_finished()
        self.query_one(ChatLog).command_output(renderable)

    def compaction_started(self) -> None:
        """Show that DeepAgents is compacting conversation context."""
        self._main_stream_active = False
        self.waiting_finished()
        notice = pop_context_overflow_notice()
        if notice:
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

    def plan(self, plan_id: int, text: str) -> None:
        """Display a saved plan."""
        self.query_one(ChatLog).plan(plan_id, text)

    def no_plans(self) -> None:
        """Display the empty saved-plan state."""
        self.query_one(ChatLog).no_plans()

    def reasoning_delta(self, delta: str) -> None:
        """Render streamed reasoning text."""
        self.waiting_finished()
        self._mark_main_stream_active()
        self.query_one(ChatLog).reasoning_delta(delta)

    def text_delta(self, delta: str) -> None:
        """Render streamed assistant text."""
        self.waiting_finished()
        self._mark_main_stream_active()
        self.query_one(ChatLog).text_delta(delta)

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

    def tool_call(self, name: str, args: Any, call_id: str = "") -> None:
        """Render a tool call in transcript order."""
        self._finish_main_stream_activity()
        self.waiting_finished()
        self.query_one(ChatLog).tool_call(name, args, call_id=call_id)
        self._rearm_waiting_if_busy()

    def tool_result(self, name: str, result: str, call_id: str = "") -> None:
        """Render a tool result in transcript order."""
        self._finish_main_stream_activity()
        self.waiting_finished()
        self.query_one(ChatLog).tool_result(name, result, call_id=call_id)
        self._rearm_waiting_if_busy()

    def delegation_started(self, calls: list[dict[str, Any]]) -> None:
        """Render task delegation summary."""
        self._finish_main_stream_activity()
        self.waiting_finished()
        self.query_one(ChatLog).delegation_started(calls)
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

    def subagent_started(self, subagent: str, task_input: str = "") -> None:
        """Render a subagent start."""
        self._finish_main_stream_activity()
        self.waiting_finished()
        self.query_one(ChatLog).subagent_started(subagent, task_input)
        self._rearm_waiting_if_busy()

    def subagent_request_updated(self, subagent: str, task_input: str) -> None:
        """Fill in a subagent request that arrived after the block started."""
        self.waiting_finished()
        self.query_one(ChatLog).subagent_request_updated(subagent, task_input)

    def subagent_finished(self, subagent: str, result: str = "") -> None:
        """Render a subagent finish."""
        self._finish_main_stream_activity()
        self.waiting_finished()
        self.query_one(ChatLog).subagent_finished(subagent, result)
        self._rearm_waiting_if_busy()

    def subagent_cancelled(self, subagent: str, result: str = "") -> None:
        """Render a subagent cancellation."""
        self._finish_main_stream_activity()
        self.waiting_finished()
        self.query_one(ChatLog).subagent_cancelled(subagent, result)

    def finish_main(self) -> None:
        """Close streamed chat blocks after a top-level turn."""
        self._finish_main_stream_activity()
        self.waiting_finished()
        self.query_one(ChatLog).finish_main()

    def usage_updated(self) -> None:
        """Refresh the status bar after token usage is committed."""
        self._set_status(state=self.status_state)

    async def ask_approvals(self, interrupts: list[Any]) -> list[dict[str, Any]]:
        """Ask the user to approve, edit, reject, or respond to interrupted actions."""
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
                elif answer == "s":
                    decisions.append(await self.respond_decision(action))
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

    async def respond_decision(self, action: Any) -> dict[str, Any]:
        """Prompt for a synthetic successful tool response."""
        message = await self._prompt_text("Respond", "Type the tool result to return without running it.")
        return {"type": "respond", "message": response_message(message, action)}

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

        from agent.factory import build_agent, build_plan_agent
        from config.metadata import infer_model_metadata

        metadata = await infer_model_metadata(self.config)
        if not metadata.context_tokens or metadata.context_tokens == self.context_limit_tokens:
            return

        self.config["llm_inferred_context_tokens"] = metadata.context_tokens
        self.config["llm_context_source"] = metadata.context_source
        self.context_limit_tokens = metadata.context_tokens
        self.context_limit_source = metadata.context_source
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
        ensure_dashboard(
            self.session,
            model_name=self.model_name,
            context_limit_tokens=self.context_limit_tokens,
            context_limit_source=self.context_limit_source,
        )
        self.store.save(self.session)
        self._set_status(state=self.status_state, detail=f"context window: {metadata.context_tokens}")

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
        if self.mode.get("plan_pending"):
            return "Action + Plan"
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
