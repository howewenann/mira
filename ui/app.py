"""Textual application shell for MIRA."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import ListView, Static

from session.dashboard import ensure_dashboard, update_duration
from ui.interrupts import (
    ASK_USER_OPEN_OPTION,
    action_choices,
    action_requests,
    action_text,
    ask_user_options,
    ask_user_question,
    ask_user_request,
    response_message,
)
from ui.repl import handle_command, initial_mode, run_user_turn
from ui.widgets import ChatLog, PromptBox, PromptPanel, SessionHistory, StatusBar
from ui.widgets.chat_log import DEFAULT_TOOL_OUTPUT_CHARS
from ui.widgets.session_history import SessionItem

Bootstrap = Callable[[Path, str | None, bool, dict[str, Any] | None, Any | None], dict[str, Any]]
GitGuard = Callable[[Path, Any], Any]


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
        self.mode: dict[str, Any] = {"planning": False}
        self.ready = False
        self.busy = False
        self.status_state = "starting"
        self.turn_worker: Any | None = None
        self.confirming_interrupt = False

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
        self._set_status(state="starting")
        self.query_one(PromptBox).disabled = True
        if self.prebuilt is not None:
            self._install_state(self.prebuilt)
            return
        self.run_worker(self._startup(), name="startup", exclusive=True)

    async def _startup(self) -> None:
        """Run Git safety checks and build agents inside the TUI."""
        try:
            if self.ensure_git_repository is not None:
                self._set_status(state="checking workspace")
                if not await self.ensure_git_repository(self.workspace, self):
                    self.exit()
                    return

            if self.bootstrap is None:
                raise RuntimeError("MIRA bootstrap function was not provided")

            self._set_status(state="loading")
            state = await asyncio.to_thread(
                self.bootstrap,
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
        self.store = state["store"]
        self.session = state["session"]
        self.model_name = str(state.get("model_name") or "")
        self.context_limit_tokens = state.get("context_limit_tokens")
        self.context_limit_source = str(state.get("context_limit_source") or "unknown")
        self.token_counter = state.get("token_counter")
        self.mode = initial_mode(self.agent, self.plan_agent)
        self.ready = True
        self.busy = False
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
            self.action_focus_prompt()
            return

        self._record_prompt_history(text)
        if await handle_command(text, self, self.session, self.model_name, self.mode):
            self._set_status(state="ready")
            if text in {"/exit", "/quit"}:
                self.exit()
            else:
                self.action_focus_prompt()
            return

        self.query_one(ChatLog).user_message(text, planning=bool(self.mode.get("planning")))
        self.busy = True
        self._set_status(state="running")
        prompt.disabled = True
        self.turn_worker = self.run_worker(self._run_turn(text), name="turn", exclusive=True)

    async def _run_turn(self, text: str) -> None:
        """Run one agent turn and restore prompt focus when done."""
        try:
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
            self.system_message("turn cancelled", kind="warning")
            self._set_status(state="ready")
            raise
        except Exception as exc:
            self.system_message(f"error: {exc}", kind="error")
            self._set_status(state="error")
        finally:
            self.turn_worker = None
            self.busy = False
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
            self.turn_worker.cancel()
            self._set_status(state="cancelling")

    def action_clear_log(self) -> None:
        """Clear chat and tool output."""
        self.clear_log()
        self._set_status(state="ready")

    def action_focus_prompt(self) -> None:
        """Focus the prompt input."""
        if self.is_mounted:
            self.query_one(PromptBox).focus()

    def system_message(self, text: str, *, kind: str = "system") -> None:
        """Write a command or status message to the chat log."""
        self.query_one(ChatLog).system_message(text, kind=kind)
        detail = text if kind in {"status", "warning"} else ""
        self._set_status(state="ready" if not self.busy else "running", detail=detail)

    def command_output(self, renderable: Any) -> None:
        """Write command output to the chat log."""
        self.query_one(ChatLog).command_output(renderable)

    def clear_log(self) -> None:
        """Clear chat output."""
        self.query_one(ChatLog).clear_log()

    def plan(self, plan_id: int, text: str) -> None:
        """Display a saved plan."""
        self.query_one(ChatLog).plan(plan_id, text)

    def no_plans(self) -> None:
        """Display the empty saved-plan state."""
        self.query_one(ChatLog).no_plans()

    def reasoning_delta(self, delta: str) -> None:
        """Render streamed reasoning text."""
        self.query_one(ChatLog).reasoning_delta(delta)

    def text_delta(self, delta: str) -> None:
        """Render streamed assistant text."""
        self.query_one(ChatLog).text_delta(delta)

    def tool_call(self, name: str, args: Any) -> None:
        """Render a tool call in transcript order."""
        self.query_one(ChatLog).tool_call(name, args)

    def tool_result(self, name: str, result: str) -> None:
        """Render a tool result in transcript order."""
        self.query_one(ChatLog).tool_result(name, result)

    def delegation_started(self, calls: list[dict[str, Any]]) -> None:
        """Render task delegation summary."""
        self.query_one(ChatLog).delegation_started(calls)

    def start_subagent_live(self) -> None:
        """Prepare subagent display."""
        self.query_one(ChatLog).start_subagent_live()

    def stop_subagent_live(self) -> None:
        """Finalize subagent display."""
        self.query_one(ChatLog).stop_subagent_live()

    def tick_subagents(self) -> None:
        """Advance subagent status animation."""
        self.query_one(ChatLog).tick_subagents()

    def subagent_label(self, subagent: Any) -> str:
        """Return a stable display label for a subagent."""
        return self.query_one(ChatLog).subagent_label(subagent)

    def subagent_started(self, subagent: str, task_input: str = "") -> None:
        """Render a subagent start."""
        self.query_one(ChatLog).subagent_started(subagent, task_input)

    def subagent_finished(self, subagent: str, result: str = "") -> None:
        """Render a subagent finish."""
        self.query_one(ChatLog).subagent_finished(subagent, result)

    def finish_main(self) -> None:
        """Close streamed chat blocks after a top-level turn."""
        self.query_one(ChatLog).finish_main()

    async def ask_approvals(self, interrupts: list[Any]) -> list[dict[str, Any]]:
        """Ask the user to approve, edit, reject, or respond to interrupted actions."""
        decisions: list[dict[str, Any]] = []
        for interrupt in interrupts:
            for index, action in enumerate(action_requests(interrupt)):
                answer = await self._prompt_choice(
                    "Approval",
                    action_text(action),
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
            state = await asyncio.to_thread(
                self.bootstrap,
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
