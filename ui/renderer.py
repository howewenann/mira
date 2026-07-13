"""Plain terminal renderer used by one-shot prompts."""

from __future__ import annotations

import asyncio
import json
from itertools import count
from typing import Any

from agent.context_overflow import pop_context_overflow_notice
from ui.terminal_colors import TerminalColorizer, enable_console_colors
from ui.interrupts import (
    ASK_USER_OPEN_OPTION,
    action_choices,
    action_requests,
    action_text,
    ask_user_options,
    ask_user_question,
    ask_user_request,
    plan_request,
)
from ui.terminal_transcript import DEFAULT_TOOL_OUTPUT_CHARS, TerminalTranscript

__all__ = ["DEFAULT_TOOL_OUTPUT_CHARS", "Renderer"]


class Renderer:
    """Small stdout renderer for non-interactive `mira --prompt` runs."""

    def __init__(self, tool_output_chars: int = DEFAULT_TOOL_OUTPUT_CHARS) -> None:
        self.tool_output_chars = tool_output_chars
        self._subagent_ids = count(1)
        enable_console_colors()
        self._colorizer = TerminalColorizer()
        self.transcript = TerminalTranscript(
            self._write,
            tool_output_chars=tool_output_chars,
            slug_fallback=self._subagent_ids,
        )

    def reasoning_delta(self, delta: str) -> None:
        """Buffer streamed reasoning text for a clean terminal block."""
        self.transcript.reasoning_delta(delta)

    def discard_reasoning(self) -> None:
        """Discard pending reasoning that has not been printed yet."""
        self.transcript.discard_reasoning()

    def text_delta(self, delta: str) -> None:
        """Print streamed assistant text."""
        self.transcript.text_delta(delta)

    def tool_call(self, name: str, args: Any, call_id: str = "") -> None:
        """Print a compact tool call."""
        self.transcript.tool_call(name, args)

    def tool_result(self, name: str, result: str, call_id: str = "") -> None:
        """Print a compact tool result."""
        self.transcript.tool_result(name, result)

    def delegation_started(self, calls: list[dict[str, Any]]) -> None:
        """Print a compact task delegation summary."""
        self.transcript.delegation_started(calls)

    def system_message(self, text: str, *, kind: str = "system") -> None:
        """Print one system-style block."""
        self.transcript.system_message(text, kind=kind)

    def compaction_started(self) -> None:
        """Print a context compaction status."""
        notice = pop_context_overflow_notice()
        if notice:
            self.system_message(notice, kind="info")
        self.transcript.compaction_started()

    def compaction_finished(self) -> None:
        """Print a context compaction completion status."""
        self.transcript.compaction_finished()

    def subagent_label(self, subagent: Any) -> str:
        """Return a stable readable label for a subagent object."""
        return self.transcript.subagent_label(subagent)

    def subagent_started(self, subagent: str, task_input: str = "", *, origin: str = "") -> None:
        """Print a subagent start."""
        self.transcript.subagent_started(subagent, task_input, origin=origin)

    def subagent_request_updated(self, subagent: str, task_input: str) -> None:
        """Print a late-arriving request for an already-started subagent."""
        self.transcript.subagent_request_updated(subagent, task_input)

    def subagent_finished(self, subagent: str, result: str = "") -> None:
        """Print a subagent finish."""
        self.transcript.subagent_finished(subagent, result)

    def subagent_cancelled(self, subagent: str, result: str = "") -> None:
        """Print a subagent cancellation."""
        self.transcript.subagent_cancelled(subagent, result)

    def subagents_cancelled(self) -> None:
        """No-op for non-live terminal output."""
        return None

    def rubric_evaluation_started(self, run_id: str, pass_number: int, max_iterations: int) -> None:  # noqa: ARG002
        """Print rubric review activity in one-shot mode."""
        self.transcript.rubric_evaluation_started(pass_number, max_iterations)

    def rubric_evaluation_finished(
        self,
        evaluation: dict[str, Any],
        max_iterations: int,
    ) -> None:
        """Print a completed rubric evaluation."""
        self.transcript.rubric_evaluation_finished(evaluation, max_iterations)

    def rubric_evaluation_status(
        self,
        run_id: str,
        pass_number: int,
        status: str,
        max_iterations: int,
    ) -> None:  # noqa: ARG002
        """Print a reconciled terminal rubric status."""
        self.transcript.rubric_evaluation_status(pass_number, status, max_iterations)

    def finish_main(self) -> None:
        """Finish the current streamed section."""
        self.transcript.finish_main()

    async def ask_approvals(self, interrupts: list[Any]) -> list[dict[str, Any]]:
        """Ask the user to approve, edit, or reject interrupted actions."""
        decisions = []
        for interrupt in interrupts:
            for index, action in enumerate(action_requests(interrupt)):
                self.transcript.block("approval", action_text(action))
                answer = await self._choice("Approve this action?", action_choices(interrupt, action, index))
                if answer == "e":
                    decisions.append(await self._edit_decision(action))
                elif answer == "r":
                    decisions.append({"type": "reject"})
                else:
                    decisions.append({"type": "approve"})
        return decisions

    async def ask_user(self, interrupt: Any) -> str:
        """Ask the user for a concrete next-step choice."""
        request = ask_user_request(interrupt)
        options = ask_user_options(request)
        self.transcript.block("question", ask_user_question(request))
        for index, option in enumerate(options, start=1):
            self.transcript.line(f"{index}. {option}")

        answer = await self._input("Choose an option: ")
        try:
            selected = options[int(answer) - 1]
        except (ValueError, IndexError):
            selected = options[-1]

        if selected != ASK_USER_OPEN_OPTION:
            return selected

        response = (await self._input(f"{ASK_USER_OPEN_OPTION}: ")).strip()
        return response or ASK_USER_OPEN_OPTION

    async def present_plan(self, interrupt: Any) -> str:
        """Print a structured plan in one-shot terminal mode."""
        plan = plan_request(interrupt)
        lines = [str(plan.get("title") or "Implementation Plan"), ""]
        for heading, key in (
            ("Summary", "summary"),
            ("Key Changes", "key_changes"),
            ("Test Plan", "test_plan"),
            ("Assumptions", "assumptions"),
        ):
            items = plan.get(key)
            if not isinstance(items, list) or not items:
                continue
            lines.append(heading)
            lines.extend(f"- {item}" for item in items)
            lines.append("")
        self.transcript.block("plan", "\n".join(lines).rstrip())
        return "Plan presented for user review."

    async def ask_create_git_repo(self, message: str) -> bool:
        """Ask whether MIRA should initialize Git for the workspace."""
        return await self._choice(message, [("y", "yes"), ("n", "no")]) == "y"

    async def ask_continue_without_git(self, message: str) -> bool:
        """Ask whether startup should continue without Git protection."""
        return await self._choice(message, [("c", "continue"), ("e", "exit")]) == "c"

    def truncate(self, value: Any) -> str:
        """Return a single-line string shortened to the configured display size."""
        return self.transcript.truncate(value)

    async def _edit_decision(self, action: Any) -> dict[str, Any]:
        """Ask for edited JSON args."""
        if not isinstance(action, dict):
            return {"type": "reject"}

        original = json.dumps(action.get("args", {}), indent=2)
        edited = await self._input(f"Edited args JSON [{original}]: ")
        try:
            args = json.loads(edited or original)
        except json.JSONDecodeError:
            self.transcript.line("invalid JSON; rejecting action")
            return {"type": "reject"}

        if not isinstance(args, dict):
            self.transcript.line("edited args must be a JSON object; rejecting action")
            return {"type": "reject"}

        return {
            "type": "edit",
            "edited_action": {
                "name": action.get("name", "tool"),
                "args": args,
            },
        }

    async def _choice(self, message: str, options: list[tuple[str, str]]) -> str:
        """Prompt until the user chooses one of the given keys."""
        labels = ", ".join(f"{key}={label}" for key, label in options)
        keys = {key for key, _ in options}
        while True:
            answer = (await self._input(f"{message} ({labels}): ")).strip().lower()
            if answer in keys:
                return answer

    async def _input(self, prompt: str) -> str:
        """Read input without blocking the event loop."""
        return await asyncio.to_thread(input, prompt)

    def _write(self, text: str) -> None:
        """Write colored transcript text to stdout."""
        print(self._colorizer.colorize(text), end="", flush=True)
