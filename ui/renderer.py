"""Plain terminal renderer used by one-shot prompts."""

from __future__ import annotations

import asyncio
import json
import re
from itertools import count
from typing import Any

from agent.context_overflow import pop_context_overflow_notice
from runtime.compaction_filter import is_compaction_reasoning, is_compaction_reasoning_fragment
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
from ui.names import generate_slug

DEFAULT_TOOL_OUTPUT_CHARS = 240


class Renderer:
    """Small stdout renderer for non-interactive `mira --prompt` runs."""

    def __init__(self, tool_output_chars: int = DEFAULT_TOOL_OUTPUT_CHARS) -> None:
        self.tool_output_chars = tool_output_chars
        self._section = ""
        self._subagent_ids = count(1)
        self._subagent_labels: dict[int, str] = {}

    def reasoning_delta(self, delta: str) -> None:
        """Print streamed reasoning text."""
        text = re.sub(r"</?[^>]+>", "", delta)
        if text:
            self._stream("thinking", text)

    def discard_reasoning(self) -> None:
        """Terminal output cannot retract already printed reasoning."""
        return

    def text_delta(self, delta: str) -> None:
        """Print streamed assistant text."""
        if delta:
            self._stream("mira", delta)

    def tool_call(self, name: str, args: Any, call_id: str = "") -> None:
        """Print a compact tool call."""
        self._block(name, f"args: {self.truncate(args)}")

    def tool_result(self, name: str, result: str, call_id: str = "") -> None:
        """Print a compact tool result."""
        if result:
            self._line(f"{name} output: {self.truncate(result)}")

    def delegation_started(self, calls: list[dict[str, Any]]) -> None:
        """Print a compact task delegation summary."""
        descriptions = []
        for call in calls:
            raw_args = call.get("args", {}) if isinstance(call, dict) else {}
            if isinstance(raw_args, str):
                try:
                    raw_args = json.loads(raw_args)
                except json.JSONDecodeError:
                    raw_args = {}
            if isinstance(raw_args, dict) and raw_args.get("description"):
                descriptions.append(str(raw_args["description"]))

        if descriptions:
            lines = [f"delegating to {len(descriptions)} subagent(s)"]
            lines.extend(f"request: {self.truncate(description)}" for description in descriptions)
            self._block("task", "\n".join(lines))

    def system_message(self, text: str, *, kind: str = "system") -> None:
        """Print one system-style block."""
        self._block(kind, text)

    def compaction_started(self) -> None:
        """Print a context compaction status."""
        notice = pop_context_overflow_notice()
        if notice and not is_compaction_notice(notice):
            self.system_message(notice, kind="info")
        self._block("mira", "compacting context...")

    def compaction_finished(self) -> None:
        """Print a context compaction completion status."""
        self._line("context compacted")

    def subagent_label(self, subagent: Any) -> str:
        """Return a stable readable label for a subagent object."""
        key = id(subagent)
        if key not in self._subagent_labels:
            name = getattr(subagent, "name", "subagent")
            self._subagent_labels[key] = f"{name} [{self._next_suffix()}]"
        return self._subagent_labels[key]

    def subagent_started(self, subagent: str, task_input: str = "") -> None:
        """Print a subagent start."""
        details = f"request: {self.truncate(task_input)}" if task_input else "running"
        self._block(f"subagent - {subagent}", details)

    def subagent_request_updated(self, subagent: str, task_input: str) -> None:
        """Print a late-arriving request for an already-started subagent."""
        if task_input:
            self._block(f"subagent - {subagent}", f"request: {self.truncate(task_input)}")

    def subagent_finished(self, subagent: str, result: str = "") -> None:
        """Print a subagent finish."""
        details = "done"
        if result:
            details += f"\noutput: {self.truncate(result)}"
        self._block(f"subagent - {subagent}", details)

    def subagent_cancelled(self, subagent: str, result: str = "") -> None:
        """Print a subagent cancellation."""
        details = "cancelled"
        if result:
            details += f"\noutput: {self.truncate(result)}"
        self._block(f"subagent - {subagent}", details)

    def subagents_cancelled(self) -> None:
        """No-op for non-live terminal output."""
        return None

    def finish_main(self) -> None:
        """Finish the current streamed section."""
        if self._section:
            print()
        self._section = ""

    async def ask_approvals(self, interrupts: list[Any]) -> list[dict[str, Any]]:
        """Ask the user to approve, edit, reject, or respond to interrupted actions."""
        decisions = []
        for interrupt in interrupts:
            for index, action in enumerate(action_requests(interrupt)):
                self._block("approval", action_text(action))
                answer = await self._choice("Approve this action?", action_choices(interrupt, action, index))
                if answer == "e":
                    decisions.append(await self._edit_decision(action))
                elif answer == "r":
                    decisions.append({"type": "reject"})
                elif answer == "s":
                    decisions.append(await self._respond_decision(action))
                else:
                    decisions.append({"type": "approve"})
        return decisions

    async def ask_user(self, interrupt: Any) -> str:
        """Ask the user for a concrete next-step choice."""
        request = ask_user_request(interrupt)
        options = ask_user_options(request)
        self._block("question", ask_user_question(request))
        for index, option in enumerate(options, start=1):
            self._line(f"{index}. {option}")

        answer = await self._input("Choose an option: ")
        try:
            selected = options[int(answer) - 1]
        except (ValueError, IndexError):
            selected = options[-1]

        if selected != ASK_USER_OPEN_OPTION:
            return selected

        response = (await self._input(f"{ASK_USER_OPEN_OPTION}: ")).strip()
        return response or ASK_USER_OPEN_OPTION

    async def ask_create_git_repo(self, message: str) -> bool:
        """Ask whether MIRA should initialize Git for the workspace."""
        return await self._choice(message, [("y", "yes"), ("n", "no")]) == "y"

    async def ask_continue_without_git(self, message: str) -> bool:
        """Ask whether startup should continue without Git protection."""
        return await self._choice(message, [("c", "continue"), ("e", "exit")]) == "c"

    def truncate(self, value: Any) -> str:
        """Return a single-line string shortened to the configured display size."""
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if self.tool_output_chars == 0 or len(text) <= self.tool_output_chars:
            return text
        return text[: self.tool_output_chars] + " ... truncated ..."

    async def _edit_decision(self, action: Any) -> dict[str, Any]:
        """Ask for edited JSON args."""
        if not isinstance(action, dict):
            return {"type": "reject"}

        original = json.dumps(action.get("args", {}), indent=2)
        edited = await self._input(f"Edited args JSON [{original}]: ")
        try:
            args = json.loads(edited or original)
        except json.JSONDecodeError:
            self._line("invalid JSON; rejecting action")
            return {"type": "reject"}

        if not isinstance(args, dict):
            self._line("edited args must be a JSON object; rejecting action")
            return {"type": "reject"}

        return {
            "type": "edit",
            "edited_action": {
                "name": action.get("name", "tool"),
                "args": args,
            },
        }

    async def _respond_decision(self, action: Any) -> dict[str, Any]:
        """Ask for a synthetic successful tool response."""
        message = await self._input("Tool response to return instead: ")
        return {"type": "respond", "message": response_message(message, action)}

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

    def _stream(self, title: str, text: str) -> None:
        """Print streamed text under a simple section heading."""
        if self._section != title:
            if self._section:
                print()
            print(f"\n{title}:")
            self._section = title
        print(text, end="", flush=True)

    def _block(self, title: str, body: str) -> None:
        """Print a titled block."""
        self.finish_main()
        print(f"\n{title}:")
        print(body)

    def _line(self, text: str) -> None:
        """Print one status line."""
        self.finish_main()
        print(text)

    def _next_suffix(self) -> str:
        """Return a readable subagent suffix."""
        return generate_slug(fallback=self._subagent_ids)


def is_compaction_notice(text: str) -> bool:
    """Return whether an info notice is really leaked compaction reasoning."""
    return is_compaction_reasoning(text) or is_compaction_reasoning_fragment(text)
