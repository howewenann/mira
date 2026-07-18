"""Shared plain terminal transcript formatting."""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from runtime.rubric_events import rubric_result_text
from ui.names import generate_slug

DEFAULT_TOOL_OUTPUT_CHARS = 240


class TerminalTranscript:
    """Format streamed agent activity as plain terminal text."""

    def __init__(
        self,
        write: Callable[[str], None],
        *,
        tool_output_chars: int = DEFAULT_TOOL_OUTPUT_CHARS,
        slug_fallback: Any = None,
    ) -> None:
        self.write = write
        self.tool_output_chars = int(tool_output_chars)
        self._section = ""
        self._reasoning_text = ""
        self._pending_tool_results: list[tuple[str, Any, bool]] = []
        self._subagent_labels: dict[int, str] = {}
        self._slug_fallback = slug_fallback

    def reasoning_delta(self, delta: str) -> None:
        """Buffer streamed reasoning text for one clean block."""
        text = re.sub(r"</?[^>]+>", "", str(delta or ""))
        if text.strip() or self._reasoning_text:
            self._reasoning_text += text

    def discard_reasoning(self) -> None:
        """Discard pending reasoning that should not be displayed."""
        self._reasoning_text = ""

    def text_delta(self, delta: str) -> None:
        """Write streamed assistant text under a single mira heading."""
        if delta:
            self._flush_reasoning()
            self.stream("mira", str(delta))

    def tool_call(self, name: str, args: Any) -> None:
        """Write a compact tool call block."""
        self.block(name, f"args: {self.truncate(args)}")

    def tool_result(self, name: str, result: Any) -> None:
        """Write a compact tool result line."""
        if result:
            self.line(f"{name} output: {self.truncate(result)}")

    def tool_error(self, name: str, error: Any) -> None:
        """Write a compact failed tool line."""
        if error:
            self.line(f"{name} error: {self.truncate(error)}")

    def completed_tool_result(self, name: str, result: Any) -> None:
        """Write a completion now, or defer it past active streamed model text."""
        if not result:
            return
        if self._section == "mira" or self._reasoning_text:
            self._pending_tool_results.append((name, result, False))
            return
        self.tool_result(name, result)

    def completed_tool_error(self, name: str, error: Any) -> None:
        """Write a failure now, or defer it past active streamed model text."""
        if not error:
            return
        if self._section == "mira" or self._reasoning_text:
            self._pending_tool_results.append((name, error, True))
            return
        self.tool_error(name, error)

    def delegation_started(self, calls: list[dict[str, Any]]) -> None:
        """Write a compact task delegation block."""
        descriptions = delegation_descriptions(calls)
        if descriptions:
            lines = [f"delegating to {len(descriptions)} subagent(s)"]
            lines.extend(f"request: {self.truncate(description)}" for description in descriptions)
            self.block("task", "\n".join(lines))

    def system_message(self, text: str, *, kind: str = "system") -> None:
        """Write a system/status/error block."""
        self.block(kind, text)

    def compaction_started(self) -> None:
        """Write a context compaction status."""
        self.block("mira", "compacting context...")

    def compaction_finished(self) -> None:
        """Write a context compaction completion line."""
        self.line("context compacted")

    def subagent_label(self, subagent: Any) -> str:
        """Return a stable readable label for a subagent object."""
        key = id(subagent)
        if key not in self._subagent_labels:
            name = getattr(subagent, "name", "subagent")
            self._subagent_labels[key] = f"{name} [{self._next_suffix()}]"
        return self._subagent_labels[key]

    def subagent_started(self, subagent: str, task_input: str = "", *, origin: str = "") -> None:
        """Write a subagent start block."""
        detail = f"request: {self.truncate(task_input)}" if task_input else "running"
        self.block(subagent_title(subagent), detail)

    def subagent_request_updated(self, subagent: str, task_input: str) -> None:
        """Write a late-arriving request for an already-started subagent."""
        if task_input:
            self.block(subagent_title(subagent), f"request: {self.truncate(task_input)}")

    def subagent_finished(self, subagent: str, result: Any = "") -> None:
        """Write a subagent completion block."""
        detail = "done"
        if result:
            detail += f"\noutput: {self.truncate(result)}"
        self.block(subagent_title(subagent), detail)

    def subagent_cancelled(self, subagent: str, result: Any = "") -> None:
        """Write a subagent cancellation block."""
        detail = "cancelled"
        if result:
            detail += f"\noutput: {self.truncate(result)}"
        self.block(subagent_title(subagent), detail)

    def rubric_evaluation_started(self, pass_number: int, max_iterations: int) -> None:
        """Write immediate rubric review activity."""
        self.block(
            "rubric review",
            f"Reviewing completion criteria · pass {pass_number} of {max_iterations}…",
        )

    def rubric_evaluation_finished(self, evaluation: dict[str, Any], max_iterations: int) -> None:
        """Write a concise rubric result."""
        self.block("rubric review", rubric_result_text(evaluation, max_iterations))

    def rubric_evaluation_status(self, pass_number: int, status: str, max_iterations: int) -> None:
        """Write a corrected terminal status when the checkpoint differs."""
        if status == "max_iterations_reached":
            self.block(
                "rubric review",
                f"Rubric review · pass {pass_number} of {max_iterations}\n"
                "Incomplete: maximum rubric iterations reached",
            )

    def finish_main(self) -> None:
        """Finish the current streamed section."""
        self._flush_reasoning()
        if self._section:
            self.write("\n")
        self._section = ""
        pending = self._pending_tool_results
        self._pending_tool_results = []
        for name, result, is_error in pending:
            label = "error" if is_error else "output"
            self.write(f"{name} {label}: {self.truncate(result)}\n")

    def stream(self, title: str, text: str) -> None:
        """Write streamed text under a simple section heading."""
        if self._section != title:
            if self._section:
                self.write("\n")
            self.write(f"\n{title}:\n")
            self._section = title
        self.write(text)

    def block(self, title: str, body: Any) -> None:
        """Write one titled block."""
        self.finish_main()
        self.write(f"\n{title}:\n")
        self.write(f"{body}\n")

    def line(self, text: str) -> None:
        """Write one status line."""
        self.finish_main()
        self.write(f"{text}\n")

    def truncate(self, value: Any) -> str:
        """Return a compact one-line value."""
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if self.tool_output_chars == 0 or len(text) <= self.tool_output_chars:
            return text
        return f"{text[: self.tool_output_chars]} ... truncated ..."

    def _flush_reasoning(self) -> None:
        """Write buffered reasoning once, preserving line breaks."""
        if not self._reasoning_text.strip():
            self._reasoning_text = ""
            return
        if self._section:
            self.write("\n")
            self._section = ""
        self.write("\nthinking:\n")
        self.write(self._reasoning_text)
        if not self._reasoning_text.endswith("\n"):
            self.write("\n")
        self._reasoning_text = ""

    def _next_suffix(self) -> str:
        """Return a readable subagent suffix."""
        return generate_slug(fallback=self._slug_fallback)


def delegation_descriptions(calls: list[dict[str, Any]]) -> list[str]:
    """Return task delegation descriptions from tool-call payloads."""
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
    return descriptions


def subagent_title(subagent: str) -> str:
    """Return the terminal title for a subagent block."""
    return f"subagent - {subagent}"
