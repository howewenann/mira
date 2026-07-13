"""Trace adapter for the optional sidecar transcript mirror."""

from __future__ import annotations

from logging import Logger
from typing import Any

from ui.terminal_transcript import DEFAULT_TOOL_OUTPUT_CHARS, TerminalTranscript


class TraceStream:
    """Forward TUI renderer callbacks to the shared terminal transcript formatter."""

    def __init__(self, logger: Logger | None = None, *, output_chars: int = DEFAULT_TOOL_OUTPUT_CHARS) -> None:
        self.logger = logger
        self._buffer: list[str] = []
        self.transcript = TerminalTranscript(self._write, tool_output_chars=output_chars)

    @classmethod
    def disabled(cls, *, output_chars: int = DEFAULT_TOOL_OUTPUT_CHARS) -> "TraceStream":
        """Return a no-op trace stream."""
        return cls(None, output_chars=output_chars)

    @property
    def enabled(self) -> bool:
        """Return whether this stream writes trace output."""
        return self.logger is not None

    def startup(self, state: str) -> None:
        """Trace startup progress."""
        self.line(f"startup: {state}")

    def user_message(self, text: str, *, planning: bool = False) -> None:
        """Trace a submitted user message."""
        self._emit("block", "user planning" if planning else "user", text)

    def system_message(self, text: str, *, kind: str = "system") -> None:
        """Trace a system/status/error message."""
        self._emit("system_message", text, kind=kind)

    def command_output(self, value: Any) -> None:
        """Trace command output."""
        self.line(f"command: {self.transcript.truncate(value)}")

    def assistant_delta(self, delta: str) -> None:
        """Buffer streamed assistant text until a transcript boundary."""
        self.transcript.text_delta(delta)

    def reasoning_delta(self, delta: str) -> None:
        """Buffer streamed reasoning until a transcript boundary."""
        self.transcript.reasoning_delta(delta)

    def discard_reasoning(self) -> None:
        """Drop buffered reasoning later classified as internal."""
        self.transcript.discard_reasoning()

    def flush_all(self) -> None:
        """Write any buffered transcript output."""
        self.transcript.finish_main()
        self._flush()

    def compaction_started(self) -> None:
        """Trace context compaction start."""
        self._emit("compaction_started")

    def compaction_finished(self) -> None:
        """Trace context compaction completion."""
        self._emit("compaction_finished")

    def tool_call(self, name: str, args: Any) -> None:
        """Trace a tool call."""
        self._emit("tool_call", name, args)

    def tool_result(self, name: str, result: Any) -> None:
        """Trace a tool result."""
        self._emit("tool_result", name, result)

    def recovered_tool_result(self, name: str, result: Any) -> None:
        """Trace a late-discovered tool result using normal terminal ordering."""
        self.tool_result(name, result)

    def delegation_started(self, calls: list[dict[str, Any]]) -> None:
        """Trace task delegation."""
        self._emit("delegation_started", calls)

    def subagent_started(self, subagent: str, task_input: str = "") -> None:
        """Trace subagent start."""
        self._emit("subagent_started", subagent, task_input)

    def subagent_finished(self, subagent: str, result: str = "") -> None:
        """Trace subagent completion."""
        self._emit("subagent_finished", subagent, result)

    def subagent_cancelled(self, subagent: str, result: str = "") -> None:
        """Trace subagent cancellation."""
        self._emit("subagent_cancelled", subagent, result)

    def rubric_evaluation_started(self, run_id: str, pass_number: int, max_iterations: int) -> None:  # noqa: ARG002
        """Trace rubric evaluation activity."""
        self._emit("rubric_evaluation_started", pass_number, max_iterations)

    def rubric_evaluation_finished(self, evaluation: dict[str, Any], max_iterations: int) -> None:
        """Trace a completed rubric evaluation."""
        self._emit("rubric_evaluation_finished", evaluation, max_iterations)

    def rubric_evaluation_status(
        self,
        run_id: str,
        pass_number: int,
        status: str,
        max_iterations: int,
    ) -> None:  # noqa: ARG002
        """Trace checkpoint status reconciliation."""
        self._emit("rubric_evaluation_status", pass_number, status, max_iterations)

    def line(self, text: str) -> None:
        """Write one plain trace line."""
        if not self.enabled:
            return
        self.transcript.line(text)
        self._flush()

    def _emit(self, method: str, *args: Any, **kwargs: Any) -> None:
        """Call one transcript method, then write completed output."""
        if not self.enabled:
            return
        getattr(self.transcript, method)(*args, **kwargs)
        self._flush()

    def _write(self, text: str) -> None:
        """Collect formatted text until a useful log boundary."""
        if self.enabled:
            self._buffer.append(text)

    def _flush(self) -> None:
        """Write buffered formatted text to diagnostics logging."""
        if not self.enabled or not self._buffer:
            return
        text = "".join(self._buffer).rstrip("\n")
        self._buffer = []
        if text:
            self.logger.info("%s", text)
