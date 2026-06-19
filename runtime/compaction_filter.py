"""Filters for hiding DeepAgents compaction internals from visible output."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from runtime.output_events import (
    could_be_compaction_summary_start,
    normalize_response_delta,
    strip_compaction_summary_prefix,
    text_has_compaction_summary_shape,
)

COMPACTION_REASONING_MARKERS = (
    "context extraction assistant",
    "primary objective",
    "extract the highest quality/most relevant context",
    "conversation history to replace it",
)
COMPACTION_REASONING_HINTS = (
    "context extraction assistant",
    "sole objective",
    "objective_information",
    "extract context from",
    "extract the most important context",
    "extract the most relevant context",
    "extract the highest quality/most relevant context",
    "key information to extract",
    "messages to summarize",
    "conversation history to replace",
    "conversation history will be replaced",
    "conversation history below will be replaced",
    "conversation history",
    "respond only with the extracted context",
    "due to nearing token limits",
    "due to token limits",
    "compact",
    "compaction",
    "summarization",
    "summarize",
    "session intent",
    "artifacts",
    "next steps",
    "output format",
)
COMPACTION_REASONING_START = "thinking process"


class ReasoningFilter:
    """Render reasoning while suppressing DeepAgents compaction internals."""

    def __init__(self, renderer: Any) -> None:
        self.renderer = renderer
        self.pending = ""
        self.observed = ""
        self.probing = True
        self.compacting = False
        self.was_compaction = False
        self.emitted_reasoning = False

    def push(self, delta: str) -> None:
        if not delta or self.compacting:
            return
        self.observed += delta
        if not self.probing:
            if is_compaction_reasoning(self.observed):
                self._start_compaction()
                return
            self.renderer.reasoning_delta(delta)
            self.emitted_reasoning = True
            return

        self.pending += delta
        if is_compaction_reasoning(self.pending):
            self._start_compaction()
            return

        if not could_be_compaction_reasoning_start(self.pending):
            self.renderer.reasoning_delta(self.pending)
            self.emitted_reasoning = True
            self.pending = ""
            self.probing = False

    def finish(self) -> None:
        if self.compacting:
            call_renderer(self.renderer, "compaction_finished")
        elif self.pending and is_compaction_reasoning_fragment(self.pending):
            self._start_compaction()
            call_renderer(self.renderer, "compaction_finished")
        elif self.pending:
            self.renderer.reasoning_delta(self.pending)
            self.emitted_reasoning = True

    def _start_compaction(self) -> None:
        self.compacting = True
        self.was_compaction = True
        self.pending = ""
        if self.emitted_reasoning:
            call_renderer(self.renderer, "discard_reasoning")
        call_renderer(self.renderer, "compaction_started")


class TextFilter:
    """Render assistant text while stripping a leading compaction summary."""

    def __init__(self, renderer: Any, allow_compaction_summary: Callable[[], bool]) -> None:
        self.renderer = renderer
        self.allow_compaction_summary = allow_compaction_summary
        self.pending = ""
        self.probing = True
        self.compacting = False
        self.has_output = False

    def push(self, delta: str) -> None:
        delta = normalize_response_delta("visible" if self.has_output else self.pending, delta)
        if not delta:
            return
        if not self.probing:
            self._emit(delta)
            return

        self.pending += delta
        visible, had_summary = strip_compaction_summary_prefix(self.pending)
        if had_summary:
            if not self.compacting and not self.allow_compaction_summary():
                self.compacting = True
                call_renderer(self.renderer, "compaction_started")
            if visible:
                if self.compacting:
                    call_renderer(self.renderer, "compaction_finished")
                    self.compacting = False
                self._emit(visible)
                self.pending = ""
                self.probing = False
            return

        if not could_be_compaction_summary_start(self.pending):
            self._emit(self.pending)
            self.pending = ""
            self.probing = False

    def finish(self) -> None:
        if self.compacting:
            call_renderer(self.renderer, "compaction_finished")
            self.compacting = False
        if self.probing and self.pending and not text_has_compaction_summary_shape(self.pending):
            self._emit(self.pending)

    def _emit(self, text: str) -> None:
        self.renderer.text_delta(text)
        if text:
            self.has_output = True


def is_compaction_reasoning(text: str) -> bool:
    """Return whether streamed reasoning belongs to DeepAgents compaction."""
    lowered = text.lower()
    if all(marker in lowered for marker in COMPACTION_REASONING_MARKERS):
        return True
    if "context extraction assistant" in lowered and "respond only with the extracted context" in lowered:
        return True
    if "your sole objective" in lowered and "extract" in lowered and "relevant context" in lowered:
        return True
    if "conversation history below will be replaced" in lowered:
        return True
    if "messages to summarize" in lowered and "respond only with the extracted context" in lowered:
        return True
    if "context extraction assistant" in lowered and (
        "conversation history" in lowered or "replace it" in lowered or "token limit" in lowered
    ):
        return True
    if "primary objective" in lowered and "conversation history" in lowered and "replace" in lowered:
        return True
    if "output format" in lowered and "session intent" in lowered and "next steps" in lowered:
        return True
    if "session intent" in lowered and "summary" in lowered and "artifacts" in lowered and "next steps" in lowered:
        return True
    if (
        "extract context from a conversation history" in lowered
        or "extract the most important context" in lowered
        or "extract the most relevant context" in lowered
        or "conversation history has been saved to a file" in lowered
    ) and (
        "session intent" in lowered
        or "condensed summary" in lowered
        or "next steps" in lowered
        or "key information to extract" in lowered
    ):
        return True
    if "conversation history" in lowered and "key information to extract" in lowered:
        return True
    if "conversation history" in lowered and "let me structure this properly" in lowered:
        return True
    if "compact" in lowered and "conversation" in lowered and ("summary" in lowered or "token" in lowered):
        return True
    if "summarization" in lowered and "conversation" in lowered:
        return True
    return False


def is_compaction_reasoning_fragment(text: str) -> bool:
    """Return whether a partial reasoning chunk is likely compaction internals."""
    lowered = text.lower()
    if is_compaction_tail_fragment(lowered):
        return True
    if "context extraction assistant" in lowered:
        return True
    if "respond only with the extracted context" in lowered:
        return True
    if "conversation history" not in lowered:
        return False
    if (
        "extract context from" in lowered
        or "extract the most important context" in lowered
        or "extract the most relevant context" in lowered
        or "extract the highest quality/most relevant context" in lowered
        or "key information to extract" in lowered
        or "let me structure this properly" in lowered
        or "will be replaced" in lowered
    ):
        return True
    if "already been summarized" in lowered and ("meta-task" in lowered or "summary" in lowered):
        return True
    return False


def is_compaction_tail_fragment(text: str) -> bool:
    """Return whether a tiny trailing fragment belongs to hidden compaction reasoning."""
    stripped = text.strip().lower()
    return stripped in {
        ": none - the task is complete",
        "- next steps: none - the task is complete",
        "next steps: none - the task is complete",
        "none - the task is complete",
    }


def should_flush_reasoning_probe(text: str) -> bool:
    """Return whether buffered reasoning is unlikely to be compaction metadata."""
    lowered = text.lower()
    if could_be_compaction_reasoning_start(text):
        return False
    if is_compaction_reasoning_fragment(text):
        return False
    if len(text) >= 1200 and not any(marker in lowered for marker in COMPACTION_REASONING_HINTS):
        return True
    if "\n\n" in text and not any(marker in lowered for marker in COMPACTION_REASONING_HINTS):
        return True
    return False


def could_be_compaction_reasoning_start(text: str) -> bool:
    """Return whether reasoning may still be DeepAgents compaction setup."""
    stripped = text.lstrip().lower()
    if not stripped:
        return True
    if COMPACTION_REASONING_START.startswith(stripped) or stripped.startswith(COMPACTION_REASONING_START):
        return True
    if stripped.startswith(
        (
            "the user wants me to extract context",
            "the user wants me to extract the most important context",
            "the user is asking me to extract context",
            "the user is asking me to extract the most important context",
        )
    ):
        return True
    if any(hint in stripped for hint in COMPACTION_REASONING_HINTS):
        return True
    return False


def call_renderer(renderer: Any, method: str, *args: Any, **kwargs: Any) -> bool:
    """Call an optional renderer method."""
    callback = getattr(renderer, method, None)
    if callback is None:
        return False
    callback(*args, **kwargs)
    return True
