"""Normalization and projection for DeepAgents rubric custom events."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from typing import Any

from core.execution.streams.output import call_renderer
from core.execution.streams.tool_args import ToolCallDrafts

RUBRIC_START = "rubric_evaluation_start"
RUBRIC_END = "rubric_evaluation_end"
RUBRIC_VERIFICATION_START = "rubric_verification_start"
RUBRIC_VERIFICATION_END = "rubric_verification_end"
RUBRIC_GRADING_START = "rubric_grading_start"
RUBRIC_GRADING_END = "rubric_grading_end"
RUBRIC_TOOL_CALL_DELTA = "rubric_tool_call_delta"
RUBRIC_TOOL_START = "rubric_tool_start"
RUBRIC_TOOL_END = "rubric_tool_end"
RUBRIC_RESULTS = {
    "satisfied",
    "needs_revision",
    "failed",
    "grader_error",
    "max_iterations_reached",
}


class RubricEventRenderer:
    """Project rubric custom events onto dedicated renderer callbacks."""

    def __init__(
        self,
        renderer: Any,
        max_iterations: int,
        *,
        grader_model: str = "",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.renderer = renderer
        self.max_iterations = max(1, int(max_iterations or 1))
        self.grader_model = str(grader_model or "").strip()
        self.clock = clock
        self.evaluations: list[dict[str, Any]] = []
        self._latest: dict[str, dict[str, Any]] = {}
        self._started_at: dict[tuple[str, int], float] = {}
        self._verifier_started_at: dict[tuple[str, int], float] = {}
        self._grader_started_at: dict[tuple[str, int], float] = {}
        self._verifier_status: dict[tuple[str, int], str] = {}
        self._grader_status: dict[tuple[str, int], str] = {}
        self._verifier_duration_ms: dict[tuple[str, int], int] = {}
        self._grader_duration_ms: dict[tuple[str, int], int] = {}
        self._verifier_tools: dict[tuple[str, int], list[dict[str, Any]]] = {}
        self._verifier_tools_by_id: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
        self._tool_drafts: dict[tuple[str, int], ToolCallDrafts] = {}

    def handle(self, event: dict[str, Any]) -> bool:
        """Render a supported event and return whether it was consumed."""
        event_type = str(event.get("type") or "")
        if event_type == RUBRIC_START:
            run_id = str(event.get("grading_run_id") or "")
            iteration = nonnegative_int(event.get("iteration"))
            key = (run_id, iteration)
            self._started_at[key] = self.clock()
            call_renderer(
                self.renderer,
                "rubric_evaluation_started",
                run_id,
                iteration + 1,
                self.max_iterations,
                grader_model=self.grader_model,
                phase="verifying",
            )
            return True
        if event_type == RUBRIC_VERIFICATION_START:
            key = rubric_event_key(event)
            self._verifier_started_at[key] = self.clock()
            self._verifier_status[key] = "running"
            call_renderer(
                self.renderer,
                "rubric_lifecycle_event",
                normalized_lifecycle_event(event),
            )
            return True
        if event_type == RUBRIC_TOOL_CALL_DELTA:
            key = rubric_event_key(event)
            drafts = self._tool_drafts.get(key)
            if drafts is None:
                drafts = ToolCallDrafts(_RubricDraftRenderer(self.renderer, key))
                self._tool_drafts[key] = drafts
            drafts.push(event.get("chunk"))
            return True
        if event_type == RUBRIC_TOOL_START:
            key = rubric_event_key(event)
            tool = normalize_verifier_tool(event)
            call_id = tool["call_id"]
            tools = self._verifier_tools.setdefault(key, [])
            by_id = self._verifier_tools_by_id.setdefault(key, {})
            existing = by_id.get(call_id) if call_id else None
            if existing is None:
                tools.append(tool)
                if call_id:
                    by_id[call_id] = tool
            else:
                existing.update(tool)
            call_renderer(
                self.renderer,
                "rubric_lifecycle_event",
                normalized_lifecycle_event(event),
            )
            return True
        if event_type == RUBRIC_TOOL_END:
            key = rubric_event_key(event)
            call_id = str(event.get("tool_call_id") or "")
            by_id = self._verifier_tools_by_id.setdefault(key, {})
            tool = by_id.get(call_id) if call_id else None
            if tool is None:
                tool = normalize_verifier_tool(event)
                self._verifier_tools.setdefault(key, []).append(tool)
                if call_id:
                    by_id[call_id] = tool
            tool.update(
                {
                    "output": str(event.get("output") or ""),
                    "is_error": bool(event.get("is_error")),
                    "duration_ms": optional_duration_ms(event.get("duration_ms")),
                }
            )
            call_renderer(
                self.renderer,
                "rubric_lifecycle_event",
                normalized_lifecycle_event(event),
            )
            return True
        if event_type == RUBRIC_VERIFICATION_END:
            key = rubric_event_key(event)
            self._verifier_status[key] = "complete" if event.get("succeeded") else "failed"
            started_at = self._verifier_started_at.pop(key, None)
            if started_at is not None:
                self._verifier_duration_ms[key] = elapsed_ms(started_at, clock=self.clock)
            call_renderer(
                self.renderer,
                "rubric_lifecycle_event",
                {
                    **normalized_lifecycle_event(event),
                    "duration_ms": self._verifier_duration_ms.get(key),
                },
            )
            return True
        if event_type == RUBRIC_GRADING_START:
            key = rubric_event_key(event)
            self._grader_started_at[key] = self.clock()
            self._grader_status[key] = "running"
            call_renderer(
                self.renderer,
                "rubric_lifecycle_event",
                normalized_lifecycle_event(event),
            )
            return True
        if event_type == RUBRIC_GRADING_END:
            key = rubric_event_key(event)
            self._grader_status[key] = "complete" if event.get("succeeded") else "failed"
            started_at = self._grader_started_at.pop(key, None)
            if started_at is not None:
                self._grader_duration_ms[key] = elapsed_ms(started_at, clock=self.clock)
            call_renderer(
                self.renderer,
                "rubric_lifecycle_event",
                {
                    **normalized_lifecycle_event(event),
                    "duration_ms": self._grader_duration_ms.get(key),
                },
            )
            return True
        if event_type != RUBRIC_END:
            return False

        evaluation = normalize_evaluation(event)
        key = (evaluation["grading_run_id"], evaluation["iteration"])
        evaluation["verifier_status"] = self._verifier_status.pop(key, "complete")
        grader_status = self._grader_status.pop(key, None)
        if grader_status is not None:
            evaluation["grader_status"] = grader_status
        elif evaluation["verifier_status"] != "failed":
            evaluation["grader_status"] = (
                "failed" if evaluation["result"] == "grader_error" else "complete"
            )
        evaluation["verifier_tools"] = [
            dict(tool) for tool in self._verifier_tools.pop(key, [])
        ]
        verifier_duration = self._verifier_duration_ms.pop(key, None)
        if verifier_duration is not None:
            evaluation["verifier_duration_ms"] = verifier_duration
        grader_duration = self._grader_duration_ms.pop(key, None)
        if grader_duration is not None:
            evaluation["grader_duration_ms"] = grader_duration
        self._verifier_tools_by_id.pop(key, None)
        self._tool_drafts.pop(key, None)
        if self.grader_model:
            evaluation["grader_model"] = self.grader_model
        started_at = self._started_at.pop(
            key,
            None,
        )
        if started_at is not None:
            evaluation["duration_ms"] = elapsed_ms(started_at, clock=self.clock)
        self.evaluations.append(evaluation)
        self._latest[evaluation["grading_run_id"]] = evaluation
        call_renderer(
            self.renderer,
            "rubric_evaluation_finished",
            evaluation,
            self.max_iterations,
        )
        return True

    def finalize(self, status: str) -> None:
        """Reconcile the final checkpoint status with the last streamed verdict."""
        if not status or not self._latest:
            return
        latest = next(reversed(self._latest.values()))
        if latest.get("result") == status:
            return
        latest["result"] = status
        call_renderer(
            self.renderer,
            "rubric_evaluation_status",
            latest["grading_run_id"],
            latest["iteration"] + 1,
            status,
            self.max_iterations,
        )

    def cancel(self) -> None:
        """Stop transient activity when an invocation ends without grader output."""
        if not (
            self._started_at
            or self._verifier_started_at
            or self._grader_started_at
        ):
            return
        self._started_at.clear()
        self._verifier_started_at.clear()
        self._grader_started_at.clear()
        self._verifier_status.clear()
        self._grader_status.clear()
        self._verifier_duration_ms.clear()
        self._grader_duration_ms.clear()
        self._verifier_tools.clear()
        self._verifier_tools_by_id.clear()
        self._tool_drafts.clear()
        call_renderer(self.renderer, "rubric_evaluations_cancelled")


class _RubricDraftRenderer:
    """Scope normal `ToolCallDrafts` callbacks to one Rubric pass."""

    def __init__(self, renderer: Any, key: tuple[str, int]) -> None:
        self.renderer = renderer
        self.key = key

    def tool_call_delta(self, name: str, args: Any, call_id: str = "") -> None:
        call_renderer(
            self.renderer,
            "rubric_lifecycle_event",
            {
                "type": RUBRIC_TOOL_CALL_DELTA,
                "grading_run_id": self.key[0],
                "iteration": self.key[1],
                "tool_call_id": call_id,
                "tool_name": name,
                "tool_args": args,
            },
        )


def rubric_event_key(event: dict[str, Any]) -> tuple[str, int]:
    """Return the stable Rubric pass identity carried by a custom event."""
    return (
        str(event.get("grading_run_id") or ""),
        nonnegative_int(event.get("iteration")),
    )


def normalized_lifecycle_event(event: dict[str, Any]) -> dict[str, Any]:
    """Keep one lifecycle payload JSON-safe without truncating evidence."""
    normalized = dict(event)
    normalized["grading_run_id"], normalized["iteration"] = rubric_event_key(event)
    if "tool_call_id" in normalized:
        normalized["tool_call_id"] = str(normalized.get("tool_call_id") or "")
    if "tool_name" in normalized:
        normalized["tool_name"] = str(normalized.get("tool_name") or "tool")
    if "output" in normalized:
        normalized["output"] = str(normalized.get("output") or "")
    return normalized


def normalize_verifier_tool(event: dict[str, Any]) -> dict[str, Any]:
    """Return one full-fidelity verifier tool lifecycle record."""
    return {
        "call_id": str(event.get("tool_call_id") or ""),
        "name": str(event.get("tool_name") or "tool"),
        "args": event.get("tool_args", {}),
        "output": str(event.get("output") or ""),
        "is_error": bool(event.get("is_error")),
        "duration_ms": optional_duration_ms(event.get("duration_ms")),
    }


def optional_duration_ms(value: Any) -> int | None:
    """Normalize an optional duration without accepting booleans."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return max(0, round(value))
    return None


def normalize_evaluation(event: dict[str, Any]) -> dict[str, Any]:
    """Return a stable, JSON-safe rubric evaluation."""
    result = str(event.get("result") or "failed")
    if result not in RUBRIC_RESULTS:
        result = "failed"
    criteria = []
    raw_criteria = event.get("criteria")
    if isinstance(raw_criteria, list):
        for raw in raw_criteria:
            if not isinstance(raw, dict):
                continue
            criteria.append(
                {
                    "name": str(raw.get("name") or "Criterion").strip(),
                    "passed": bool(raw.get("passed")),
                    "gap": str(raw.get("gap") or "").strip(),
                }
            )
    evaluation = {
        "grading_run_id": str(event.get("grading_run_id") or ""),
        "iteration": nonnegative_int(event.get("iteration")),
        "result": result,
        "explanation": str(event.get("explanation") or "").strip(),
        "criteria": criteria,
    }
    diagnostics = rubric_diagnostics(event)
    if diagnostics:
        evaluation["diagnostics"] = diagnostics
    return evaluation


def rubric_diagnostics(event: dict[str, Any]) -> dict[str, Any]:
    """Preserve structured DeepAgents grader diagnostics when supplied."""
    diagnostics: dict[str, Any] = {}
    configured_model = event.get("configured_model", event.get("rubric_grader_configured_model"))
    if isinstance(configured_model, str) and configured_model.strip():
        diagnostics["configured_model"] = configured_model.strip()
    strategy = event.get("structured_output_strategy", event.get("rubric_grader_effective_strategy"))
    if isinstance(strategy, str) and strategy.strip():
        diagnostics["structured_output_strategy"] = strategy.strip()
    http_status = event.get("http_status", event.get("status_code"))
    if isinstance(http_status, int) and not isinstance(http_status, bool):
        diagnostics["http_status"] = http_status
    return diagnostics


def nonnegative_int(value: Any) -> int:
    """Return a safe zero-based integer."""
    return max(0, value if isinstance(value, int) and not isinstance(value, bool) else 0)


def rubric_result_text(evaluation: dict[str, Any], max_iterations: int) -> str:
    """Return concise human-readable evaluation text without raw JSON."""
    iteration = nonnegative_int(evaluation.get("iteration")) + 1
    lines = [f"Rubric review · pass {iteration} of {max_iterations}"]
    grader_model = str(evaluation.get("grader_model") or "").strip()
    has_lifecycle = bool(
        evaluation.get("verifier_status")
        or evaluation.get("grader_status")
        or isinstance(evaluation.get("verifier_tools"), list)
    )
    if has_lifecycle:
        if grader_model:
            lines.append(f"Model: {grader_model}")
        lines.extend(
            (
                "",
                lifecycle_heading(
                    "Verifier",
                    evaluation.get("verifier_status"),
                    evaluation.get("verifier_duration_ms"),
                ),
            )
        )
        tools = evaluation.get("verifier_tools")
        if isinstance(tools, list) and tools:
            for tool in tools:
                if not isinstance(tool, dict):
                    continue
                args = compact_json(tool.get("args", {}))
                lines.extend(("", f"tool - {tool.get('name') or 'tool'} · call: {args}"))
                output = terminal_tool_preview(tool.get("output"))
                if output:
                    lines.append(output)
                failed = bool(tool.get("is_error"))
                duration = tool.get("duration_ms")
                verb = "Failed after" if failed else "Completed in"
                lines.append(
                    f"{verb} {format_elapsed(duration)}"
                    if isinstance(duration, (int, float)) and not isinstance(duration, bool)
                    else ("Failed" if failed else "Completed")
                )
        else:
            lines.extend(("", "No tools called."))
        if evaluation.get("verifier_status") != "failed" or evaluation.get("grader_status"):
            lines.extend(
                (
                    "",
                    lifecycle_heading(
                        "Grader",
                        evaluation.get("grader_status"),
                        evaluation.get("grader_duration_ms"),
                    ),
                )
            )
        lines.append("")
    else:
        if grader_model:
            lines.append(f"Grader: {grader_model}")
        duration_ms = evaluation.get("duration_ms")
        if isinstance(duration_ms, (int, float)) and not isinstance(duration_ms, bool):
            lines.append(f"Completed in {format_elapsed(duration_ms)}")
        if grader_model or isinstance(duration_ms, (int, float)):
            lines.append("")
    lines.extend(rubric_result_body_text(evaluation).splitlines())
    return "\n".join(lines)


def rubric_result_body_text(evaluation: dict[str, Any]) -> str:
    """Return the existing criterion and verdict body without lifecycle headings."""
    result = str(evaluation.get("result") or "failed")
    criteria = evaluation.get("criteria") if isinstance(evaluation.get("criteria"), list) else []
    passed = sum(1 for item in criteria if isinstance(item, dict) and item.get("passed"))
    lines: list[str] = []
    if criteria:
        lines.append(f"{passed} of {len(criteria)} criteria satisfied")
        for item in criteria:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "Criterion").strip()
            if item.get("passed"):
                lines.append(f"✓ {name}")
                continue
            gap = str(item.get("gap") or "").strip()
            lines.append(f"✗ {name}: {gap}" if gap else f"✗ {name}")

    explanation = str(evaluation.get("explanation") or "").strip()
    labels = {
        "satisfied": "Satisfied",
        "needs_revision": "Needs revision",
        "failed": "Review failed",
        "grader_error": "Grader error",
        "max_iterations_reached": "Incomplete: maximum rubric iterations reached",
    }
    detail = labels.get(result, "Review failed")
    if criteria:
        lines.append("")
    lines.append(f"{detail}: {explanation}" if explanation else detail)
    diagnostics = evaluation.get("diagnostics")
    if isinstance(diagnostics, dict) and diagnostics:
        values = []
        if diagnostics.get("configured_model"):
            values.append(f"model={diagnostics['configured_model']}")
        if diagnostics.get("structured_output_strategy"):
            values.append(f"strategy={diagnostics['structured_output_strategy']}")
        if isinstance(diagnostics.get("http_status"), int):
            values.append(f"HTTP {diagnostics['http_status']}")
        if values:
            lines.append(f"Grader diagnostics: {', '.join(values)}")
    return "\n".join(lines)


def lifecycle_heading(name: str, status: Any, duration_ms: Any) -> str:
    """Format one completed Rubric phase heading."""
    label = "Failed" if str(status or "complete") == "failed" else "Complete"
    heading = f"{name} · {label}"
    if isinstance(duration_ms, (int, float)) and not isinstance(duration_ms, bool):
        heading += f" · {format_elapsed(duration_ms)}"
    return heading


def compact_json(value: Any, limit: int = 240) -> str:
    """Render a bounded one-line argument preview outside the interactive TUI."""
    if isinstance(value, str):
        text = re.sub(r"\s+", " ", value).strip()
    else:
        try:
            text = json.dumps(
                value,
                ensure_ascii=False,
                separators=(", ", ": "),
                default=str,
            )
        except (TypeError, ValueError):
            text = re.sub(r"\s+", " ", str(value)).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def terminal_tool_preview(value: Any, limit: int = 240) -> str:
    """Bound non-widget output while the TUI uses its actual available width."""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def format_elapsed(duration_ms: int | float) -> str:
    """Format a monotonic duration for stable terminal and session replay."""
    total_seconds = max(0, int(float(duration_ms) / 1000))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def elapsed_ms(
    started_at: float,
    *,
    clock: Callable[[], float] | None = None,
) -> int:
    """Return a non-negative whole-millisecond monotonic duration."""
    active_clock = clock or time.monotonic
    return max(0, round((active_clock() - float(started_at)) * 1000))
