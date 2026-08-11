"""Normalization and rendering for DeepAgents rubric custom events."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from runtime.renderer_calls import call_renderer

RUBRIC_START = "rubric_evaluation_start"
RUBRIC_END = "rubric_evaluation_end"
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

    def handle(self, event: dict[str, Any]) -> bool:
        """Render a supported event and return whether it was consumed."""
        event_type = str(event.get("type") or "")
        if event_type == RUBRIC_START:
            run_id = str(event.get("grading_run_id") or "")
            iteration = nonnegative_int(event.get("iteration"))
            self._started_at[(run_id, iteration)] = self.clock()
            call_renderer(
                self.renderer,
                "rubric_evaluation_started",
                run_id,
                iteration + 1,
                self.max_iterations,
                grader_model=self.grader_model,
            )
            return True
        if event_type != RUBRIC_END:
            return False

        evaluation = normalize_evaluation(event)
        if self.grader_model:
            evaluation["grader_model"] = self.grader_model
        started_at = self._started_at.pop(
            (evaluation["grading_run_id"], evaluation["iteration"]),
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
        if not self._started_at:
            return
        self._started_at.clear()
        call_renderer(self.renderer, "rubric_evaluations_cancelled")


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
    result = str(evaluation.get("result") or "failed")
    criteria = evaluation.get("criteria") if isinstance(evaluation.get("criteria"), list) else []
    passed = sum(1 for item in criteria if isinstance(item, dict) and item.get("passed"))
    lines = [f"Rubric review · pass {iteration} of {max_iterations}"]
    grader_model = str(evaluation.get("grader_model") or "").strip()
    if grader_model:
        lines.append(f"Grader: {grader_model}")
    duration_ms = evaluation.get("duration_ms")
    if isinstance(duration_ms, (int, float)) and not isinstance(duration_ms, bool):
        lines.append(f"Completed in {format_elapsed(duration_ms)}")
    if grader_model or isinstance(duration_ms, (int, float)):
        lines.append("")
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
