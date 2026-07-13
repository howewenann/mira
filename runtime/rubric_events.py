"""Normalization and rendering for DeepAgents rubric custom events."""

from __future__ import annotations

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

    def __init__(self, renderer: Any, max_iterations: int) -> None:
        self.renderer = renderer
        self.max_iterations = max(1, int(max_iterations or 1))
        self.evaluations: list[dict[str, Any]] = []
        self._latest: dict[str, dict[str, Any]] = {}

    def handle(self, event: dict[str, Any]) -> bool:
        """Render a supported event and return whether it was consumed."""
        event_type = str(event.get("type") or "")
        if event_type == RUBRIC_START:
            run_id = str(event.get("grading_run_id") or "")
            iteration = nonnegative_int(event.get("iteration"))
            call_renderer(
                self.renderer,
                "rubric_evaluation_started",
                run_id,
                iteration + 1,
                self.max_iterations,
            )
            return True
        if event_type != RUBRIC_END:
            return False

        evaluation = normalize_evaluation(event)
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
    return {
        "grading_run_id": str(event.get("grading_run_id") or ""),
        "iteration": nonnegative_int(event.get("iteration")),
        "result": result,
        "explanation": str(event.get("explanation") or "").strip(),
        "criteria": criteria,
    }


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
    if criteria:
        lines.append(f"{passed} of {len(criteria)} criteria satisfied")

    explanation = str(evaluation.get("explanation") or "").strip()
    labels = {
        "satisfied": "Satisfied",
        "needs_revision": "Needs revision",
        "failed": "Review failed",
        "grader_error": "Grader error",
        "max_iterations_reached": "Incomplete: maximum rubric iterations reached",
    }
    detail = labels.get(result, "Review failed")
    lines.append(f"{detail}: {explanation}" if explanation else detail)
    for item in criteria:
        if not isinstance(item, dict) or item.get("passed"):
            continue
        name = str(item.get("name") or "Criterion").strip()
        gap = str(item.get("gap") or "").strip()
        lines.append(f"- {name}: {gap}" if gap else f"- {name}")
    return "\n".join(lines)
