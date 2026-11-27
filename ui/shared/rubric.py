"""Shared Rubric text and elapsed-time presentation."""

import json
import re
import time
from collections.abc import Callable
from typing import Any


def nonnegative_int(value: Any) -> int:
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
