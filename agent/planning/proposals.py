"""Small helpers for canonical goal-and-plan proposal state."""

from __future__ import annotations

from typing import Any

PROPOSAL_ORIGINS = {"plan_mode", "goal_command"}


def effective_objective(original: str, decisions: list[dict[str, str]] | None = None) -> str:
    """Return an objective with resolved planning decisions kept explicit."""
    original = str(original or "").strip()
    resolved = [item for item in decisions or [] if item.get("answer")]
    if not resolved:
        return original
    lines = [original, "", "Resolved planning decisions:"]
    for item in resolved:
        question = str(item.get("question") or "Decision").strip()
        answer = str(item.get("answer") or "").strip()
        lines.append(f"- {question}: {answer}")
    return "\n".join(lines)


def proposal(
    *,
    proposal_id: str,
    origin: str,
    original_objective: str,
    decisions: list[dict[str, str]] | None,
    criteria: str,
    plan: dict[str, Any],
    rubric_iterations: int,
) -> dict[str, Any]:
    """Return the complete proposal shape used by both entry paths."""
    if origin not in PROPOSAL_ORIGINS:
        raise ValueError(f"unsupported proposal origin: {origin}")
    if not isinstance(plan, dict):
        raise TypeError("new proposals require a structured plan")
    resolved = [dict(item) for item in decisions or []]
    return {
        "id": proposal_id,
        "original_objective": str(original_objective or "").strip(),
        "objective": effective_objective(original_objective, resolved),
        "resolved_decisions": resolved,
        "criteria": str(criteria or "").strip(),
        "plan": dict(plan),
        "origin": origin,
        "rubric_iterations": int(rubric_iterations),
    }


def proposal_title(value: dict[str, Any]) -> str:
    """Return a compact title for status messages."""
    plan = value.get("plan")
    if isinstance(plan, dict) and plan.get("title"):
        return str(plan["title"])
    objective = " ".join(str(value.get("original_objective") or "Goal").split())
    return objective[:60].rstrip() or "Goal"
