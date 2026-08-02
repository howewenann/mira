"""One authoritative durable Plan and its execution lifecycle."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, TypedDict

from session.values import strict_items, strict_text, valid_attempts, valid_iterations

PLAN_STATUSES = {
    "proposed",
    "active",
    "paused",
    "max_iterations_reached",
    "completed",
}
PLAN_ARTIFACT_FIELDS = {
    "id",
    "title",
    "objective",
    "context_and_constraints",
    "key_changes",
    "test_plan",
    "assumptions",
    "success_criteria",
    "status",
    "rubric_enabled",
    "rubric_iterations",
    "last_rubric_status",
    "completion_source",
    "attempts",
    "created_at",
    "updated_at",
}
RESUMABLE_PLAN_STATUSES = {
    "proposed",
    "active",
    "paused",
    "max_iterations_reached",
}


class Plan(TypedDict):
    """The user-visible Plan fields."""

    title: str
    objective: str
    context_and_constraints: str
    key_changes: list[str]
    test_plan: list[str]
    assumptions: list[str]


class SuccessCriteria(TypedDict):
    """Success Criteria kept separate from the Plan fields."""

    markdown: str


class PlanArtifact(Plan):
    """The exact retained Plan plus Success Criteria and lifecycle metadata."""

    id: str
    success_criteria: str
    status: str
    rubric_enabled: bool
    rubric_iterations: int
    last_rubric_status: str
    completion_source: str
    attempts: int
    created_at: str
    updated_at: str


def normalize_current_plan(value: Any) -> dict[str, Any] | None:
    """Return an exact current PlanArtifact without legacy coercion."""
    if not isinstance(value, dict):
        return None
    if set(value) != PLAN_ARTIFACT_FIELDS:
        return None
    plan_id = strict_text(value["id"], compact=True)
    title = strict_text(value["title"], compact=True)
    objective = strict_text(value["objective"])
    context = strict_text(value["context_and_constraints"])
    key_changes = strict_items(value["key_changes"])
    test_plan = strict_items(value["test_plan"])
    assumptions = strict_items(value["assumptions"])
    criteria = strict_text(value["success_criteria"])
    if not all((plan_id, title, objective, context, key_changes, test_plan, assumptions, criteria)):
        return None
    status = strict_text(value["status"], compact=True)
    if status not in PLAN_STATUSES:
        return None
    rubric_applies = value["rubric_enabled"]
    rubric_iterations = value["rubric_iterations"]
    attempts = value["attempts"]
    if not isinstance(rubric_applies, bool):
        return None
    if not valid_iterations(rubric_iterations) or not valid_attempts(attempts):
        return None
    last_rubric_status = strict_text(
        value["last_rubric_status"], compact=True, allow_empty=True
    )
    completion_source = strict_text(
        value["completion_source"], compact=True, allow_empty=True
    )
    created_at = strict_text(value["created_at"])
    updated_at = strict_text(value["updated_at"])
    if created_at is None or updated_at is None:
        return None
    return {
        "id": plan_id,
        "title": title,
        "objective": objective,
        "context_and_constraints": context,
        "key_changes": key_changes,
        "test_plan": test_plan,
        "assumptions": assumptions,
        "success_criteria": criteria,
        "status": status,
        "rubric_enabled": rubric_applies,
        "rubric_iterations": rubric_iterations,
        "last_rubric_status": last_rubric_status,
        "completion_source": completion_source,
        "attempts": attempts,
        "created_at": created_at,
        "updated_at": updated_at,
    }


def plan_artifact(
    *,
    plan_id: str,
    title: str,
    objective: str,
    context_and_constraints: str,
    key_changes: list[str],
    test_plan: list[str],
    assumptions: list[str],
    success_criteria: str,
    rubric_enabled: bool,
    rubric_iterations: int,
) -> dict[str, Any]:
    """Create a complete proposed PlanArtifact."""
    now = now_iso()
    value = normalize_current_plan(
        {
            "id": plan_id,
            "title": title,
            "objective": objective,
            "context_and_constraints": context_and_constraints,
            "key_changes": key_changes,
            "test_plan": test_plan,
            "assumptions": assumptions,
            "success_criteria": success_criteria,
            "status": "proposed",
            "rubric_enabled": rubric_enabled,
            "rubric_iterations": rubric_iterations,
            "last_rubric_status": "",
            "completion_source": "",
            "attempts": 0,
            "created_at": now,
            "updated_at": now,
        }
    )
    if value is None:
        raise ValueError("Plan fields and Success Criteria must be complete")
    return value


def current_plan(record: dict[str, Any]) -> dict[str, Any] | None:
    """Return the session's single authoritative Plan."""
    return normalize_current_plan(record.get("current_plan"))


def replace_current_plan(record: dict[str, Any], value: dict[str, Any]) -> dict[str, Any]:
    """Replace the authoritative Plan and supersede any current formal artifact."""
    normalized = normalize_current_plan(value)
    if normalized is None:
        raise ValueError("replacement Plan is incomplete")
    previous_plan = current_plan(record)
    if previous_plan is not None:
        _set_plan_event_status(record, previous_plan["id"], "superseded")
    previous_goal = record.get("current_goal")
    if isinstance(previous_goal, dict):
        _set_goal_event_status(record, str(previous_goal.get("id") or ""), "superseded")
    record["current_goal"] = None
    record["current_plan"] = normalized
    return normalized


def start_plan_attempt(record: dict[str, Any]) -> dict[str, Any] | None:
    """Start or restart execution of the exact retained Plan."""
    value = current_plan(record)
    if value is None:
        return None
    value["status"] = "active"
    value["last_rubric_status"] = ""
    value["completion_source"] = ""
    value["attempts"] += 1
    value["updated_at"] = now_iso()
    record["current_plan"] = value
    _set_plan_event_status(record, value["id"], "active")
    return value


def finish_plan_attempt(
    record: dict[str, Any],
    *,
    rubric_status: str = "",
    outcome: str = "completed",
) -> dict[str, Any] | None:
    """Finish or pause the active Plan attempt from observable runtime state."""
    value = current_plan(record)
    if value is None or value["status"] != "active":
        return value
    status = compact_text(rubric_status)
    if status:
        value["last_rubric_status"] = status
    if outcome != "completed":
        value["status"] = "paused"
    elif not value["rubric_enabled"]:
        value["status"] = "completed"
        value["completion_source"] = "agent-declared"
    elif status == "satisfied":
        value["status"] = "completed"
        value["completion_source"] = "rubric-verified"
    elif status == "max_iterations_reached":
        value["status"] = "max_iterations_reached"
    else:
        value["status"] = "paused"
    value["updated_at"] = now_iso()
    record["current_plan"] = value
    _set_plan_event_status(record, value["id"], value["status"])
    return value


def pause_current_plan(record: dict[str, Any]) -> dict[str, Any] | None:
    """Leave an interrupted or failed active Plan resumable."""
    value = current_plan(record)
    if value is None:
        return None
    if value["status"] == "active":
        value["status"] = "paused"
        value["updated_at"] = now_iso()
        record["current_plan"] = value
        _set_plan_event_status(record, value["id"], "paused")
    return value


def clear_current_plan(record: dict[str, Any]) -> dict[str, Any] | None:
    """Remove only current_plan while retaining immutable transcript events."""
    value = current_plan(record)
    record["current_plan"] = None
    return value


def plan_artifact_text(value: dict[str, Any]) -> str:
    """Render the exact Plan and Success Criteria as binding model context."""
    lines = [
        f"Title: {value.get('title') or 'Plan'}",
        "",
        "Objective:",
        str(value.get("objective") or ""),
        "",
        "Context and Constraints:",
        str(value.get("context_and_constraints") or ""),
    ]
    for heading, key in (
        ("Key Changes", "key_changes"),
        ("Test Plan", "test_plan"),
        ("Assumptions", "assumptions"),
    ):
        lines.extend(["", f"{heading}:"])
        lines.extend(f"- {item}" for item in value.get(key, []))
    lines.extend(["", "Success Criteria:", str(value.get("success_criteria") or "")])
    return "\n".join(lines)


def _set_plan_event_status(record: dict[str, Any], plan_id: str, status: str) -> None:
    for event in record.get("events", []):
        if not isinstance(event, dict):
            continue
        plan = event.get("plan")
        if event.get("type") == "plan" and isinstance(plan, dict) and str(plan.get("id") or "") == plan_id:
            event["status"] = status
            return


def _set_goal_event_status(record: dict[str, Any], goal_id: str, status: str) -> None:
    for event in record.get("events", []):
        if not isinstance(event, dict):
            continue
        goal = event.get("goal")
        if event.get("type") == "goal" and isinstance(goal, dict) and str(goal.get("id") or "") == goal_id:
            event["status"] = status
            return


def compact_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
