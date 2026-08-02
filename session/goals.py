"""One authoritative durable Goal and its execution lifecycle."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, TypedDict

from session.values import valid_artifact

GOAL_STATUSES = {
    "proposed",
    "active",
    "paused",
    "max_iterations_reached",
    "completed",
}
RESUMABLE_GOAL_STATUSES = {
    "proposed",
    "active",
    "paused",
    "max_iterations_reached",
}


class Goal(TypedDict):
    """The user-visible Goal fields."""

    title: str
    objective: str
    success_criteria: str


class GoalArtifact(Goal):
    """The exact retained Goal plus lifecycle metadata."""

    id: str
    status: str
    rubric_enabled: bool
    rubric_iterations: int
    last_rubric_status: str
    completion_source: str
    attempts: int
    created_at: str
    updated_at: str


def normalize_current_goal(value: Any) -> dict[str, Any] | None:
    """Return an exact current GoalArtifact without legacy coercion."""
    if valid_artifact(
        value,
        fields=GoalArtifact.__required_keys__,
        list_fields=(),
        statuses=GOAL_STATUSES,
    ):
        return dict(value)
    return None


def goal_artifact(
    *,
    goal_id: str,
    title: str,
    objective: str,
    success_criteria: str,
    rubric_enabled: bool,
    rubric_iterations: int,
) -> dict[str, Any]:
    """Create a complete proposed GoalArtifact."""
    now = now_iso()
    value = normalize_current_goal(
        {
            "id": goal_id,
            "title": title,
            "objective": objective,
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
        raise ValueError("Goal title, Objective, and Success Criteria must be complete")
    return value


def current_goal(record: dict[str, Any]) -> dict[str, Any] | None:
    """Return the session's single authoritative Goal."""
    return normalize_current_goal(record.get("current_goal"))


def replace_current_goal(record: dict[str, Any], value: dict[str, Any]) -> dict[str, Any]:
    """Replace the authoritative Goal and supersede any current formal artifact."""
    normalized = normalize_current_goal(value)
    if normalized is None:
        raise ValueError("replacement Goal is incomplete")
    previous_goal = current_goal(record)
    if previous_goal is not None:
        _set_goal_event_status(record, previous_goal["id"], "superseded")
    previous_plan = record.get("current_plan")
    if isinstance(previous_plan, dict):
        _set_plan_event_status(record, str(previous_plan.get("id") or ""), "superseded")
    record["current_plan"] = None
    record["current_goal"] = normalized
    return normalized


def start_goal_attempt(record: dict[str, Any]) -> dict[str, Any] | None:
    """Start or restart execution of the exact retained Goal."""
    value = current_goal(record)
    if value is None:
        return None
    value["status"] = "active"
    value["last_rubric_status"] = ""
    value["completion_source"] = ""
    value["attempts"] += 1
    value["updated_at"] = now_iso()
    record["current_goal"] = value
    _set_goal_event_status(record, value["id"], "active")
    return value


def finish_goal_attempt(
    record: dict[str, Any],
    *,
    rubric_status: str = "",
    outcome: str = "completed",
) -> dict[str, Any] | None:
    """Finish or pause the active Goal attempt from observable runtime state."""
    value = current_goal(record)
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
    record["current_goal"] = value
    _set_goal_event_status(record, value["id"], value["status"])
    return value


def pause_current_goal(record: dict[str, Any]) -> dict[str, Any] | None:
    """Leave an interrupted or failed active Goal resumable."""
    value = current_goal(record)
    if value is not None and value["status"] == "active":
        value["status"] = "paused"
        value["updated_at"] = now_iso()
        record["current_goal"] = value
        _set_goal_event_status(record, value["id"], "paused")
    return value


def clear_current_goal(record: dict[str, Any]) -> dict[str, Any] | None:
    """Remove only current_goal while retaining immutable transcript events."""
    value = current_goal(record)
    record["current_goal"] = None
    return value


def goal_artifact_text(value: dict[str, Any]) -> str:
    """Render the exact Goal as binding model context."""
    return "\n".join(
        [
            f"Title: {value.get('title') or 'Goal'}",
            "",
            "Objective:",
            str(value.get("objective") or ""),
            "",
            "Success Criteria:",
            str(value.get("success_criteria") or ""),
        ]
    )


def _set_goal_event_status(record: dict[str, Any], goal_id: str, status: str) -> None:
    for event in record.get("events", []):
        goal = event.get("goal") if isinstance(event, dict) else None
        if event.get("type") == "goal" and isinstance(goal, dict) and str(goal.get("id") or "") == goal_id:
            event["status"] = status
            return


def _set_plan_event_status(record: dict[str, Any], plan_id: str, status: str) -> None:
    for event in record.get("events", []):
        plan = event.get("plan") if isinstance(event, dict) else None
        if event.get("type") == "plan" and isinstance(plan, dict) and str(plan.get("id") or "") == plan_id:
            event["status"] = status
            return


def compact_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
