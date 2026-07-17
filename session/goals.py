"""Durable active-goal state derived from approved proposals."""

from __future__ import annotations

from typing import Any

ACTIVE_GOAL_STATUSES = {"active", "complete", "cleared", "superseded"}


def normalize_active_goal(value: Any) -> dict[str, Any] | None:
    """Return a safe active-goal record without inferring one from history."""
    if not isinstance(value, dict):
        return None
    plan = value.get("plan")
    if not isinstance(plan, dict):
        return None
    proposal_id = str(value.get("proposal_id") or value.get("id") or "").strip()
    objective = str(value.get("objective") or "").strip()
    criteria = str(value.get("criteria") or "").strip()
    if not proposal_id or not objective or not criteria:
        return None
    status = str(value.get("status") or "active").strip()
    if status not in ACTIVE_GOAL_STATUSES:
        status = "active"
    origin = str(value.get("origin") or "goal_command")
    if origin not in {"plan_mode", "goal_command"}:
        origin = "goal_command"
    raw_decisions = value.get("resolved_decisions")
    decisions = raw_decisions if isinstance(raw_decisions, list) else []
    return {
        "proposal_id": proposal_id,
        "id": str(value.get("id") or proposal_id),
        "original_objective": str(value.get("original_objective") or objective).strip(),
        "objective": objective,
        "resolved_decisions": [dict(item) for item in decisions if isinstance(item, dict)],
        "criteria": criteria,
        "plan": dict(plan),
        "origin": origin,
        "rubric_iterations": bounded_iterations(value.get("rubric_iterations")),
        "status": status,
        "last_rubric_status": str(value.get("last_rubric_status") or "").strip(),
    }


def current_active_goal(record: dict[str, Any]) -> dict[str, Any] | None:
    """Return the authoritative goal only while it remains active."""
    value = normalize_active_goal(record.get("active_goal"))
    return value if value and value["status"] == "active" else None


def activate_goal(record: dict[str, Any], proposal: dict[str, Any]) -> dict[str, Any]:
    """Persist an approved proposal and explicitly supersede a prior goal."""
    previous = current_active_goal(record)
    if previous is not None:
        _set_proposal_status(record, previous["proposal_id"], "superseded")
    value = normalize_active_goal(
        {
            **proposal,
            "proposal_id": proposal.get("id"),
            "status": "active",
            "last_rubric_status": "",
        }
    )
    if value is None:
        raise ValueError("approved proposal is incomplete")
    record["active_goal"] = value
    return value


def update_active_goal_after_turn(record: dict[str, Any], rubric_status: str) -> dict[str, Any] | None:
    """Persist the latest rubric result and complete only satisfied goals."""
    value = current_active_goal(record)
    if value is None:
        return None
    status = str(rubric_status or "").strip()
    value["last_rubric_status"] = status
    if status == "satisfied":
        value["status"] = "complete"
        _set_proposal_status(record, value["proposal_id"], "complete")
    record["active_goal"] = value
    return value


def clear_active_goal(record: dict[str, Any]) -> dict[str, Any] | None:
    """Clear the active goal without deleting proposal or rubric history."""
    value = current_active_goal(record)
    if value is None:
        return None
    value["status"] = "cleared"
    record["active_goal"] = value
    _set_proposal_status(record, value["proposal_id"], "cleared")
    return value


def _set_proposal_status(record: dict[str, Any], proposal_id: str, status: str) -> None:
    for event in record.get("events", []):
        if not isinstance(event, dict):
            continue
        proposal = event.get("proposal")
        if (
            event.get("type") == "proposal"
            and isinstance(proposal, dict)
            and str(proposal.get("id") or "") == proposal_id
        ):
            event["status"] = status
            return


def bounded_iterations(value: Any) -> int:
    """Return the persisted rubric cap within the supported range."""
    if isinstance(value, bool):
        return 3
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 3
    return parsed if 1 <= parsed <= 20 else 3
