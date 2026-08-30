"""Headless preparation and resolution of formal Goal/Plan reviews."""

from __future__ import annotations

from typing import Any, Literal

from session.context import append_event
from session.goals import goal_artifact, replace_current_goal, start_goal_attempt
from session.plans import plan_artifact, replace_current_plan, start_plan_attempt

ArtifactKind = Literal["goal", "plan"]


def prepare_artifact_review(
    kind: ArtifactKind,
    interrupt: Any,
    record: dict[str, Any],
    mode: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create and persist the exact provisional artifact behind a native interrupt."""
    raw_value = getattr(interrupt, "value", interrupt)
    raw = raw_value if isinstance(raw_value, dict) else {}
    artifact_id = next_artifact_id(kind, record, mode)
    common = {
        "rubric_enabled": bool(mode.get("rubric_enabled")),
        "rubric_iterations": int(mode.get("rubric_max_iterations") or 3),
    }
    if kind == "goal":
        objective = str(raw.get("objective") or "")
        criteria = str(raw.get("success_criteria") or "")
        if not objective or not criteria:
            raise RuntimeError("finalize_goal requires staged Objective and Success Criteria")
        artifact = goal_artifact(
            goal_id=artifact_id,
            title=compact_text(raw.get("title")) or "Goal",
            objective=objective,
            success_criteria=criteria,
            **common,
        )
    else:
        objective = str(raw.get("objective") or "")
        criteria = str(raw.get("success_criteria") or "")
        if not objective or not criteria:
            raise RuntimeError("finalize_plan requires staged Plan context and Success Criteria")
        artifact = plan_artifact(
            plan_id=artifact_id,
            title=compact_text(raw.get("title")) or "Plan",
            objective=objective,
            context_and_constraints=str(raw.get("context_and_constraints") or ""),
            key_changes=compact_items(raw.get("key_changes"), "List the key implementation changes."),
            test_plan=compact_items(raw.get("test_plan"), "Describe the tests or checks to create."),
            assumptions=compact_items(raw.get("assumptions"), "No additional assumptions."),
            success_criteria=criteria,
            **common,
        )
    event = append_event(
        record,
        {
            "type": kind,
            "mode": "planning" if mode.get("planning") else "action",
            kind: artifact,
            "status": "proposed",
        },
    )
    return artifact, event


def resolve_artifact_review(
    kind: ArtifactKind,
    artifact: dict[str, Any],
    decision: dict[str, Any],
    record: dict[str, Any],
    mode: dict[str, Any],
) -> dict[str, Any]:
    """Apply one frontend review decision to authoritative MIRA state."""
    action = str(decision.get("action") or "")
    if action not in {"implement", "close", "revise", "clear"}:
        raise RuntimeError(f"unknown {kind} review action")
    artifact_id = str(artifact.get("id") or "")
    previous_plan = record.get("current_plan") if isinstance(record.get("current_plan"), dict) else None
    previous_goal = record.get("current_goal") if isinstance(record.get("current_goal"), dict) else None
    detail = {
        "previous_plan_id": str((previous_plan or {}).get("id") or ""),
        "previous_goal_id": str((previous_goal or {}).get("id") or ""),
    }

    if action in {"revise", "clear"}:
        update_artifact_event_status(record, kind, artifact_id, "rejected" if action == "revise" else "cleared")
        return detail

    if kind == "plan":
        accepted = replace_current_plan(record, artifact)
        mode["current_plan"] = accepted
        mode["current_goal"] = None
    else:
        accepted = replace_current_goal(record, artifact)
        mode["current_goal"] = accepted
        mode["current_plan"] = None

    if action == "implement":
        accepted = start_plan_attempt(record) if kind == "plan" else start_goal_attempt(record)
        mode[f"executing_{kind}"] = True
        mode[f"executing_{'goal' if kind == 'plan' else 'plan'}"] = False
        mode["planning"] = False
        update_artifact_event_status(record, kind, artifact_id, "active")
    else:
        update_artifact_event_status(record, kind, artifact_id, "closed")
    mode["planning_stage"] = "plan_research"
    detail["accepted"] = accepted
    return detail


def update_artifact_event_status(
    record: dict[str, Any],
    kind: ArtifactKind,
    artifact_id: str,
    status: str,
) -> None:
    for event in record.get("events", ()):
        if not isinstance(event, dict):
            continue
        value = event.get(kind)
        if event.get("type") == kind and isinstance(value, dict) and str(value.get("id") or "") == artifact_id:
            event["status"] = status
            return


def next_artifact_id(kind: ArtifactKind, record: dict[str, Any], mode: dict[str, Any]) -> str:
    existing = {
        str(event.get(kind, {}).get("id") or "")
        for event in record.get("events", ())
        if isinstance(event, dict) and isinstance(event.get(kind), dict)
    }
    counter = int(mode.get("plan_counter") or 0)
    while True:
        counter += 1
        artifact_id = f"{kind}-{counter}"
        if artifact_id not in existing:
            mode["plan_counter"] = counter
            return artifact_id


def compact_items(value: Any, fallback: str) -> list[str]:
    values = [value] if isinstance(value, str) else value if isinstance(value, (list, tuple)) else ()
    items = [text for item in values if (text := compact_text(item))]
    return items or [fallback]


def compact_text(value: Any) -> str:
    return " ".join(str(value or "").split())


__all__ = [
    "ArtifactKind",
    "prepare_artifact_review",
    "resolve_artifact_review",
    "update_artifact_event_status",
]
