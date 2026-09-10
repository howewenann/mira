"""Durable subagent run records stored beside the main transcript."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

RUNNING = "RUNNING"
DONE = "DONE"
ERROR = "ERROR"
CANCELLED = "CANCELLED"
INTERRUPTED = "INTERRUPTED"
TERMINAL_STATUSES = {DONE, ERROR, CANCELLED, INTERRUPTED}
RUN_STATUSES = {RUNNING, *TERMINAL_STATUSES}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_anchor_id() -> str:
    """Return a MIRA-owned identity persisted on an originating tool event."""
    return uuid.uuid4().hex


def normalize_runs(value: Any) -> list[dict[str, Any]]:
    """Return JSON-safe current-schema run records."""
    if not isinstance(value, list):
        return []
    runs: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        run_id = str(item.get("id") or "")
        anchor_id = str(item.get("anchor_id") or "")
        if not run_id or not anchor_id:
            continue
        status = str(item.get("status") or RUNNING).upper()
        if status not in RUN_STATUSES:
            status = ERROR
        duration = item.get("duration_ms")
        run = {
            "id": run_id,
            "turn_id": str(item.get("turn_id") or ""),
            "anchor_id": anchor_id,
            "task_call_id": str(item.get("task_call_id") or ""),
            "tool_call_id": str(item.get("tool_call_id") or ""),
            "eval_id": str(item.get("eval_id") or ""),
            "name": str(item.get("name") or "subagent"),
            "display_name": str(item.get("display_name") or item.get("name") or "subagent"),
            "task_input": str(item.get("task_input") or ""),
            "label": str(item.get("label") or ""),
            "model": str(item.get("model") or ""),
            "status": status,
            "duration_ms": (
                max(0, round(duration))
                if isinstance(duration, (int, float)) and not isinstance(duration, bool)
                else None
            ),
            "started_at": str(item.get("started_at") or now_iso()),
            "finished_at": str(item.get("finished_at") or ""),
            "output": str(item.get("output") or ""),
            "events": normalize_run_events(item.get("events")),
        }
        runs.append(run)
    return runs


def normalize_run_events(value: Any) -> list[dict[str, Any]]:
    """Normalize the compact transcript schema shared with ChatLog."""
    if not isinstance(value, list):
        return []
    events: list[dict[str, Any]] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            continue
        event_type = str(item.get("type") or "")
        if event_type not in {"assistant", "reasoning", "tool_call", "tool_result", "delegation"}:
            continue
        event = {
            "id": int(item.get("id") or index),
            "type": event_type,
            "created_at": str(item.get("created_at") or now_iso()),
        }
        if event_type in {"assistant", "reasoning"}:
            event["text"] = str(item.get("text") or "")
            if not event["text"]:
                continue
            stream_id = str(item.get("stream_id") or "")
            if stream_id:
                event["stream_id"] = stream_id
        elif event_type == "tool_call":
            event["name"] = str(item.get("name") or "tool")
            event["args"] = item.get("args", {})
            call_id = str(item.get("call_id") or "")
            if call_id:
                event["call_id"] = call_id
        elif event_type == "tool_result":
            event["name"] = str(item.get("name") or "tool")
            event["output"] = str(item.get("output") or "")
            call_id = str(item.get("call_id") or "")
            if call_id:
                event["call_id"] = call_id
            if str(item.get("status") or "") == "error":
                event["status"] = "error"
            duration = item.get("duration_ms")
            if isinstance(duration, (int, float)) and not isinstance(duration, bool):
                event["duration_ms"] = max(0, round(duration))
        else:
            calls = item.get("calls")
            if not isinstance(calls, list) or not calls:
                continue
            event["calls"] = calls
        events.append(event)
    return events


def start_run(
    session: dict[str, Any],
    *,
    anchor_id: str,
    turn_id: str,
    task_call_id: str = "",
    tool_call_id: str = "",
    eval_id: str = "",
    name: str = "subagent",
    display_name: str = "",
    task_input: str = "",
    label: str = "",
    model: str = "",
) -> dict[str, Any]:
    """Create one run, or enrich the existing task-dispatch record."""
    existing = run_for_task_call(session, task_call_id) if task_call_id else None
    if existing is not None and existing.get("status") == RUNNING:
        updates = {
            "anchor_id": anchor_id,
            "turn_id": turn_id,
            "tool_call_id": tool_call_id,
            "eval_id": eval_id,
            "name": name,
            "display_name": display_name or name,
            "task_input": task_input,
            "label": label,
            "model": model,
        }
        stable_identity = {"name", "display_name", "task_input", "label", "model"}
        for key, value in updates.items():
            if value and (key not in stable_identity or not existing.get(key)):
                existing[key] = value
        return existing

    run = {
        "id": uuid.uuid4().hex,
        "turn_id": str(turn_id),
        "anchor_id": str(anchor_id),
        "task_call_id": str(task_call_id),
        "tool_call_id": str(tool_call_id),
        "eval_id": str(eval_id),
        "name": str(name or "subagent"),
        "display_name": str(display_name or name or "subagent"),
        "task_input": str(task_input or ""),
        "label": str(label or ""),
        "model": str(model or ""),
        "status": RUNNING,
        "duration_ms": None,
        "started_at": now_iso(),
        "finished_at": "",
        "output": "",
        "events": [],
    }
    session.setdefault("runs", []).append(run)
    return run


def append_run_event(session: dict[str, Any], run_id: str, event: dict[str, Any]) -> dict[str, Any] | None:
    """Append or update one continuously captured child transcript event."""
    run = get_run(session, run_id)
    if run is None:
        return None
    events = run.setdefault("events", [])
    value = dict(event)
    event_type = str(value.get("type") or "")
    call_id = str(value.get("call_id") or "")
    stream_id = str(value.get("stream_id") or "")
    if stream_id:
        for existing in reversed(events):
            if str(existing.get("stream_id") or "") == stream_id:
                existing.update(value)
                return existing
    if event_type == "tool_call" and call_id:
        for existing in reversed(events):
            if existing.get("type") == "tool_call" and str(existing.get("call_id") or "") == call_id:
                existing.update(value)
                return existing
    value["id"] = max((int(item.get("id") or 0) for item in events if isinstance(item, dict)), default=0) + 1
    value.setdefault("created_at", now_iso())
    events.append(value)
    return value


def finish_run(
    session: dict[str, Any],
    run_id: str,
    *,
    status: str,
    output: str = "",
    duration_ms: int | None = None,
) -> dict[str, Any] | None:
    """Persist terminal execution facts for one run."""
    run = get_run(session, run_id)
    if run is None:
        return None
    terminal = str(status or DONE).upper()
    if terminal not in TERMINAL_STATUSES:
        terminal = ERROR
    run["status"] = terminal
    run["output"] = str(output or "")
    finished_at = now_iso()
    if duration_ms is None and run.get("duration_ms") is None:
        try:
            started_at = datetime.fromisoformat(str(run.get("started_at") or ""))
            finished = datetime.fromisoformat(finished_at)
            duration_ms = int(max(0.0, (finished - started_at).total_seconds()) * 1000)
        except ValueError:
            duration_ms = None
    run["duration_ms"] = max(0, int(duration_ms)) if duration_ms is not None else run.get("duration_ms")
    run["finished_at"] = finished_at
    return run


def get_run(session: dict[str, Any], run_id: str) -> dict[str, Any] | None:
    return next(
        (run for run in session.get("runs", []) if isinstance(run, dict) and str(run.get("id") or "") == run_id),
        None,
    )


def run_for_task_call(session: dict[str, Any], task_call_id: str) -> dict[str, Any] | None:
    if not task_call_id:
        return None
    return next(
        (
            run
            for run in reversed(session.get("runs", []))
            if isinstance(run, dict) and str(run.get("task_call_id") or "") == task_call_id
        ),
        None,
    )


def runs_for_anchor(session: dict[str, Any], anchor_id: str) -> list[dict[str, Any]]:
    return [
        run
        for run in session.get("runs", [])
        if isinstance(run, dict) and str(run.get("anchor_id") or "") == str(anchor_id)
    ]


def run_events(session: dict[str, Any], run_id: str) -> list[dict[str, Any]]:
    run = get_run(session, run_id)
    return list(run.get("events", [])) if run is not None else []


def reconcile_stale_runs(session: dict[str, Any]) -> bool:
    """Mark runs loaded without an active parent execution as interrupted."""
    changed = False
    for run in session.get("runs", []):
        if isinstance(run, dict) and run.get("status") == RUNNING:
            run["status"] = INTERRUPTED
            run["finished_at"] = run.get("finished_at") or now_iso()
            changed = True
    for event in session.get("events", []):
        if (
            isinstance(event, dict)
            and event.get("type") == "subagent"
            and event.get("status") == RUNNING
        ):
            event["status"] = INTERRUPTED
            changed = True
    return changed
