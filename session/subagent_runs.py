"""Durable, model-isolated subagent inspection records."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from typing import Any


RUN_STATUSES = {"RUNNING", "DONE", "ERROR", "CANCELLED"}
EVENT_KINDS = {
    "user",
    "reasoning",
    "assistant",
    "tool_call",
    "tool_result",
    "tool_error",
    "error",
}


def normalize_runs(value: Any) -> list[dict[str, Any]]:
    """Return valid persisted inspection runs without touching session events."""
    if not isinstance(value, list):
        return []
    runs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        run_id = str(item.get("id") or "")
        if not run_id or run_id in seen:
            continue
        seen.add(run_id)
        origin_event_id = item.get("origin_event_id")
        if isinstance(origin_event_id, bool):
            origin_event_id = None
        elif origin_event_id is not None:
            try:
                origin_event_id = int(origin_event_id)
            except (TypeError, ValueError):
                origin_event_id = None
            if origin_event_id is not None and origin_event_id <= 0:
                origin_event_id = None
        status = str(item.get("status") or "RUNNING").upper()
        if status not in RUN_STATUSES:
            status = "ERROR"
        duration = item.get("duration_ms")
        duration_ms = (
            max(0, round(duration))
            if isinstance(duration, (int, float)) and not isinstance(duration, bool)
            else None
        )
        run = {
            "id": run_id,
            "inspection_id": str(item.get("inspection_id") or ""),
            "origin_event_id": origin_event_id,
            "origin_tool": str(item.get("origin_tool") or ""),
            "origin_call_id": str(item.get("origin_call_id") or ""),
            "row_id": str(item.get("row_id") or ""),
            "eval_id": str(item.get("eval_id") or ""),
            "inspection_type": str(item.get("inspection_type") or "subagent"),
            "display_name": str(item.get("display_name") or "subagent"),
            "task": str(item.get("task") or ""),
            "status": status,
            "started_at": str(item.get("started_at") or _now_iso()),
            "updated_at": str(item.get("updated_at") or item.get("started_at") or _now_iso()),
            "duration_ms": duration_ms,
            "output": str(item.get("output") or ""),
            "events": normalize_run_events(item.get("events")),
        }
        finished_at = str(item.get("finished_at") or "")
        if finished_at:
            run["finished_at"] = finished_at
        runs.append(run)
    return runs


def normalize_run_events(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    events: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "")
        if kind not in EVENT_KINDS:
            continue
        event: dict[str, Any] = {"kind": kind}
        text = str(item.get("text") or "")
        name = str(item.get("name") or "")
        call_id = str(item.get("call_id") or "")
        if text or kind in {"user", "reasoning", "assistant", "error"}:
            event["text"] = text
        if name:
            event["name"] = name
        if kind == "tool_call":
            event["args"] = item.get("args")
        if call_id:
            event["call_id"] = call_id
        events.append(event)
    return events


def runs_for_origin(value: Any, origin_event_id: int) -> list[dict[str, Any]]:
    return [run for run in normalize_runs(value) if run.get("origin_event_id") == origin_event_id]


def run_for_id(value: Any, run_id: str) -> dict[str, Any] | None:
    return next((run for run in normalize_runs(value) if run["id"] == run_id), None)


def run_for_inspection_id(value: Any, inspection_id: str) -> dict[str, Any] | None:
    return next(
        (
            run
            for run in normalize_runs(value)
            if run["inspection_id"] == inspection_id
        ),
        None,
    )


def run_count(value: Any, origin_event_id: int | None) -> int:
    if not origin_event_id:
        return 0
    return sum(1 for run in normalize_runs(value) if run.get("origin_event_id") == origin_event_id)


def upsert_run(record: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Insert or update one logical row while preserving its durable identity."""
    normalized = normalize_runs([candidate])
    if not normalized:
        raise ValueError("subagent run candidate is invalid")
    incoming = normalized[0]
    runs = normalize_runs(record.get("runs"))
    current = next((run for run in runs if run["id"] == incoming["id"]), None)
    same_inspection = current is not None
    if current is None and incoming["inspection_id"]:
        current = next(
            (run for run in runs if run["inspection_id"] == incoming["inspection_id"]),
            None,
        )
        same_inspection = current is not None
    if current is None and incoming.get("origin_event_id") and incoming["row_id"]:
        current = next(
            (
                run for run in runs
                if run.get("origin_event_id") == incoming["origin_event_id"]
                and run["row_id"] == incoming["row_id"]
            ),
            None,
        )
    if current is None:
        runs.append(incoming)
        current = incoming
    else:
        durable_id = current["id"]
        started_at = current["started_at"]
        inspection_id = current["inspection_id"]
        display_name = current["display_name"]
        finished_at = current.get("finished_at") if current["status"] != "RUNNING" else None
        current.update(incoming)
        current["id"] = durable_id
        current["started_at"] = started_at
        if not same_inspection:
            current["inspection_id"] = inspection_id
            current["display_name"] = display_name
        if current["status"] == "RUNNING":
            current.pop("finished_at", None)
        elif finished_at:
            current["finished_at"] = finished_at
    record["runs"] = runs
    return current


def clear_runs(record: dict[str, Any]) -> None:
    """Remove every retrospective run when its owning transcript is cleared."""
    record["runs"] = []


def reconcile_stale_runs(record: dict[str, Any]) -> bool:
    """Freeze runs left active by a process that no longer owns the session."""
    runs = normalize_runs(record.get("runs"))
    changed = False
    interrupted_origins: set[int] = set()
    now = _now_iso()
    for run in runs:
        if run["status"] != "RUNNING":
            continue
        changed = True
        run["status"] = "CANCELLED"
        run["finished_at"] = run.get("updated_at") or now
        run["duration_ms"] = _duration_ms(run.get("started_at"), run.get("finished_at"))
        run["updated_at"] = now
        _close_pending_tools(run["events"])
        origin = run.get("origin_event_id")
        if isinstance(origin, int):
            interrupted_origins.add(origin)

    events = record.get("events")
    if isinstance(events, list):
        unterminated = _unterminated_tool_event_ids(events)
        for item in events:
            if not isinstance(item, dict) or item.get("type") != "tool_call":
                continue
            event_id = int(item.get("id") or 0)
            if event_id not in interrupted_origins or event_id not in unterminated or item.get("status"):
                continue
            item["status"] = "interrupted"
            matching = [run for run in runs if run.get("origin_event_id") == event_id]
            item["duration_ms"] = max((int(run.get("duration_ms") or 0) for run in matching), default=0)
            changed = True
    record["runs"] = runs
    return changed


def _unterminated_tool_event_ids(events: list[Any]) -> set[int]:
    by_id: dict[str, deque[int]] = defaultdict(deque)
    by_name: dict[str, deque[int]] = defaultdict(deque)
    for item in events:
        if not isinstance(item, dict):
            continue
        event_type = item.get("type")
        call_id = str(item.get("call_id") or "")
        name = str(item.get("name") or "tool")
        if event_type == "tool_call":
            event_id = int(item.get("id") or 0)
            if event_id:
                (by_id[call_id] if call_id else by_name[name]).append(event_id)
        elif event_type == "tool_result":
            queue = by_id.get(call_id) if call_id else by_name.get(name)
            if queue:
                queue.popleft()
    return {
        event_id
        for queues in (by_id, by_name)
        for queue in queues.values()
        for event_id in queue
    }


def _close_pending_tools(events: list[dict[str, Any]]) -> None:
    completions = Counter(
        (str(event.get("call_id") or ""), str(event.get("name") or "tool"))
        for event in events
        if event.get("kind") in {"tool_result", "tool_error"}
    )
    pending: list[dict[str, Any]] = []
    for event in events:
        if event.get("kind") != "tool_call":
            continue
        key = (str(event.get("call_id") or ""), str(event.get("name") or "tool"))
        if completions[key]:
            completions[key] -= 1
        else:
            pending.append(event)
    for event in pending:
        completion = {
            "kind": "tool_error",
            "text": "interrupted by session restart",
            "name": str(event.get("name") or "tool"),
        }
        if event.get("call_id"):
            completion["call_id"] = str(event["call_id"])
        events.append(completion)


def _duration_ms(started: Any, finished: Any) -> int:
    try:
        return max(0, round((_parse_time(finished) - _parse_time(started)).total_seconds() * 1000))
    except (TypeError, ValueError):
        return 0


def _parse_time(value: Any) -> datetime:
    text = str(value or "")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "normalize_run_events",
    "normalize_runs",
    "reconcile_stale_runs",
    "clear_runs",
    "run_count",
    "run_for_id",
    "run_for_inspection_id",
    "runs_for_origin",
    "upsert_run",
]
