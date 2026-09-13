"""Observational projection of live inspections into durable session runs."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Any

from core.diagnostics.logging import get_diagnostics_logger
from core.execution.inspection.live import InspectionEvent, InspectionUpdate, LiveInspection, LiveInspectionStore
from core.execution.inspection.subagents import live_inspection_store
from session.subagent_runs import normalize_runs, upsert_run


TEXT_FLUSH_SECONDS = 0.25


class PersistentSubagentRuns:
    """Transparent renderer wrapper that persists existing Phase 1 truth."""

    def __init__(self, renderer: Any, record: dict[str, Any], store: Any) -> None:
        self.renderer = renderer
        self.record = record
        self.session_store = store
        self.inspections = live_inspection_store(renderer)
        self._metadata: dict[str, dict[str, Any]] = {}
        self._subscriptions: set[str] = set()
        self._dirty = False
        self._flush_handle: asyncio.TimerHandle | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.renderer, name)

    def tool_call(self, name: str, args: Any, call_id: str = "", **kwargs: Any) -> Any:
        result = self.renderer.tool_call(name, args, call_id=call_id, **kwargs)
        if name in {"task", "eval"}:
            self._origin_available(name, call_id, args)
        return result

    def recovered_tool_call(self, name: str, args: Any, call_id: str = "", **kwargs: Any) -> Any:
        callback = getattr(self.renderer, "recovered_tool_call", None)
        result = callback(name, args, call_id=call_id, **kwargs) if callable(callback) else self.renderer.tool_call(
            name, args, call_id=call_id, **kwargs
        )
        if name in {"task", "eval"}:
            self._origin_available(name, call_id, args)
        return result

    def subagent_started(
        self,
        subagent: str,
        task_input: str = "",
        *,
        origin: str = "",
        eval_id: str = "",
        row_id: str = "",
        model: str = "",
        inspection_id: str = "",
        **kwargs: Any,
    ) -> Any:
        result = self.renderer.subagent_started(
            subagent,
            task_input,
            origin=origin,
            eval_id=eval_id,
            row_id=row_id,
            model=model,
            inspection_id=inspection_id,
            **kwargs,
        )
        self._attach(
            inspection_id,
            origin_tool="task",
            origin_call_id=row_id,
            row_id=row_id,
            eval_id="",
            task=task_input,
            display_name=subagent,
        )
        return result

    def subagent_finished(self, subagent: str, result: str = "", **kwargs: Any) -> Any:
        outcome = self.renderer.subagent_finished(subagent, result, **kwargs)
        self._terminal(kwargs.get("inspection_id", ""), kwargs.get("row_id", ""), result, kwargs.get("duration_ms"))
        return outcome

    def subagent_cancelled(self, subagent: str, result: str = "", **kwargs: Any) -> Any:
        callback = getattr(self.renderer, "subagent_cancelled", None)
        outcome = callback(subagent, result, **kwargs) if callable(callback) else None
        self._terminal("", kwargs.get("row_id", ""), result, kwargs.get("duration_ms"), status="CANCELLED")
        return outcome

    def eval_subagent_started(
        self,
        subagent: str,
        task_input: str = "",
        *,
        eval_id: str = "",
        row_id: str = "",
        model: str = "",
        label: str = "",
        inspection_id: str = "",
        **kwargs: Any,
    ) -> Any:
        callback = getattr(self.renderer, "eval_subagent_started", None)
        outcome = callback(
            subagent,
            task_input,
            eval_id=eval_id,
            row_id=row_id,
            model=model,
            label=label,
            inspection_id=inspection_id,
            **kwargs,
        ) if callable(callback) else None
        self._attach(
            inspection_id,
            origin_tool="eval",
            origin_call_id=eval_id,
            row_id=row_id,
            eval_id=eval_id,
            task=task_input,
            display_name=subagent,
        )
        return outcome

    def eval_subagent_finished(self, subagent: str, result: str = "", **kwargs: Any) -> Any:
        callback = getattr(self.renderer, "eval_subagent_finished", None)
        outcome = callback(subagent, result, **kwargs) if callable(callback) else None
        self._terminal("", kwargs.get("row_id", ""), result, kwargs.get("duration_ms"))
        return outcome

    def eval_subagent_cancelled(self, subagent: str, result: str = "", **kwargs: Any) -> Any:
        callback = getattr(self.renderer, "eval_subagent_cancelled", None)
        outcome = callback(subagent, result, **kwargs) if callable(callback) else None
        status = "ERROR" if result else "CANCELLED"
        self._terminal("", kwargs.get("row_id", ""), result, kwargs.get("duration_ms"), status=status)
        return outcome

    def flush(self) -> None:
        if self._flush_handle is not None:
            self._flush_handle.cancel()
            self._flush_handle = None
        if not self._dirty:
            return
        try:
            self.session_store.save(self.record)
        except Exception as exc:  # persistence must never fail execution
            get_diagnostics_logger().warning("subagent run persistence failed: %s", exc)
            return
        self._dirty = False

    def close(self) -> None:
        self.flush()
        if self.inspections is not None:
            for inspection_id in self._subscriptions:
                self.inspections.unsubscribe(inspection_id, self._inspection_updated)
        self._subscriptions.clear()

    def _attach(self, inspection_id: str, **metadata: Any) -> None:
        if not inspection_id or self.inspections is None:
            return
        self._metadata[inspection_id] = {**self._metadata.get(inspection_id, {}), **metadata}
        if inspection_id not in self._subscriptions:
            self.inspections.subscribe(inspection_id, self._inspection_updated)
            self._subscriptions.add(inspection_id)
        self._snapshot(inspection_id, immediate=True)

    def _terminal(
        self,
        inspection_id: str,
        row_id: str,
        output: str,
        duration_ms: Any,
        *,
        status: str = "",
    ) -> None:
        target = inspection_id or next(
            (key for key, value in self._metadata.items() if value.get("row_id") == row_id),
            "",
        )
        if not target:
            return
        metadata = self._metadata.setdefault(target, {})
        if output:
            metadata["output"] = str(output)
        if isinstance(duration_ms, (int, float)) and not isinstance(duration_ms, bool):
            metadata["duration_ms"] = max(0, round(duration_ms))
        if status:
            metadata["status"] = status
        self._snapshot(target, immediate=True)

    def _inspection_updated(self, inspection_id: str, update: InspectionUpdate) -> None:
        structural = update.operation in {"reset", "status"} or (
            update.event is not None
            and update.event.kind not in {"reasoning", "assistant"}
        )
        self._snapshot(inspection_id, immediate=structural)

    def _snapshot(self, inspection_id: str, *, immediate: bool) -> None:
        try:
            inspection = self.inspections.get(inspection_id) if self.inspections is not None else None
            metadata = self._metadata.get(inspection_id)
            if inspection is None or metadata is None:
                return
            origin_event_id = self._origin_event_id(
                str(metadata.get("origin_tool") or ""),
                str(metadata.get("origin_call_id") or ""),
                str(metadata.get("task") or ""),
                str(metadata.get("row_id") or ""),
                inspection_id,
            )
            now = _now_iso()
            events = [_event_dict(event) for event in inspection.events]
            status = str(metadata.get("status") or inspection.status or "RUNNING")
            output = str(metadata.get("output") or _final_output(events))
            if status != "RUNNING" and origin_event_id is None:
                # A child stream may finish before its owning top-level call is
                # delivered. Keep the last RUNNING snapshot durable and let the
                # later call backfill ownership and the terminal state together.
                return
            candidate = {
                "id": str(uuid.uuid4()),
                "inspection_id": inspection_id,
                "origin_event_id": origin_event_id,
                "origin_tool": str(metadata.get("origin_tool") or ""),
                "origin_call_id": str(metadata.get("origin_call_id") or ""),
                "row_id": str(metadata.get("row_id") or ""),
                "eval_id": str(metadata.get("eval_id") or ""),
                "inspection_type": inspection.inspection_type,
                "display_name": inspection.title,
                "task": _task_text(events) or str(metadata.get("task") or ""),
                "status": status,
                "started_at": now,
                "updated_at": now,
                "duration_ms": metadata.get("duration_ms"),
                "output": output,
                "events": events,
            }
            if status != "RUNNING":
                candidate["finished_at"] = now
            current = upsert_run(self.record, candidate)
            if status != "RUNNING" and current.get("duration_ms") is None:
                current["duration_ms"] = _elapsed_ms(current["started_at"], current["finished_at"])
            self._dirty = True
            if immediate:
                self.flush()
            else:
                self._schedule_flush()
        except Exception as exc:  # observational projection must remain isolated
            get_diagnostics_logger().warning("subagent run projection failed: %s", exc)

    def _origin_available(self, name: str, call_id: str, args: Any) -> None:
        task = str(args.get("description") or "") if isinstance(args, dict) else ""
        for inspection_id, metadata in self._metadata.items():
            if metadata.get("origin_tool") != name:
                continue
            if call_id and metadata.get("origin_call_id") != call_id:
                continue
            if not call_id and task and metadata.get("task") != task:
                continue
            self._snapshot(inspection_id, immediate=True)

    def _origin_event_id(
        self,
        tool: str,
        call_id: str,
        task: str,
        row_id: str,
        inspection_id: str,
    ) -> int | None:
        events = self.record.get("events")
        if not isinstance(events, list):
            return None
        existing = next(
            (
                run.get("origin_event_id")
                for run in normalize_runs(self.record.get("runs"))
                if run.get("origin_tool") == tool
                and (
                    run.get("inspection_id") == inspection_id
                    or (row_id and run.get("row_id") == row_id)
                )
            ),
            None,
        )
        if existing:
            return int(existing)
        candidates: list[int] = []
        for event in reversed(events):
            if not isinstance(event, dict) or event.get("type") != "tool_call" or event.get("name") != tool:
                continue
            if call_id and str(event.get("call_id") or "") == call_id:
                return int(event.get("id") or 0) or None
            args = event.get("args")
            if (
                not call_id
                and not event.get("call_id")
                and (not task or (isinstance(args, dict) and str(args.get("description") or "") == task))
            ):
                event_id = int(event.get("id") or 0)
                if event_id:
                    candidates.append(event_id)
        if not candidates:
            return None
        claimed = {
            int(run["origin_event_id"])
            for run in normalize_runs(self.record.get("runs"))
            if run.get("origin_tool") == tool
            and run.get("origin_event_id")
            and run.get("inspection_id") != inspection_id
            and (not row_id or run.get("row_id") != row_id)
        }
        return next((event_id for event_id in reversed(candidates) if event_id not in claimed), None)

    def _schedule_flush(self) -> None:
        if self._flush_handle is not None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self.flush()
            return
        self._flush_handle = loop.call_later(TEXT_FLUSH_SECONDS, self._scheduled_flush)

    def _scheduled_flush(self) -> None:
        self._flush_handle = None
        self.flush()


def inspection_from_run(run: dict[str, Any]) -> LiveInspection:
    return LiveInspection(
        id=str(run.get("id") or ""),
        title=str(run.get("display_name") or "subagent"),
        inspection_type=str(run.get("inspection_type") or "subagent"),
        status=str(run.get("status") or "CANCELLED"),
        events=[
            InspectionEvent(
                str(event.get("kind") or ""),
                text=str(event.get("text") or ""),
                name=str(event.get("name") or ""),
                args=event.get("args"),
                call_id=str(event.get("call_id") or ""),
            )
            for event in run.get("events", [])
            if isinstance(event, dict)
        ],
    )


def _event_dict(event: InspectionEvent) -> dict[str, Any]:
    value: dict[str, Any] = {"kind": event.kind}
    if event.text or event.kind in {"user", "reasoning", "assistant", "error"}:
        value["text"] = event.text
    if event.name:
        value["name"] = event.name
    if event.kind == "tool_call":
        value["args"] = _json_value(event.args)
    if event.call_id:
        value["call_id"] = event.call_id
    return value


def _json_value(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _task_text(events: list[dict[str, Any]]) -> str:
    return next((str(event.get("text") or "") for event in events if event.get("kind") == "user"), "")


def _final_output(events: list[dict[str, Any]]) -> str:
    return next((str(event.get("text") or "") for event in reversed(events) if event.get("kind") == "assistant"), "")


def _elapsed_ms(started: str, finished: str) -> int:
    try:
        return max(0, round((_parse_time(finished) - _parse_time(started)).total_seconds() * 1000))
    except (TypeError, ValueError):
        return 0


def _parse_time(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = ["PersistentSubagentRuns", "inspection_from_run"]
