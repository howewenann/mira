"""Observational projection of nested Rubric execution into live inspections."""

from __future__ import annotations

from collections import Counter
from typing import Any

from core.context.usage import field as value_field
from core.execution.inspection.live import InspectionEvent, LiveInspectionStore
from core.execution.inspection.subagents import live_inspection_store
from core.execution.streams.messages import streamed_message_deltas
from core.execution.streams.output import is_tool_message, message_text
from core.execution.streams.tool_args import normalized_call

RUBRIC_PHASES = {"verifier", "grader"}


def rubric_inspection_id(run_id: str, iteration: int, phase: str) -> str:
    """Return the deterministic identity for one Rubric phase transcript."""
    return f"rubric:{run_id}:{max(0, int(iteration))}:{phase}"


def rubric_inspection_title(iteration: int, phase: str) -> str:
    """Return the concise title shown by the generic Inspector."""
    return f"{phase.title()} · Pass {max(0, int(iteration)) + 1}"


def inspection_event_values(messages: Any) -> list[dict[str, Any]]:
    """Normalize the exact nested-agent input into the Inspector vocabulary."""
    values: list[dict[str, Any]] = []
    for message in messages if isinstance(messages, (list, tuple)) else ():
        if is_tool_message(message):
            status = str(value_field(message, "status") or "")
            values.append(
                {
                    "kind": "tool_error" if status == "error" else "tool_result",
                    "text": message_text(message),
                    "name": str(value_field(message, "name") or "tool"),
                    "call_id": str(
                        value_field(message, "tool_call_id")
                        or value_field(message, "id")
                        or ""
                    ),
                }
            )
            continue

        message_type = str(value_field(message, "type") or "")
        if message_type in {"human", "user", "system"}:
            values.append({"kind": "user", "text": message_text(message)})
            continue
        if message_type not in {"ai", "assistant"}:
            continue

        reasoning, text = streamed_message_deltas(message)
        if reasoning:
            values.append({"kind": "reasoning", "text": reasoning})
        if text:
            values.append({"kind": "assistant", "text": text})
        for raw_call in value_field(message, "tool_calls") or []:
            call = normalized_call(raw_call)
            values.append(
                {
                    "kind": "tool_call",
                    "name": str(call.get("name") or "tool"),
                    "args": call.get("args", {}),
                    "call_id": str(call.get("id") or ""),
                }
            )
    return values


class RubricInspectionProjector:
    """Keep Rubric observation small, process-local, and execution-neutral."""

    def __init__(self, renderer: Any) -> None:
        self.store: LiveInspectionStore | None = live_inspection_store(renderer)

    def start(self, event: dict[str, Any], phase: str) -> str:
        if self.store is None or phase not in RUBRIC_PHASES:
            return ""
        inspection_id = self._event_id(event, phase)
        title = rubric_inspection_title(int(event.get("iteration") or 0), phase)
        raw_events = event.get("inspection_events")
        input_events = [
            self._inspection_event(value)
            for value in raw_events
            if isinstance(value, dict)
        ] if isinstance(raw_events, list) else []

        current = self.store.get(inspection_id)
        if current is not None and current.status == "RUNNING":
            self.store.start(
                inspection_id,
                title,
                inspection_type=phase,
            )
            return inspection_id

        first_user = input_events[0] if input_events and input_events[0].kind == "user" else None
        if current is None:
            self.store.start(
                inspection_id,
                title,
                first_user.text if first_user is not None else "",
                inspection_type=phase,
            )
            remaining = input_events[1:] if first_user is not None else input_events
        else:
            current.status = "RUNNING"
            self.store.start(inspection_id, title, inspection_type=phase)
            remaining = input_events
        for item in remaining:
            self.store.append(inspection_id, item)
        return inspection_id

    def append_delta(self, event: dict[str, Any]) -> None:
        if self.store is None:
            return
        phase = str(event.get("phase") or "")
        kind = str(event.get("kind") or "")
        if phase not in RUBRIC_PHASES or kind not in {"reasoning", "assistant"}:
            return
        self.store.append_delta(
            self._event_id(event, phase),
            kind,
            str(event.get("text") or ""),
        )

    def tool_call(self, event: dict[str, Any]) -> None:
        if self.store is None:
            return
        inspection_id = self._event_id(event, "verifier")
        call_id = str(event.get("tool_call_id") or "")
        name = str(event.get("tool_name") or "tool")
        if call_id and not call_id.startswith("index:"):
            inspection = self.store.get(inspection_id)
            completed = {
                item.call_id
                for item in inspection.events
                if item.kind in {"tool_result", "tool_error"}
            } if inspection is not None else set()
            for item in reversed(inspection.events if inspection is not None else []):
                if (
                    item.kind == "tool_call"
                    and item.name == name
                    and item.call_id.startswith("index:")
                    and item.call_id not in completed
                ):
                    item.call_id = call_id
                    break
        self.store.upsert_tool_call(
            inspection_id,
            InspectionEvent(
                "tool_call",
                name=name,
                args=event.get("tool_args", {}),
                call_id=call_id,
            ),
        )

    def tool_completion(self, event: dict[str, Any]) -> None:
        if self.store is None:
            return
        self.store.upsert_tool_completion(
            self._event_id(event, "verifier"),
            InspectionEvent(
                "tool_error" if event.get("is_error") else "tool_result",
                text=str(event.get("output") or ""),
                name=str(event.get("tool_name") or "tool"),
                call_id=str(event.get("tool_call_id") or ""),
            ),
        )

    def finish(self, event: dict[str, Any], phase: str) -> None:
        if self.store is None or phase not in RUBRIC_PHASES:
            return
        inspection_id = self._event_id(event, phase)
        succeeded = bool(event.get("succeeded"))
        cancelled = bool(event.get("cancelled"))
        error = str(event.get("error") or "")
        if not succeeded:
            self._close_pending_tools(
                inspection_id,
                error or ("cancelled" if cancelled else f"{phase} failed"),
            )
        self.store.finish(
            inspection_id,
            status="CANCELLED" if cancelled else ("DONE" if succeeded else "ERROR"),
            final_response=(
                str(event.get("final_response") or "")
                if phase == "verifier" and event.get("final_response") is not None
                else None
            ),
            error=error,
        )

    @staticmethod
    def _inspection_event(value: dict[str, Any]) -> InspectionEvent:
        return InspectionEvent(
            str(value.get("kind") or ""),
            text=str(value.get("text") or ""),
            name=str(value.get("name") or ""),
            args=value.get("args"),
            call_id=str(value.get("call_id") or ""),
        )

    @staticmethod
    def _event_id(event: dict[str, Any], phase: str) -> str:
        expected = rubric_inspection_id(
            str(event.get("grading_run_id") or ""),
            int(event.get("iteration") or 0),
            phase,
        )
        supplied = str(event.get("inspection_id") or "")
        return supplied if supplied == expected else expected

    def _close_pending_tools(self, inspection_id: str, text: str) -> None:
        if self.store is None:
            return
        inspection = self.store.get(inspection_id)
        if inspection is None:
            return
        completed = Counter(
            (event.call_id, event.name)
            for event in inspection.events
            if event.kind in {"tool_result", "tool_error"}
        )
        for event in list(inspection.events):
            if event.kind != "tool_call":
                continue
            key = (event.call_id, event.name)
            if completed[key]:
                completed[key] -= 1
                continue
            self.store.upsert_tool_completion(
                inspection_id,
                InspectionEvent(
                    "tool_error",
                    text=text,
                    name=event.name,
                    call_id=event.call_id,
                ),
            )


__all__ = [
    "RubricInspectionProjector",
    "inspection_event_values",
    "rubric_inspection_id",
    "rubric_inspection_title",
]
