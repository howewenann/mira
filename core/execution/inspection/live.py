"""Small in-memory store for live inspection transcripts."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import count
from typing import Any


@dataclass
class InspectionEvent:
    """One renderable item in a live inspection transcript."""

    kind: str
    text: str = ""
    name: str = ""
    args: Any = None
    call_id: str = ""


@dataclass
class LiveInspection:
    """Process-local transcript for one running or completed child."""

    id: str
    title: str
    events: list[InspectionEvent] = field(default_factory=list)
    status: str = "RUNNING"


@dataclass(frozen=True)
class InspectionUpdate:
    """Incremental notification delivered to an open Inspector."""

    operation: str
    event: InspectionEvent | None = None


InspectionListener = Callable[[str, InspectionUpdate], None]


class LiveInspectionStore:
    """Own live inspection state without persistence or session coupling."""

    def __init__(self) -> None:
        self._items: dict[str, LiveInspection] = {}
        self._listeners: dict[str, set[InspectionListener]] = defaultdict(set)
        self._fallback_ids = count(1)

    def allocate_id(self, hint: str = "") -> str:
        """Return a stable unused ID derived from native execution identity."""
        base = f"subagent:{hint}" if hint else f"subagent:live-{next(self._fallback_ids)}"
        candidate = base
        suffix = 2
        while candidate in self._items:
            candidate = f"{base}:{suffix}"
            suffix += 1
        return candidate

    def start(self, inspection_id: str, title: str, task: str = "") -> LiveInspection:
        """Create a transcript whose first item is always the child request."""
        current = self._items.get(inspection_id)
        if current is None:
            current = LiveInspection(
                id=inspection_id,
                title=title,
                events=[InspectionEvent("user", text=str(task or ""))],
            )
            self._items[inspection_id] = current
            self._notify(inspection_id, InspectionUpdate("reset"))
            return current
        changed = False
        if title and current.title != title:
            current.title = title
            changed = True
        if task and current.events and current.events[0].kind == "user":
            if current.events[0].text != task:
                current.events[0].text = str(task)
                changed = True
        if changed:
            self._notify(inspection_id, InspectionUpdate("reset"))
        return current

    def update_task(self, inspection_id: str, task: str) -> None:
        """Fill a request that arrived after the native child handle."""
        current = self._items.get(inspection_id)
        if current is None or not task:
            return
        if not current.events:
            current.events.append(InspectionEvent("user", text=str(task)))
        elif current.events[0].kind == "user":
            current.events[0].text = str(task)
        else:
            current.events.insert(0, InspectionEvent("user", text=str(task)))
        self._notify(inspection_id, InspectionUpdate("reset"))

    def append(self, inspection_id: str, event: InspectionEvent) -> None:
        current = self._items.get(inspection_id)
        if current is None:
            return
        current.events.append(event)
        self._notify(inspection_id, InspectionUpdate("append", event))

    def upsert_tool_call(self, inspection_id: str, event: InspectionEvent) -> None:
        """Promote a streamed tool draft without creating a duplicate bubble."""
        current = self._items.get(inspection_id)
        if current is None:
            return
        if event.call_id:
            for existing in reversed(current.events):
                if existing.kind == "tool_call" and existing.call_id == event.call_id:
                    existing.name = event.name
                    existing.args = event.args
                    self._notify(inspection_id, InspectionUpdate("reset"))
                    return
        self.append(inspection_id, event)

    def append_delta(self, inspection_id: str, kind: str, delta: str) -> None:
        """Append streamed text while retaining logical transcript bubbles."""
        current = self._items.get(inspection_id)
        text = str(delta or "")
        if current is None or not text:
            return
        if current.events and current.events[-1].kind == kind:
            current.events[-1].text += text
            self._notify(
                inspection_id,
                InspectionUpdate("delta", InspectionEvent(kind, text=text)),
            )
            return
        event = InspectionEvent(kind, text=text)
        current.events.append(event)
        self._notify(inspection_id, InspectionUpdate("append", event))

    def finish(
        self,
        inspection_id: str,
        *,
        status: str,
        final_response: str | None = None,
        error: str = "",
    ) -> None:
        """Mark terminal and guarantee the parent-facing response is last."""
        current = self._items.get(inspection_id)
        if current is None:
            return
        if error:
            self.append(inspection_id, InspectionEvent("error", text=error))
        if final_response is not None:
            response = str(final_response)
            if not (
                current.events
                and current.events[-1].kind == "assistant"
                and current.events[-1].text == response
            ):
                self.append(inspection_id, InspectionEvent("assistant", text=response))
        current.status = status
        self._notify(inspection_id, InspectionUpdate("status"))

    def get(self, inspection_id: str) -> LiveInspection | None:
        return self._items.get(inspection_id)

    def subscribe(self, inspection_id: str, listener: InspectionListener) -> None:
        self._listeners[inspection_id].add(listener)

    def unsubscribe(self, inspection_id: str, listener: InspectionListener) -> None:
        listeners = self._listeners.get(inspection_id)
        if listeners is None:
            return
        listeners.discard(listener)
        if not listeners:
            self._listeners.pop(inspection_id, None)

    def _notify(self, inspection_id: str, update: InspectionUpdate) -> None:
        for listener in tuple(self._listeners.get(inspection_id, ())):
            listener(inspection_id, update)
