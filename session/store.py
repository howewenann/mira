"""JSON-backed session metadata storage."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from session.context import context_policy, normalize_session
from session.dashboard import normalize_dashboard


def new_session_id() -> str:
    """Return a sortable, readable default session id."""
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S%z")
    suffix = uuid.uuid4().hex[:8]
    return f"{timestamp}-{suffix}"


class SessionStore:
    """Persist durable session records as JSON files."""

    def __init__(self, root: Path) -> None:
        """Create the session directory if it does not exist yet."""
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def load(
        self,
        session_id: str | None,
        resume: bool,
        workspace: Path,
        policy: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """Load an explicit session, resume the latest, or create a new one."""
        if session_id:
            path = self.path(session_id)
            if path.exists():
                return self.read(path, policy=policy)

            record = self.new(session_id=session_id, workspace=workspace, policy=policy)
            self.save(record)
            return record

        if resume:
            latest = self.latest()
            if latest:
                return self.read(latest, policy=policy)

        record = self.new(session_id=None, workspace=workspace, policy=policy)
        self.save(record)
        return record

    def new(
        self,
        session_id: str | None,
        workspace: Path,
        policy: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """Build a new session record without writing it to disk."""
        now = datetime.now(timezone.utc).isoformat()

        return {
            "id": session_id or new_session_id(),
            "title": "Untitled session",
            "workspace": str(workspace),
            "created_at": now,
            "updated_at": now,
            "turns": 0,
            "dashboard": normalize_dashboard(None),
            "context_policy": policy or context_policy(),
            "summary": None,
            "messages": [],
        }

    def save(self, record: dict[str, Any]) -> None:
        """Update the timestamp and write the session JSON file."""
        record["updated_at"] = datetime.now(timezone.utc).isoformat()
        normalized = normalize_session(record, policy=record.get("context_policy"))
        self.path(str(record["id"])).write_text(json.dumps(normalized, indent=2), encoding="utf-8")
        record.clear()
        record.update(normalized)

    def read(self, path: Path, policy: dict[str, int] | None = None) -> dict[str, Any]:
        """Read a session record from a JSON file."""
        record = json.loads(path.read_text(encoding="utf-8"))
        return normalize_session(record, policy=policy or record.get("context_policy"))

    def latest(self) -> Path | None:
        """Return the most recently modified session file, if any exist."""
        paths = list(self.root.glob("*.json"))
        if not paths:
            return None

        return max(paths, key=lambda path: path.stat().st_mtime)

    def path(self, session_id: str) -> Path:
        """Return the JSON path for a session id."""
        return self.root / f"{session_id}.json"
