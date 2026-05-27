import json
import uuid
from datetime import datetime, timezone
from pathlib import Path


class SessionStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def load(self, session_id: str | None, resume: bool, workspace: Path) -> dict:
        if session_id:
            path = self.path(session_id)
            if path.exists():
                return self.read(path)

            record = self.new(session_id=session_id, workspace=workspace)
            self.save(record)
            return record

        if resume:
            latest = self.latest()
            if latest:
                return self.read(latest)

        record = self.new(session_id=None, workspace=workspace)
        self.save(record)
        return record

    def new(self, session_id: str | None, workspace: Path) -> dict:
        now = datetime.now(timezone.utc).isoformat()

        return {
            "id": session_id or uuid.uuid4().hex[:12],
            "workspace": str(workspace),
            "created_at": now,
            "updated_at": now,
            "turns": 0,
        }

    def save(self, record: dict) -> None:
        record["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.path(record["id"]).write_text(json.dumps(record, indent=2), encoding="utf-8")

    def read(self, path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    def latest(self) -> Path | None:
        paths = list(self.root.glob("*.json"))
        if not paths:
            return None

        return max(paths, key=lambda path: path.stat().st_mtime)

    def path(self, session_id: str) -> Path:
        return self.root / f"{session_id}.json"
