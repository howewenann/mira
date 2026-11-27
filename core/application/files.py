"""Project file discovery through the configured native backend."""

import asyncio
from collections import deque
from typing import Any

FILE_DISCOVERY_CONCURRENCY = 32

EXCLUDED_FILE_COMPONENTS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    "build",
    "dist",
}

async def discover_project_files(backend: Any) -> list[str]:
    """Enumerate project files through backend ls calls while pruning noisy trees."""
    pending = deque(["/"])
    visited: set[str] = set()
    files: set[str] = set()
    while pending:
        directories: list[str] = []
        while pending and len(directories) < FILE_DISCOVERY_CONCURRENCY:
            directory = pending.popleft()
            if directory in visited:
                continue
            visited.add(directory)
            directories.append(directory)
        if not directories:
            continue

        results = await asyncio.gather(*(backend.als(directory) for directory in directories))
        for result in results:
            for entry in getattr(result, "entries", None) or []:
                if not isinstance(entry, dict):
                    continue
                path = str(entry.get("path") or "").replace("\\", "/")
                relative = path.strip("/")
                if not relative or _excluded_project_path(relative):
                    continue
                if entry.get("is_dir"):
                    pending.append(f"/{relative}")
                else:
                    files.add(relative)
    return sorted(files, key=lambda path: (path.casefold(), path))

def _excluded_project_path(path: str) -> bool:
    components = {component.casefold() for component in path.split("/")}
    return bool(components & EXCLUDED_FILE_COMPONENTS)
