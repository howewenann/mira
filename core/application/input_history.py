"""Workspace history for submitted frontend input."""

from pathlib import Path
from datetime import datetime

def read_prompt_history(path: Path) -> list[str]:
    """Read MIRA prompt history entries from a prompt-toolkit-style file."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []

    entries: list[str] = []
    current: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            if current:
                entries.append("\n".join(current).strip())
                current = []
            continue
        if stripped.startswith("+"):
            current.append(stripped[1:])
    if current:
        entries.append("\n".join(current).strip())
    return entries

def append_prompt_history(path: Path, text: str) -> None:
    """Append one submitted prompt to the workspace history file."""
    entry = text.strip()
    if not entry:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat(sep=" ", timespec="microseconds")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n# {timestamp}\n")
        for line in entry.splitlines():
            handle.write(f"+{line}\n")
