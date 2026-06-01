"""Startup guard for running MIRA in unversioned workspaces."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

FIRST_GIT_PROMPT = "Git is not initialized for this workspace. Create a repository?"
SECOND_GIT_PROMPT = "Without Git, MIRA changes may be difficult to undo. Create a repository now?"
GIT_FAILURE_PROMPT = "Git could not initialize this workspace. Continue without Git?"
GIT_SAFETY_FILE = "git_safety.json"


async def ensure_git_repository(workspace: Path, renderer: Any) -> bool:
    """Return whether startup should continue after checking Git protection."""
    workspace = workspace.expanduser().resolve()

    if is_git_worktree(workspace):
        return True

    if continue_without_git_is_saved(workspace):
        return True

    if await renderer.ask_create_git_repo(FIRST_GIT_PROMPT):
        return await initialize_or_ask_to_continue(workspace, renderer)

    if await renderer.ask_create_git_repo(SECOND_GIT_PROMPT):
        return await initialize_or_ask_to_continue(workspace, renderer)

    save_continue_without_git(workspace)
    return True


def is_git_worktree(workspace: Path) -> bool:
    """Return whether workspace is inside a Git worktree."""
    try:
        result = subprocess.run(
            ["git", "-C", str(workspace), "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            check=False,
        )
    except (FileNotFoundError, OSError):
        return has_git_marker(workspace)

    return (result.returncode == 0 and result.stdout.strip() == "true") or has_git_marker(workspace)


def has_git_marker(workspace: Path) -> bool:
    """Return whether workspace or one of its parents contains a .git entry."""
    return any((path / ".git").exists() for path in (workspace, *workspace.parents))


async def initialize_or_ask_to_continue(workspace: Path, renderer: Any) -> bool:
    """Initialize Git, or ask whether to continue if initialization fails."""
    if init_git_repository(workspace):
        return True

    if await renderer.ask_continue_without_git(GIT_FAILURE_PROMPT):
        save_continue_without_git(workspace)
        return True

    return False


def init_git_repository(workspace: Path) -> bool:
    """Return whether `git init` succeeded for the workspace."""
    try:
        result = subprocess.run(
            ["git", "init", str(workspace)],
            capture_output=True,
            text=True,
            check=False,
        )
    except (FileNotFoundError, OSError):
        return False

    return result.returncode == 0


def continue_without_git_is_saved(workspace: Path) -> bool:
    """Return whether the user previously chose to run without Git here."""
    data = read_git_safety(workspace)
    return bool(data.get("continue_without_git"))


def save_continue_without_git(workspace: Path) -> bool:
    """Persist the user's choice to continue without Git for this workspace."""
    path = git_safety_path(workspace)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"continue_without_git": True}, indent=2), encoding="utf-8")
    except OSError:
        return False

    return True


def read_git_safety(workspace: Path) -> dict[str, Any]:
    """Read the persisted Git safety decision, returning empty data on errors."""
    path = git_safety_path(workspace)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    return data if isinstance(data, dict) else {}


def git_safety_path(workspace: Path) -> Path:
    """Return the workspace-local Git safety preference path."""
    return workspace / ".mira" / GIT_SAFETY_FILE
