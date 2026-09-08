"""Workspace-level capabilities shared by every MIRA frontend."""

from __future__ import annotations

import subprocess
from pathlib import Path

from core.diagnostics.issues import Issue

GIT_NOT_CONFIGURED_SUMMARY = "Git protection is not configured"


def is_git_worktree(workspace: Path) -> bool:
    """Return whether the workspace is inside a Git worktree."""
    workspace = workspace.expanduser().resolve()
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
    """Return whether the workspace or one of its parents contains a .git entry."""
    workspace = workspace.expanduser().resolve()
    return any((path / ".git").exists() for path in (workspace, *workspace.parents))


def init_git_repository(workspace: Path) -> bool:
    """Initialize a normal Git repository without raising into MIRA startup."""
    try:
        result = subprocess.run(
            ["git", "init", str(workspace.expanduser().resolve())],
            capture_output=True,
            text=True,
            check=False,
        )
    except (FileNotFoundError, OSError):
        return False
    return result.returncode == 0


def git_protection_issue(workspace: Path, preference: bool | None) -> Issue | None:
    """Return the non-blocking undecided-Git diagnostic when applicable."""
    if preference is not None or is_git_worktree(workspace):
        return None
    return Issue(
        "STARTUP",
        GIT_NOT_CONFIGURED_SUMMARY,
        str(workspace.expanduser().resolve()),
        "This workspace is not inside a Git worktree.",
        "Open Settings > General and choose Configure for Git Protection.",
    )


__all__ = [
    "GIT_NOT_CONFIGURED_SUMMARY",
    "git_protection_issue",
    "has_git_marker",
    "init_git_repository",
    "is_git_worktree",
]
