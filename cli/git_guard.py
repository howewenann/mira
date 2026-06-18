"""Startup guard for running MIRA in unversioned workspaces."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from config.settings import git_protection_enabled, load_settings, save_settings, set_git_protection

FIRST_GIT_PROMPT = "Git is not initialized for this workspace. Create a repository?"
SECOND_GIT_PROMPT = "Without Git, MIRA changes may be difficult to undo. Create a repository now?"
GIT_FAILURE_PROMPT = "Git could not initialize this workspace. Continue without Git?"


async def ensure_git_repository(workspace: Path, renderer: Any) -> bool:
    """Return whether startup should continue after checking Git protection."""
    workspace = workspace.expanduser().resolve()
    settings = load_settings(workspace)

    if not git_protection_enabled(settings):
        return True

    if is_git_worktree(workspace):
        return True

    if await renderer.ask_create_git_repo(FIRST_GIT_PROMPT):
        return await initialize_or_ask_to_continue(workspace, renderer)

    if await renderer.ask_create_git_repo(SECOND_GIT_PROMPT):
        return await initialize_or_ask_to_continue(workspace, renderer)

    save_settings(workspace, set_git_protection(settings, False))
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
        save_settings(workspace, set_git_protection(load_settings(workspace), False))
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
