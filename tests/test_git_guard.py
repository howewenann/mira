"""Tests for the startup Git safety guard."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cli import git_guard
from config.settings import load_settings, save_settings, set_git_protection, git_protection_enabled


class PromptRenderer:
    """Renderer double that records Git safety prompt decisions."""

    def __init__(
        self,
        create_answers: list[bool] | None = None,
        continue_answers: list[bool] | None = None,
    ) -> None:
        """Create scripted answers for the Git prompt flow."""
        self.create_answers = list(create_answers or [])
        self.continue_answers = list(continue_answers or [])
        self.create_messages: list[str] = []
        self.continue_messages: list[str] = []

    async def ask_create_git_repo(self, message: str) -> bool:
        """Record a repository-creation prompt and return the next answer."""
        self.create_messages.append(message)
        if not self.create_answers:
            raise AssertionError("unexpected create-git prompt")
        return self.create_answers.pop(0)

    async def ask_continue_without_git(self, message: str) -> bool:
        """Record a continue-without-Git prompt and return the next answer."""
        self.continue_messages.append(message)
        if not self.continue_answers:
            raise AssertionError("unexpected continue-without-git prompt")
        return self.continue_answers.pop(0)


class GitGuardTests(unittest.IsolatedAsyncioTestCase):
    """Tests for Git safety prompt routing and persistence."""

    async def test_existing_git_worktree_skips_prompts(self) -> None:
        """A protected workspace should proceed without asking anything."""
        renderer = PromptRenderer()

        with patch("cli.git_guard.is_git_worktree", return_value=True):
            result = await git_guard.ensure_git_repository(Path.cwd(), renderer)

        self.assertTrue(result)
        self.assertEqual(renderer.create_messages, [])
        self.assertEqual(renderer.continue_messages, [])

    async def test_missing_git_prompts_once_and_initializes_on_yes(self) -> None:
        """Selecting yes on the first prompt should initialize Git."""
        renderer = PromptRenderer(create_answers=[True])

        with (
            patch("cli.git_guard.is_git_worktree", return_value=False),
            patch("cli.git_guard.init_git_repository", return_value=True) as init_git,
            tempfile.TemporaryDirectory(dir=Path.cwd()) as directory,
        ):
            workspace = Path(directory)
            result = await git_guard.ensure_git_repository(workspace, renderer)
            self.assertFalse((workspace / ".mira" / "settings.yml").exists())

        self.assertTrue(result)
        init_git.assert_called_once_with(workspace)
        self.assertEqual(renderer.create_messages, [git_guard.FIRST_GIT_PROMPT])

    async def test_no_then_yes_initializes_after_warning(self) -> None:
        """Selecting yes on the second prompt should initialize Git."""
        renderer = PromptRenderer(create_answers=[False, True])

        with (
            patch("cli.git_guard.is_git_worktree", return_value=False),
            patch("cli.git_guard.init_git_repository", return_value=True) as init_git,
            tempfile.TemporaryDirectory(dir=Path.cwd()) as directory,
        ):
            workspace = Path(directory)
            result = await git_guard.ensure_git_repository(workspace, renderer)

        self.assertTrue(result)
        init_git.assert_called_once_with(workspace)
        self.assertEqual(renderer.create_messages, [git_guard.FIRST_GIT_PROMPT, git_guard.SECOND_GIT_PROMPT])

    async def test_no_twice_disables_git_protection_and_continues(self) -> None:
        """Declining both prompts should save disabled Git protection."""
        renderer = PromptRenderer(create_answers=[False, False])

        with (
            patch("cli.git_guard.is_git_worktree", return_value=False),
            tempfile.TemporaryDirectory(dir=Path.cwd()) as directory,
        ):
            workspace = Path(directory)
            result = await git_guard.ensure_git_repository(workspace, renderer)
            self.assertFalse(git_protection_enabled(load_settings(workspace)))

        self.assertTrue(result)
        self.assertEqual(renderer.create_messages, [git_guard.FIRST_GIT_PROMPT, git_guard.SECOND_GIT_PROMPT])

    async def test_disabled_git_protection_suppresses_future_prompts(self) -> None:
        """A saved disabled Git protection setting should avoid repeat prompts."""
        renderer = PromptRenderer()

        with (
            patch("cli.git_guard.is_git_worktree", return_value=False),
            tempfile.TemporaryDirectory(dir=Path.cwd()) as directory,
        ):
            workspace = Path(directory)
            save_settings(workspace, set_git_protection(load_settings(workspace), False))
            result = await git_guard.ensure_git_repository(workspace, renderer)

        self.assertTrue(result)
        self.assertEqual(renderer.create_messages, [])
        self.assertEqual(renderer.continue_messages, [])

    async def test_init_failure_can_continue_and_save_preference(self) -> None:
        """If git init fails, choosing continue should save the preference."""
        renderer = PromptRenderer(create_answers=[True], continue_answers=[True])

        with (
            patch("cli.git_guard.is_git_worktree", return_value=False),
            patch("cli.git_guard.init_git_repository", return_value=False),
            tempfile.TemporaryDirectory(dir=Path.cwd()) as directory,
        ):
            workspace = Path(directory)
            result = await git_guard.ensure_git_repository(workspace, renderer)
            self.assertFalse(git_protection_enabled(load_settings(workspace)))

        self.assertTrue(result)
        self.assertEqual(renderer.continue_messages, [git_guard.GIT_FAILURE_PROMPT])

    async def test_init_failure_can_exit_startup(self) -> None:
        """If git init fails, choosing exit should stop startup."""
        renderer = PromptRenderer(create_answers=[True], continue_answers=[False])

        with (
            patch("cli.git_guard.is_git_worktree", return_value=False),
            patch("cli.git_guard.init_git_repository", return_value=False),
            tempfile.TemporaryDirectory(dir=Path.cwd()) as directory,
        ):
            workspace = Path(directory)
            result = await git_guard.ensure_git_repository(workspace, renderer)
            self.assertTrue(git_protection_enabled(load_settings(workspace)))

        self.assertFalse(result)
        self.assertEqual(renderer.continue_messages, [git_guard.GIT_FAILURE_PROMPT])

    async def test_git_command_failures_return_false(self) -> None:
        """Missing Git should be treated as unprotected and uninitialized."""
        with (
            patch("cli.git_guard.subprocess.run", side_effect=FileNotFoundError),
            patch("cli.git_guard.has_git_marker", return_value=False),
        ):
            self.assertFalse(git_guard.is_git_worktree(Path.cwd()))
            self.assertFalse(git_guard.init_git_repository(Path.cwd()))

    async def test_git_marker_parent_counts_when_git_is_unavailable(self) -> None:
        """A parent .git entry should still count if the git command is missing."""
        with (
            patch("cli.git_guard.subprocess.run", side_effect=FileNotFoundError),
            tempfile.TemporaryDirectory(dir=Path.cwd()) as directory,
        ):
            parent = Path(directory)
            child = parent / "child"
            (parent / ".git").mkdir()
            child.mkdir()

            self.assertTrue(git_guard.is_git_worktree(child))

    @unittest.skipUnless(shutil.which("git"), "git is not available")
    async def test_is_git_worktree_counts_parent_repository(self) -> None:
        """A child folder inside a parent Git repository should count."""
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            parent = Path(directory)
            child = parent / "child"
            child.mkdir()
            subprocess.run(
                ["git", "init", str(parent)],
                capture_output=True,
                text=True,
                check=True,
            )

            self.assertTrue(git_guard.is_git_worktree(child))


if __name__ == "__main__":
    unittest.main()
