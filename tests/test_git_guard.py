"""Tests for non-blocking workspace Git protection."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from config.loader import load_config
from config.settings import git_protection_preference, load_settings, save_settings, set_git_protection
from core import workspace as workspace_git


class WorkspaceGitTests(unittest.TestCase):
    """Git state should produce diagnostics, never a startup gate."""

    def test_existing_worktree_has_no_issue(self) -> None:
        with (
            tempfile.TemporaryDirectory(dir=Path.cwd()) as directory,
            patch("core.workspace.is_git_worktree", return_value=True),
        ):
            config = load_config(Path(directory))

        self.assertFalse(
            any(issue.summary == workspace_git.GIT_NOT_CONFIGURED_SUMMARY for issue in config["issues"])
        )

    def test_parent_worktree_is_detected(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            parent = Path(directory)
            (parent / ".git").mkdir()
            child = parent / "nested" / "workspace"
            child.mkdir(parents=True)
            with patch("core.workspace.subprocess.run", side_effect=FileNotFoundError):
                self.assertTrue(workspace_git.is_git_worktree(child))
                self.assertIsNone(workspace_git.git_protection_issue(child, None))

    def test_undecided_unversioned_workspace_gets_startup_issue(self) -> None:
        with (
            tempfile.TemporaryDirectory(dir=Path.cwd()) as directory,
            patch("core.workspace.is_git_worktree", return_value=False),
        ):
            workspace = Path(directory)
            config = load_config(workspace)

        issue = next(
            item for item in config["issues"] if item.summary == workspace_git.GIT_NOT_CONFIGURED_SUMMARY
        )
        self.assertEqual(issue.category, "STARTUP")
        self.assertIsNone(git_protection_preference(config["settings"]))

    def test_explicit_preference_suppresses_unconfigured_issue(self) -> None:
        with (
            tempfile.TemporaryDirectory(dir=Path.cwd()) as directory,
            patch("core.workspace.is_git_worktree", return_value=False),
        ):
            workspace = Path(directory)
            configured = set_git_protection(load_settings(workspace), False)
            self.assertTrue(save_settings(workspace, configured))
            config = load_config(workspace)

        self.assertFalse(
            any(issue.summary == workspace_git.GIT_NOT_CONFIGURED_SUMMARY for issue in config["issues"])
        )

    def test_git_init_reports_success_and_failure_without_raising(self) -> None:
        workspace = Path.cwd()
        with patch(
            "core.workspace.subprocess.run",
            return_value=SimpleNamespace(returncode=0),
        ) as run:
            self.assertTrue(workspace_git.init_git_repository(workspace))
        self.assertEqual(run.call_args.args[0][:2], ["git", "init"])

        with patch("core.workspace.subprocess.run", side_effect=OSError):
            self.assertFalse(workspace_git.init_git_repository(workspace))


if __name__ == "__main__":
    unittest.main()
