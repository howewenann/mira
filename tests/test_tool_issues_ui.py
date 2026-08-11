"""Focused tests for the generic, view-only Issues surface."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from textual.app import App, ComposeResult
from textual.widgets import Button, Collapsible, Static

from agent.resources.tool_failures import ToolLoadFailure, tool_failure_issues
from runtime.issues import Issue
from ui.widgets import IssuesScreen


class IssuesApp(App[None]):
    CSS_PATH = "../ui/styles/mira.tcss"

    def __init__(self, issues: list[Issue]) -> None:
        super().__init__()
        self._issues = issues

    def compose(self) -> ComposeResult:
        yield Static("host")

    def on_mount(self) -> None:
        self.push_screen(IssuesScreen(self._issues))


class IssuesScreenTests(unittest.IsolatedAsyncioTestCase):
    async def test_rows_are_flat_collapsed_sorted_and_view_only(self) -> None:
        app = IssuesApp(
            [
                Issue("TOOL", "tool failed", "tool.py", "boom", "fix it"),
                Issue("STARTUP", "startup failed", "settings.yml", "bad", "reload"),
            ]
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            self.assertIsInstance(app.screen, IssuesScreen)
            rows = list(app.screen.query(Collapsible))
            self.assertEqual([row.title for row in rows], ["[STARTUP] startup failed", "[TOOL] tool failed"])
            self.assertTrue(all(row.collapsed for row in rows))
            self.assertEqual(len(app.screen.query("Input")), 0)
            self.assertEqual(len(app.screen.query("#issues-title-close")), 1)
            self.assertFalse(app.screen.query_one("#issues-title-close", Button).has_focus)
            self.assertIsNone(app.focused)

    async def test_escape_closes_screen(self) -> None:
        app = IssuesApp([Issue("MODEL", "missing")])
        async with app.run_test() as pilot:
            await pilot.press("escape")
            await pilot.pause()
            self.assertNotIsInstance(app.screen, IssuesScreen)


class ToolIssueProjectionTests(unittest.TestCase):
    def test_missing_dependency_guidance_has_wrapper_pip_and_reload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "broken.py"
            failure = ToolLoadFailure(
                identifier="x",
                source_path=source,
                display_path=".mira/tools/broken.py",
                exception_type="ModuleNotFoundError",
                message="No module named 'example_dep'",
                line_number=3,
                source_line="import example_dep",
                traceback_text="traceback",
                missing_module="example_dep",
                suggested_requirement="example_dep",
            )
            issue = tool_failure_issues([failure])[0]

        self.assertEqual(issue.category, "TOOL")
        self.assertIn("@project_tool", issue.guidance)
        self.assertIn(f'"{sys.executable}" -m pip install example_dep', issue.guidance)
        self.assertIn("/reload", issue.guidance)


if __name__ == "__main__":
    unittest.main()
