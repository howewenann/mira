"""Textual coverage for the focused custom-tool Issues flow."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from textual.containers import Horizontal
from textual.widgets import Button, Input, Static

from agent.resources.tool_failures import ToolLoadFailure
from tests.test_textual_app import make_app, renderable_plain, wait_until
from ui.widgets import PromptBox, ToolIssuesScreen
from ui.widgets.tool_issues import PipInstallResult, parse_requirements


def failure(
    workspace: Path,
    filename: str,
    *,
    missing: str = "",
    error_type: str = "ModuleNotFoundError",
    message: str | None = None,
) -> ToolLoadFailure:
    path = workspace / ".mira" / "tools" / filename
    return ToolLoadFailure(
        identifier=f"{filename}:{missing}:{error_type}",
        source_path=path,
        display_path=f".mira/tools/{filename}",
        exception_type=error_type,
        message=message or (f"No module named '{missing}'" if missing else "expected ':'"),
        line_number=4,
        source_line=f"import {missing}" if missing else "def broken()",
        traceback_text="traceback details",
        missing_module=missing,
        suggested_requirement=missing,
    )


class ToolIssuesUiTests(unittest.IsolatedAsyncioTestCase):
    async def test_indicator_modal_grouping_close_and_command_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            failures = [
                failure(workspace, "one.py", missing="shared_dep"),
                failure(workspace, "two.py", missing="shared_dep"),
                failure(workspace, "syntax.py", error_type="SyntaxError"),
            ]
            session = {"id": "thread", "workspace": str(workspace), "turns": 0, "events": [], "dashboard": {}}
            app = make_app(workspace, session=session, tool_failures=failures)

            async with app.run_test() as pilot:
                button = app.query_one("#tool-issues-button", Button)
                self.assertTrue(button.display)
                self.assertEqual(str(button.label), "Issues 3")
                await pilot.click("#tool-issues-button")
                await pilot.pause()
                self.assertIsInstance(app.screen, ToolIssuesScreen)
                screen = app.screen
                summary = renderable_plain(screen.query_one("#tool-issues-summary", Static))
                self.assertIn("2 files are missing packages", summary)
                self.assertIn("1 file has another error", summary)
                self.assertEqual(summary.count("@tool runs inside MIRA."), 1)
                self.assertIn(".mira/examples/tools/project_runtime_tool.py", summary)
                self.assertEqual(screen.query_one("#tool-issues-packages", Input).value, "shared_dep")
                actions = screen.query_one("#tool-issues-actions", Horizontal)
                self.assertEqual(len(actions.query(Button)), 2)
                await pilot.press("escape")
                await pilot.pause()
                self.assertNotIsInstance(app.screen, ToolIssuesScreen)
                self.assertTrue(button.display)
                self.assertEqual(session["events"], [])

                prompt = app.query_one(PromptBox)
                prompt.value = "/issues"
                prompt.focus()
                await pilot.press("enter")
                await pilot.pause()
                self.assertIsInstance(app.screen, ToolIssuesScreen)
                self.assertEqual(session["events"], [])

    async def test_no_failures_hides_indicator(self) -> None:
        app = make_app(tool_failures=[])
        async with app.run_test():
            self.assertFalse(app.query_one("#tool-issues-button", Button).display)

    async def test_escape_is_disabled_during_install(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            app = make_app(workspace, tool_failures=[failure(workspace, "one.py", missing="dep")])
            async with app.run_test() as pilot:
                app._open_tool_issues()
                await pilot.pause()
                screen = app.screen
                self.assertIsInstance(screen, ToolIssuesScreen)
                screen.installing = True
                screen._sync_controls()
                await pilot.press("escape")
                await pilot.pause()
                self.assertIs(app.screen, screen)
                self.assertTrue(screen.query_one("#tool-issues-close", Button).disabled)

    async def test_install_worker_uses_one_shell_free_mira_python_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            app = make_app(workspace, tool_failures=[failure(workspace, "one.py", missing="alpha")])
            async with app.run_test() as pilot:
                app._open_tool_issues()
                await pilot.pause()
                screen = app.screen
                self.assertIsInstance(screen, ToolIssuesScreen)
                screen.query_one("#tool-issues-packages", Input).value = "alpha beta==2"
                app.reload_after_tool_install = AsyncMock()  # type: ignore[method-assign]
                completed = Mock(returncode=0, stdout="installed", stderr="")
                with patch("ui.widgets.tool_issues.subprocess.run", return_value=completed) as run:
                    await pilot.click("#tool-issues-install")
                    await wait_until(lambda: run.called)
                    await wait_until(lambda: app.reload_after_tool_install.await_count == 1)
                command = run.call_args.args[0]
                self.assertEqual(command[1:4], ["-m", "pip", "install"])
                self.assertEqual(command[4:], ["alpha", "beta==2"])
                self.assertFalse(run.call_args.kwargs["shell"])

    async def test_pip_failure_reenables_controls_without_reload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            app = make_app(workspace, tool_failures=[failure(workspace, "one.py", missing="alpha")])
            async with app.run_test() as pilot:
                app._open_tool_issues()
                await pilot.pause()
                screen = app.screen
                self.assertIsInstance(screen, ToolIssuesScreen)
                app.reload_after_tool_install = AsyncMock()  # type: ignore[method-assign]
                screen.installing = True
                await screen._install_finished(PipInstallResult(["alpha"], 1, "out", "network failed"))
                self.assertEqual(app.reload_after_tool_install.await_count, 0)
                self.assertFalse(screen.installing)
                self.assertFalse(screen.query_one("#tool-issues-install", Button).disabled)
                self.assertIn("network failed", renderable_plain(screen.query_one("#tool-issues-summary", Static)))
                self.assertIn("stdout:", renderable_plain(screen.query_one("#tool-issues-summary", Static)))

    async def test_success_closes_resolved_screen_and_remaining_failure_refreshes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            first = failure(workspace, "one.py", missing="alpha")
            second = failure(workspace, "two.py", missing="beta")
            app = make_app(workspace, tool_failures=[first])
            async with app.run_test() as pilot:
                app._open_tool_issues()
                await pilot.pause()
                screen = app.screen
                self.assertIsInstance(screen, ToolIssuesScreen)

                async def resolve_all() -> None:
                    app.tool_failures = []

                app._reload_runtime = AsyncMock(side_effect=resolve_all)  # type: ignore[method-assign]
                screen.installing = True
                await app.reload_after_tool_install(screen, PipInstallResult(["alpha"], 0, "ok", ""))
                await pilot.pause()
                self.assertNotIsInstance(app.screen, ToolIssuesScreen)
                self.assertFalse(app.query_one("#tool-issues-button", Button).display)

                app.tool_failures = [first]
                app._open_tool_issues()
                await pilot.pause()
                screen = app.screen

                async def leave_second() -> None:
                    app.tool_failures = [second]

                app._reload_runtime = AsyncMock(side_effect=leave_second)  # type: ignore[method-assign]
                screen.installing = True
                await app.reload_after_tool_install(screen, PipInstallResult(["alpha"], 0, "ok", ""))
                self.assertIs(app.screen, screen)
                self.assertFalse(screen.installing)
                self.assertEqual(screen.query_one("#tool-issues-packages", Input).value, "beta")

    async def test_toast_is_grouped_and_deduplicated_by_failure_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            first = failure(workspace, "one.py", missing="alpha")
            second = failure(workspace, "two.py", missing="beta")
            app = make_app(workspace, tool_failures=[])
            async with app.run_test():
                app._notified_tool_failure_sets.clear()
                app.notify = Mock()  # type: ignore[method-assign]
                app.tool_failures = [first]
                app._sync_tool_issues(notify=True)
                app._sync_tool_issues(notify=True)
                self.assertEqual(app.notify.call_count, 1)
                app.tool_failures = [first, second]
                app._sync_tool_issues(notify=True)
                self.assertEqual(app.notify.call_count, 2)

    def test_requirement_parser_rejects_empty_shell_operators_and_pip_options(self) -> None:
        self.assertEqual(parse_requirements("alpha beta==2"), ["alpha", "beta==2"])
        self.assertEqual(parse_requirements('"./local packages/alpha"'), ["./local packages/alpha"])
        for value in ("", "alpha | beta", "--target elsewhere alpha"):
            with self.assertRaises(ValueError):
                parse_requirements(value)


if __name__ == "__main__":
    unittest.main()
