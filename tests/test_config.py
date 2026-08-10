"""Tests for CLI configuration and startup behavior."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import typer
from typer.testing import CliRunner

from cli import commands
from cli.main import app as cli_app
from config.llm import ConfigError
from config.runtime import LaunchOptions


class CLIConfigTests(unittest.TestCase):
    """Tests for user-facing CLI config errors."""

    def test_help_includes_short_flags_without_workspace_default_path(self) -> None:
        """The CLI should expose short aliases without leaking cwd as a default."""
        result = CliRunner().invoke(cli_app, ["--help"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("-d", result.output)
        self.assertIn("--direct", result.output)
        self.assertIn("-t", result.output)
        self.assertIn("--trace", result.output)
        self.assertIn("-f", result.output)
        self.assertIn("--file", result.output)
        self.assertIn("-h", result.output)
        self.assertIn("--help", result.output)
        self.assertIn("-w", result.output)
        self.assertIn("--workspace", result.output)
        self.assertNotIn("[default: ", result.output)

    def test_cli_file_options_pass_prompt_file_to_run(self) -> None:
        """The public file options should map to the internal prompt_file argument."""
        for option in ("--file", "-f"):
            with self.subTest(option=option), patch("cli.main.run") as run:
                result = CliRunner().invoke(cli_app, [option, "prompt.markdown"])

                self.assertEqual(result.exit_code, 0)
                run.assert_called_once()
                self.assertEqual(run.call_args.kwargs["prompt"], None)
                self.assertEqual(run.call_args.kwargs["prompt_file"], Path("prompt.markdown"))

    def test_cli_trace_option_passes_to_run(self) -> None:
        """The public trace option should map to the internal trace argument."""
        for option in ("--trace", "-t"):
            with self.subTest(option=option), patch("cli.main.run") as run:
                result = CliRunner().invoke(cli_app, [option])

                self.assertEqual(result.exit_code, 0)
                run.assert_called_once()
                self.assertTrue(run.call_args.kwargs["trace"])

    def test_cli_direct_option_passes_to_run(self) -> None:
        """The public direct option should remain a launch-time CLI argument."""
        for option in ("--direct", "-d"):
            with self.subTest(option=option), patch("cli.main.run") as run:
                result = CliRunner().invoke(cli_app, [option])

                self.assertEqual(result.exit_code, 0)
                run.assert_called_once()
                self.assertTrue(run.call_args.kwargs["direct"])

    def test_run_prints_config_errors_without_traceback(self) -> None:
        """Config errors should exit cleanly through Typer."""
        with (
            patch("cli.commands._suppress_known_warnings"),
            patch("config.loader.load_config", side_effect=ConfigError("choose one provider")),
            patch("typer.echo") as echo,
        ):
            with self.assertRaises(typer.Exit) as raised:
                commands.run(prompt=None, resume=False, workspace=Path("."), session=None)

        self.assertEqual(raised.exception.exit_code, 2)
        echo.assert_called_once_with("choose one provider", err=True)


class CLIStartupTests(unittest.IsolatedAsyncioTestCase):
    """Tests for startup ordering around the Git safety guard."""

    async def _captured_one_shot_text(
        self,
        *,
        prompt: str | None = None,
        prompt_file: Path | None = None,
        workspace: Path,
    ) -> tuple[str, str]:
        config = {
            "tool_output_chars": 123,
            "session_dir": "unused",
            "llm_provider": "lmstudio",
            "llm_model": "local-model",
        }
        session_record = {"id": "thread-1", "events": [], "turns": 0, "dashboard": {}}
        captured: list[str] = []

        async def ensure_git_repository(workspace: Path, guard_renderer: object) -> bool:
            return True

        async def bootstrap(
            workspace: Path,
            session: str | None,
            resume: bool,
            config: dict[str, object] | None = None,
            renderer: object | None = None,
        ) -> dict[str, object]:
            return {
                "agent": "agent",
                "renderer": renderer,
                "session": session_record,
                "store": type("Store", (), {"save": lambda self, record: None})(),
            }

        async def run_turn(agent: object, text: str, renderer: object, thread_id: str) -> object:
            captured.append(text)
            return type("Result", (), {"final_text": "done"})()

        with (
            patch("config.loader.load_config", return_value=config),
            patch("ui.renderer.Renderer", return_value=object()),
            patch("cli.git_guard.ensure_git_repository", ensure_git_repository),
            patch("cli.commands._bootstrap", bootstrap),
            patch("runtime.runner.run_turn", run_turn),
        ):
            await commands._run(
                prompt=prompt,
                resume=False,
                workspace=workspace,
                session=None,
                prompt_file=prompt_file,
            )

        return captured[0], session_record["events"][0]["text"]

    async def test_run_checks_git_before_bootstrap(self) -> None:
        """The Git guard should run before sessions, resources, or agents are created."""
        events: list[str] = []
        config = {
            "tool_output_chars": 123,
            "session_dir": "unused",
            "llm_provider": "lmstudio",
            "llm_model": "local-model",
        }
        renderer = object()
        session_record = {"id": "thread-1"}
        case = self

        class Store:
            """Store double that records saves."""

            def save(self, record: dict[str, str]) -> None:
                """Record that the one-shot session was saved."""
                events.append("save")
                case.assertEqual(record["id"], session_record["id"])
                case.assertEqual(record["events"][0]["text"], "hello")

        def load_config(workspace: Path) -> dict[str, object]:
            events.append("config")
            return config

        def make_renderer(tool_output_chars: int) -> object:
            events.append("renderer")
            self.assertEqual(tool_output_chars, 123)
            return renderer

        async def ensure_git_repository(workspace: Path, guard_renderer: object) -> bool:
            events.append("guard")
            self.assertIs(guard_renderer, renderer)
            return True

        async def bootstrap(
            workspace: Path,
            session: str | None,
            resume: bool,
            config: dict[str, object] | None = None,
            renderer: object | None = None,
        ) -> dict[str, object]:
            events.append("bootstrap")
            self.assertIsNot(config, config_data)
            self.assertEqual(
                config,
                {**config_data, "llm_direct": False},
            )
            self.assertIs(renderer, renderer_obj)
            return {
                "agent": "agent",
                "renderer": renderer_obj,
                "session": session_record,
                "store": Store(),
            }

        async def run_turn(
            agent: object,
            text: str,
            renderer: object,
            thread_id: str,
        ) -> None:
            events.append("run_turn")
            self.assertEqual((agent, text, thread_id), ("agent", "hello", "thread-1"))
            self.assertIs(getattr(renderer, "renderer", None), renderer_obj)

        config_data = config
        renderer_obj = renderer

        with (
            patch("config.loader.load_config", load_config),
            patch("ui.renderer.Renderer", make_renderer),
            patch("cli.git_guard.ensure_git_repository", ensure_git_repository),
            patch("cli.commands._bootstrap", bootstrap),
            patch("runtime.runner.run_turn", run_turn),
        ):
            await commands._run(prompt="hello", resume=False, workspace=Path("."), session=None)

        self.assertEqual(events, ["config", "renderer", "guard", "bootstrap", "save", "run_turn", "save"])

    async def test_run_sets_direct_config_flag(self) -> None:
        """The CLI flag should be carried into bootstrap config."""
        config = {
            "tool_output_chars": 123,
            "session_dir": "unused",
            "llm_provider": "lmstudio",
            "llm_model": "local-model",
        }

        async def ensure_git_repository(workspace: Path, guard_renderer: object) -> bool:
            return True

        async def bootstrap(
            workspace: Path,
            session: str | None,
            resume: bool,
            config: dict[str, object] | None = None,
            renderer: object | None = None,
        ) -> dict[str, object]:
            self.assertIsNotNone(config)
            self.assertTrue(config["llm_direct"])
            return {
                "agent": "agent",
                "renderer": renderer,
                "session": {"id": "thread-1"},
                "store": type("Store", (), {"save": lambda self, record: None})(),
            }

        async def run_turn(*args: object, **kwargs: object) -> object:
            return type("Result", (), {"final_text": "done"})()

        with (
            patch("config.loader.load_config", return_value=config),
            patch("ui.renderer.Renderer", return_value=object()),
            patch("cli.git_guard.ensure_git_repository", ensure_git_repository),
            patch("cli.commands._bootstrap", bootstrap),
            patch("runtime.runner.run_turn", run_turn),
        ):
            await commands._run(
                prompt="hello",
                resume=False,
                workspace=Path("."),
                session=None,
                launch_options=LaunchOptions(llm_direct=True),
            )

    async def test_run_defaults_one_shot_to_non_direct_config(self) -> None:
        """A plain one-shot invocation should explicitly disable direct mode."""
        config = {
            "tool_output_chars": 123,
            "session_dir": "unused",
            "llm_provider": "lmstudio",
            "llm_model": "local-model",
        }

        async def ensure_git_repository(workspace: Path, guard_renderer: object) -> bool:
            return True

        async def bootstrap(
            workspace: Path,
            session: str | None,
            resume: bool,
            config: dict[str, object] | None = None,
            renderer: object | None = None,
        ) -> dict[str, object]:
            self.assertIsNotNone(config)
            self.assertFalse(config["llm_direct"])
            return {
                "agent": "agent",
                "renderer": renderer,
                "session": {"id": "thread-1"},
                "store": type("Store", (), {"save": lambda self, record: None})(),
            }

        async def run_turn(*args: object, **kwargs: object) -> object:
            return type("Result", (), {"final_text": "done"})()

        with (
            patch("config.loader.load_config", return_value=config),
            patch("ui.renderer.Renderer", return_value=object()),
            patch("cli.git_guard.ensure_git_repository", ensure_git_repository),
            patch("cli.commands._bootstrap", bootstrap),
            patch("runtime.runner.run_turn", run_turn),
        ):
            await commands._run(
                prompt="hello",
                resume=False,
                workspace=Path("."),
                session=None,
            )

    async def test_one_shot_records_system_error_when_turn_fails(self) -> None:
        """One-shot prompt failures should be visible in the saved session."""
        config = {
            "tool_output_chars": 123,
            "session_dir": "unused",
            "llm_provider": "lmstudio",
            "llm_model": "local-model",
        }
        session_record = {"id": "thread-1", "events": [], "turns": 0, "dashboard": {}}
        saved: list[dict[str, object]] = []

        async def ensure_git_repository(workspace: Path, guard_renderer: object) -> bool:
            return True

        async def bootstrap(
            workspace: Path,
            session: str | None,
            resume: bool,
            config: dict[str, object] | None = None,
            renderer: object | None = None,
        ) -> dict[str, object]:
            return {
                "agent": "agent",
                "renderer": renderer,
                "session": session_record,
                "store": type("Store", (), {"save": lambda self, record: saved.append(record.copy())})(),
            }

        async def run_turn(*args: object, **kwargs: object) -> object:
            raise RuntimeError("unexecuted tool call")

        with (
            patch("config.loader.load_config", return_value=config),
            patch("ui.renderer.Renderer", return_value=object()),
            patch("cli.git_guard.ensure_git_repository", ensure_git_repository),
            patch("cli.commands._bootstrap", bootstrap),
            patch("runtime.runner.run_turn", run_turn),
            patch("runtime.error_report.write_error_report", return_value=Path("report.txt")) as report,
        ):
            with self.assertRaisesRegex(RuntimeError, "unexecuted tool call"):
                await commands._run(prompt="hello", resume=False, workspace=Path("."), session=None)

        self.assertEqual([event["type"] for event in session_record["events"]], ["user", "system_error"])
        self.assertIn("unexecuted tool call", session_record["events"][-1]["text"])
        self.assertIn("error report: report.txt", session_record["events"][-1]["text"])
        report.assert_called_once()
        self.assertTrue(saved)

    def test_run_writes_backup_report_for_unexpected_top_level_failures(self) -> None:
        """Unexpected errors escaping the async runner should get a backup report."""
        async def fail(*args: object, **kwargs: object) -> None:
            raise RuntimeError("startup boom")

        with (
            patch("cli.commands._suppress_known_warnings"),
            patch("cli.commands._run", fail),
            patch("runtime.error_report.write_error_report", return_value=Path("backup.txt")) as report,
        ):
            with self.assertRaisesRegex(RuntimeError, "startup boom"):
                commands.run(prompt=None, resume=False, workspace=Path("."), session="requested")

        report.assert_called_once()
        self.assertEqual(report.call_args.kwargs["source"], "cli.run")
        self.assertEqual(report.call_args.kwargs["session_id"], "requested")

    def test_run_does_not_duplicate_already_reported_errors(self) -> None:
        """The backup boundary should skip exceptions reported by lower layers."""
        error = RuntimeError("already reported")
        setattr(error, "__mira_error_report_path__", "existing.txt")

        async def fail(*args: object, **kwargs: object) -> None:
            raise error

        with (
            patch("cli.commands._suppress_known_warnings"),
            patch("cli.commands._run", fail),
            patch("runtime.error_report.write_error_report") as report,
        ):
            with self.assertRaises(RuntimeError):
                commands.run(prompt="hello", resume=False, workspace=Path("."), session=None)

        report.assert_not_called()

    async def test_run_exits_when_git_guard_blocks_startup(self) -> None:
        """Choosing exit after a Git failure should stop before bootstrap."""
        renderer = object()

        async def ensure_git_repository(workspace: Path, guard_renderer: object) -> bool:
            return False

        with (
            patch("config.loader.load_config", return_value={"tool_output_chars": 123}),
            patch("ui.renderer.Renderer", return_value=renderer),
            patch("cli.git_guard.ensure_git_repository", ensure_git_repository),
            patch("cli.commands._bootstrap") as bootstrap,
        ):
            with self.assertRaises(typer.Exit) as raised:
                await commands._run(prompt="hello", resume=False, workspace=Path("."), session=None)

        self.assertEqual(raised.exception.exit_code, 1)
        bootstrap.assert_not_called()

    async def test_short_file_flag_reads_markdown_prompt(self) -> None:
        """The file prompt path should be read and sent to one-shot mode."""
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "prompt.md").write_text("# Task\nDo the thing.\n", encoding="utf-8")

            run_text, recorded_text = await self._captured_one_shot_text(
                prompt_file=Path("prompt.md"),
                workspace=workspace,
            )

        self.assertEqual(run_text, "# Task\nDo the thing.\n")
        self.assertEqual(recorded_text, "# Task\nDo the thing.\n")

    async def test_long_file_flag_accepts_markdown_extension(self) -> None:
        """The long Markdown extension should be accepted for file prompts."""
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "prompt.markdown").write_text("Use this prompt.\n", encoding="utf-8")

            run_text, recorded_text = await self._captured_one_shot_text(
                prompt_file=Path("prompt.markdown"),
                workspace=workspace,
            )

        self.assertEqual(run_text, "Use this prompt.\n")
        self.assertEqual(recorded_text, "Use this prompt.\n")

    async def test_prompt_text_remains_literal_when_markdown_file_exists(self) -> None:
        """The -p prompt text should not auto-read matching workspace files."""
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "README.md").write_text("file contents\n", encoding="utf-8")

            run_text, recorded_text = await self._captured_one_shot_text(
                prompt="README.md",
                workspace=workspace,
            )

        self.assertEqual(run_text, "README.md")
        self.assertEqual(recorded_text, "README.md")

    async def test_prompt_and_file_flags_cannot_be_combined(self) -> None:
        """One-shot startup should reject ambiguous prompt input."""
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "prompt.md").write_text("file contents\n", encoding="utf-8")

            with self.assertRaises(typer.Exit) as raised:
                await commands._run(
                    prompt="literal",
                    resume=False,
                    workspace=workspace,
                    session=None,
                    prompt_file=Path("prompt.md"),
                )

        self.assertEqual(raised.exception.exit_code, 2)

    async def test_file_flag_rejects_missing_and_directory_paths(self) -> None:
        """The file prompt input should fail before model startup for invalid paths."""
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "folder.md").mkdir()

            cases = [Path("missing.md"), Path("folder.md")]
            for prompt_file in cases:
                with self.subTest(prompt_file=prompt_file):
                    with self.assertRaises(typer.Exit) as raised:
                        await commands._run(
                            prompt=None,
                            resume=False,
                            workspace=workspace,
                            session=None,
                            prompt_file=prompt_file,
                        )

                    self.assertEqual(raised.exception.exit_code, 2)


if __name__ == "__main__":
    unittest.main()
