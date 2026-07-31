"""Focused tests for one-shot task and rubric inputs."""

from __future__ import annotations

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import typer
from typer.testing import CliRunner

from cli import commands
from cli.commands import _read_text_input, _resolve_one_shot_inputs
from cli.main import app as cli_app
from runtime.runner import TurnResult


class Store:
    def __init__(self) -> None:
        self.saved: list[dict[str, Any]] = []

    def save(self, record: dict[str, Any]) -> None:
        self.saved.append(record)


class OutcomeRenderer:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def system_message(self, text: str, *, kind: str = "system") -> None:
        self.messages.append((kind, text))


class OneShotInputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.workspace = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_bytes(self, name: str, value: bytes) -> Path:
        path = self.workspace / name
        path.write_bytes(value)
        return path

    def assert_exit_two(self, operation: Any) -> None:
        with self.assertRaises(typer.Exit) as raised:
            operation()
        self.assertEqual(raised.exception.exit_code, 2)

    def test_cli_exposes_all_one_shot_inputs(self) -> None:
        result = CliRunner().invoke(cli_app, ["--help"])
        self.assertEqual(result.exit_code, 0)
        for option in ("--prompt", "-p", "--file", "-f", "--rubric", "--rubric-file"):
            self.assertIn(option, result.output)

    def test_literal_task_and_rubric_are_preserved_exactly(self) -> None:
        task = "  final task\n"
        rubric = "  exact rubric\n"
        self.assertEqual(
            _resolve_one_shot_inputs(task, None, rubric, None, self.workspace),
            (task, rubric),
        )

    def test_task_files_accept_arbitrary_extensions(self) -> None:
        for name in ("task.md", "task.txt", "task.yaml", "task"):
            with self.subTest(name=name):
                path = self.write_bytes(name, b"exact task\n")
                task, rubric = _resolve_one_shot_inputs(None, path, None, None, self.workspace)
                self.assertEqual(task, "exact task\n")
                self.assertIsNone(rubric)

    def test_rubric_file_accepts_arbitrary_extension_and_is_exact(self) -> None:
        path = self.write_bytes("criteria.json", b'{"must":"remain text"}\n')
        task, rubric = _resolve_one_shot_inputs("task", None, None, path, self.workspace)
        self.assertEqual(task, "task")
        self.assertEqual(rubric, '{"must":"remain text"}\n')

    def test_conflicting_task_inputs_return_exit_two(self) -> None:
        path = self.write_bytes("task", b"task")
        self.assert_exit_two(
            lambda: _resolve_one_shot_inputs("task", path, None, None, self.workspace)
        )

    def test_conflicting_rubric_inputs_return_exit_two(self) -> None:
        path = self.write_bytes("rubric", b"criteria")
        self.assert_exit_two(
            lambda: _resolve_one_shot_inputs("task", None, "criteria", path, self.workspace)
        )

    def test_rubric_without_task_returns_exit_two(self) -> None:
        self.assert_exit_two(
            lambda: _resolve_one_shot_inputs(None, None, "criteria", None, self.workspace)
        )

    def test_blank_literal_inputs_return_exit_two(self) -> None:
        for task, rubric in ((" \n", None), ("task", "\t")):
            with self.subTest(task=task, rubric=rubric):
                self.assert_exit_two(
                    lambda task=task, rubric=rubric: _resolve_one_shot_inputs(
                        task,
                        None,
                        rubric,
                        None,
                        self.workspace,
                    )
                )

    def test_invalid_task_and_rubric_files_return_exit_two(self) -> None:
        cases = [
            self.workspace / "missing",
            self.workspace,
            self.write_bytes("empty", b""),
            self.write_bytes("whitespace", b" \n"),
            self.write_bytes("invalid", b"\xff"),
        ]
        for argument in ("--file/-f", "--rubric-file"):
            for path in cases:
                with self.subTest(argument=argument, path=path):
                    self.assert_exit_two(
                        lambda path=path, argument=argument: _read_text_input(
                            path,
                            self.workspace,
                            argument,
                        )
                    )

    def test_unreadable_file_returns_exit_two(self) -> None:
        path = self.write_bytes("unreadable", b"text")
        with patch.object(Path, "read_text", side_effect=PermissionError("denied")):
            self.assert_exit_two(
                lambda: _read_text_input(path, self.workspace, "--rubric-file")
            )


class OneShotExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_prompt_and_file_without_rubric_preserve_one_shot_dispatch(self) -> None:
        for use_file in (False, True):
            with self.subTest(use_file=use_file):
                captured: dict[str, Any] = {}
                config = {"tool_output_chars": 80, "settings": {}}

                async def bootstrap(**kwargs: Any) -> dict[str, Any]:
                    return {"renderer": object(), "tool_failures": []}

                async def one_shot(
                    app_state: dict[str, Any],
                    prompt: str,
                    *,
                    rubric: str | None = None,
                ) -> TurnResult:
                    captured.update(prompt=prompt, rubric=rubric)
                    return TurnResult()

                with (
                    tempfile.TemporaryDirectory(dir=Path.cwd()) as directory,
                    patch("config.runtime.load_effective_config", return_value=config),
                    patch("ui.renderer.Renderer", return_value=object()),
                    patch(
                        "cli.git_guard.ensure_git_repository",
                        new=AsyncMock(return_value=True),
                    ),
                    patch("cli.commands._bootstrap", new=bootstrap),
                    patch("cli.commands._run_one_shot", new=one_shot),
                ):
                    workspace = Path(directory)
                    task_file = workspace / "task.anything"
                    task_file.write_text("file task", encoding="utf-8")
                    await commands._run(
                        prompt=None if use_file else "literal task",
                        prompt_file=task_file if use_file else None,
                        resume=False,
                        workspace=workspace,
                        session=None,
                    )

                self.assertEqual(captured["prompt"], "file task" if use_file else "literal task")
                self.assertIsNone(captured["rubric"])

    async def test_invocation_rubric_enables_middleware_without_saving_setting(self) -> None:
        config = {
            "tool_output_chars": 80,
            "settings": {"system": {"rubric": {"enabled": False, "max_iterations": 4}}},
        }
        original = deepcopy(config)
        captured: dict[str, Any] = {}

        async def bootstrap(**kwargs: Any) -> dict[str, Any]:
            captured["config"] = kwargs["config"]
            return {"renderer": object(), "tool_failures": []}

        async def one_shot(
            app_state: dict[str, Any],
            prompt: str,
            *,
            rubric: str | None = None,
        ) -> TurnResult:
            captured.update(prompt=prompt, rubric=rubric)
            return TurnResult(rubric_status="satisfied")

        with (
            tempfile.TemporaryDirectory(dir=Path.cwd()) as directory,
            patch("config.runtime.load_effective_config", return_value=config),
            patch("ui.renderer.Renderer", return_value=object()),
            patch(
                "cli.git_guard.ensure_git_repository",
                new=AsyncMock(return_value=True),
            ),
            patch("cli.commands._bootstrap", new=bootstrap),
            patch("cli.commands._run_one_shot", new=one_shot),
        ):
            workspace = Path(directory)
            settings_path = workspace / ".mira" / "settings.yml"
            settings_path.parent.mkdir()
            settings_path.write_text(
                "system:\n  rubric:\n    enabled: false\n    max_iterations: 4\n",
                encoding="utf-8",
            )
            before = settings_path.read_bytes()
            await commands._run(
                prompt="  exact task\n",
                rubric="  exact rubric\n",
                resume=False,
                workspace=workspace,
                session=None,
            )
            after = settings_path.read_bytes()

        self.assertEqual(captured["prompt"], "  exact task\n")
        self.assertEqual(captured["rubric"], "  exact rubric\n")
        self.assertTrue(captured["config"]["settings"]["system"]["rubric"]["enabled"])
        self.assertEqual(config, original)
        self.assertEqual(after, before)

    async def test_exact_rubric_reaches_runner_without_goal_or_plan_state(self) -> None:
        session = {"id": "session-1", "events": [], "turns": 0}
        outcome_renderer = OutcomeRenderer()
        app_state = {
            "agent": object(),
            "config": {
                "settings": {
                    "system": {"rubric": {"enabled": False, "max_iterations": 4}}
                }
            },
            "renderer": outcome_renderer,
            "session": session,
            "store": Store(),
        }
        result = TurnResult(final_text="done", rubric_status="satisfied")
        run_turn = AsyncMock(return_value=result)

        with (
            patch("runtime.runner.run_turn", new=run_turn),
            patch(
                "session.context.sync_deepagents_compaction",
                new=AsyncMock(return_value=False),
            ),
        ):
            returned = await commands._run_one_shot(
                app_state,
                "task",
                rubric="  exact rubric\n",
            )

        self.assertIs(returned, result)
        self.assertEqual(run_turn.await_args.kwargs["rubric"], "  exact rubric\n")
        self.assertEqual(run_turn.await_args.kwargs["rubric_max_iterations"], 4)
        self.assertTrue(run_turn.await_args.kwargs["include_rubric_state"])
        event_types = [event["type"] for event in session["events"]]
        self.assertEqual(event_types, ["user", "assistant", "info"])
        self.assertNotIn("plan", event_types)
        self.assertEqual(
            outcome_renderer.messages,
            [("info", "Rubric outcome: satisfied.")],
        )

    async def test_success_and_exhaustion_exit_codes(self) -> None:
        config = {"tool_output_chars": 80, "settings": {}}
        for status, expected in (("satisfied", None), ("max_iterations_reached", 3)):
            with self.subTest(status=status):
                with (
                    tempfile.TemporaryDirectory(dir=Path.cwd()) as directory,
                    patch("config.runtime.load_effective_config", return_value=config),
                    patch("ui.renderer.Renderer", return_value=object()),
                    patch(
                        "cli.git_guard.ensure_git_repository",
                        new=AsyncMock(return_value=True),
                    ),
                    patch(
                        "cli.commands._bootstrap",
                        new=AsyncMock(
                            return_value={"renderer": object(), "tool_failures": []}
                        ),
                    ),
                    patch(
                        "cli.commands._run_one_shot",
                        new=AsyncMock(
                            return_value=TurnResult(rubric_status=status)
                        ),
                    ),
                ):
                    operation = commands._run(
                        prompt="task",
                        rubric="criteria",
                        resume=False,
                        workspace=Path(directory),
                        session=None,
                    )
                    if expected is None:
                        await operation
                    else:
                        with self.assertRaises(typer.Exit) as raised:
                            await operation
                        self.assertEqual(raised.exception.exit_code, expected)

    def test_runtime_failure_returns_exit_one(self) -> None:
        with patch("cli.main.run", side_effect=RuntimeError("provider failed")):
            result = CliRunner().invoke(cli_app, ["--prompt", "task"])
        self.assertEqual(result.exit_code, 1)


if __name__ == "__main__":
    unittest.main()
