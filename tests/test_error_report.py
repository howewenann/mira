"""Tests for automatic error reports and diagnostics logging."""

from __future__ import annotations

import logging
import sys
import tempfile
import unittest
import builtins
from pathlib import Path
from unittest.mock import patch
from io import StringIO
from contextlib import redirect_stdout

from logging.handlers import RotatingFileHandler

from runtime.diagnostics import (
    get_diagnostics_logger,
    open_trace_window,
    setup_diagnostics_logging,
)
from runtime.error_report import clear_error_reports, error_report_path, write_error_report
from runtime.trace_tail import main as trace_tail_main
from ui.terminal_colors import TerminalColorizer, color_for_label, colorize_line, enable_console_colors, strip_ansi


class ErrorReportTests(unittest.TestCase):
    """Tests for copy-pasteable failure artifacts."""

    def test_write_error_report_creates_timestamped_and_latest_reports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            try:
                raise RuntimeError("manual failure")
            except RuntimeError as exc:
                report_path = write_error_report(
                    exc,
                    workspace=workspace,
                    source="test.source",
                    session_id="20260702-183012+0800-a1b2c3d4",
                    context={"mode": "action"},
                )

            latest = workspace / ".mira" / "_errors" / "latest_error.txt"
            self.assertTrue(report_path.exists())
            self.assertEqual(latest.read_text(encoding="utf-8"), report_path.read_text(encoding="utf-8"))
            self.assertEqual(report_path.parent.name, "20260702-183012+0800-a1b2c3d4")
            content = report_path.read_text(encoding="utf-8")
            self.assertIn("MIRA error report", content)
            self.assertIn("Source: test.source", content)
            self.assertIn("Session ID: 20260702-183012+0800-a1b2c3d4", content)
            self.assertIn("RuntimeError: manual failure", content)
            self.assertIn('"mode": "action"', content)
            self.assertIn("Traceback (most recent call last):", content)

    def test_write_error_report_sanitizes_session_id_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            try:
                raise ValueError("bad")
            except ValueError as exc:
                report_path = write_error_report(
                    exc,
                    workspace=workspace,
                    source="test",
                    session_id="../outside/session",
                )

            errors_root = (workspace / ".mira" / "_errors").resolve()
            self.assertTrue(str(report_path.resolve()).startswith(str(errors_root)))
            self.assertNotIn("..", report_path.relative_to(errors_root).as_posix())

    def test_reported_exception_returns_existing_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            exc = RuntimeError("same")
            first = write_error_report(exc, workspace=workspace, source="first", session_id="thread")
            second = write_error_report(exc, workspace=workspace, source="second", session_id="thread")

            self.assertEqual(first, second)
            self.assertEqual(error_report_path(exc), first)
            reports = list((workspace / ".mira" / "_errors" / "thread").glob("*.txt"))
            self.assertEqual(len(reports), 1)

    def test_clear_error_reports_missing_directory_returns_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(clear_error_reports(Path(directory)), 0)

    def test_clear_error_reports_deletes_reports_and_keeps_other_mira_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            try:
                raise RuntimeError("first")
            except RuntimeError as exc:
                first = write_error_report(exc, workspace=workspace, source="test", session_id="thread-one")
            try:
                raise RuntimeError("second")
            except RuntimeError as exc:
                second = write_error_report(exc, workspace=workspace, source="test", session_id="thread-two")
            settings = workspace / ".mira" / "settings.yml"
            settings.write_text("keep: true\n", encoding="utf-8")

            removed = clear_error_reports(workspace)

            self.assertEqual(removed, 3)
            self.assertFalse(first.exists())
            self.assertFalse(second.exists())
            self.assertFalse((workspace / ".mira" / "_errors" / "latest_error.txt").exists())
            self.assertEqual(settings.read_text(encoding="utf-8"), "keep: true\n")


class DiagnosticsTests(unittest.TestCase):
    """Tests for bounded diagnostics logging and trace windows."""

    def test_setup_diagnostics_logging_uses_one_rotating_handler(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            logger = get_diagnostics_logger()
            old_handlers = list(logger.handlers)
            logger.handlers = []
            try:
                first = setup_diagnostics_logging(workspace)
                second = setup_diagnostics_logging(workspace)
                handlers = [handler for handler in logger.handlers if isinstance(handler, RotatingFileHandler)]

                self.assertEqual(first, second)
                self.assertEqual(len(handlers), 1)
                self.assertEqual(handlers[0].maxBytes, 2 * 1024 * 1024)
                self.assertEqual(handlers[0].backupCount, 3)
                self.assertEqual(handlers[0].formatter._fmt, "%(message)s")
            finally:
                for handler in logger.handlers:
                    handler.close()
                logger.handlers = old_handlers
                logger.setLevel(logging.NOTSET)

    def test_setup_diagnostics_logging_resets_current_trace_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            log_dir = workspace / ".mira" / "_logs"
            log_dir.mkdir(parents=True)
            (log_dir / "mira.log").write_text("old log\n", encoding="utf-8")
            (log_dir / "mira.log.1").write_text("old rotation\n", encoding="utf-8")
            logger = get_diagnostics_logger()
            old_handlers = list(logger.handlers)
            logger.handlers = []
            try:
                log_path = setup_diagnostics_logging(workspace)

                content = log_path.read_text(encoding="utf-8")
                self.assertIn("diagnostics logging started", content)
                self.assertNotIn("old log", content)
                self.assertFalse((log_dir / "mira.log.1").exists())
            finally:
                for handler in logger.handlers:
                    handler.close()
                logger.handlers = old_handlers
                logger.setLevel(logging.NOTSET)

    def test_open_trace_window_non_windows_is_non_fatal(self) -> None:
        with patch.object(sys, "platform", "linux"):
            self.assertFalse(open_trace_window(Path("mira.log")))

    def test_open_trace_window_launch_failure_is_non_fatal(self) -> None:
        with (
            patch.object(sys, "platform", "win32"),
            patch("runtime.diagnostics.subprocess.Popen", side_effect=OSError("no window")),
        ):
            self.assertFalse(open_trace_window(Path("mira.log")))

    def test_open_trace_window_launches_trace_tail_module(self) -> None:
        with (
            patch.object(sys, "platform", "win32"),
            patch("runtime.diagnostics.subprocess.Popen") as popen,
        ):
            self.assertTrue(open_trace_window(Path("mira.log")))

        command = popen.call_args.args[0]
        self.assertIn("-m", command)
        self.assertIn("runtime.trace_tail", command)
        self.assertEqual(command[-1], "mira.log")

    def test_trace_tail_reports_missing_log_path(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            code = trace_tail_main([])

        self.assertEqual(code, 2)
        self.assertIn("Usage:", output.getvalue())

    def test_trace_tail_prints_recent_existing_lines_before_following(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "mira.log"
            log_path.write_text("old one\nold two\n", encoding="utf-8")

            def stop_after_backlog(seconds: float) -> None:
                raise KeyboardInterrupt

            output = StringIO()
            with patch("runtime.trace_tail.time.sleep", stop_after_backlog), redirect_stdout(output):
                with self.assertRaises(KeyboardInterrupt):
                    trace_tail_main([str(log_path)])

            rendered = output.getvalue()
            self.assertIn("MIRA Trace", rendered)
            self.assertIn("old one", rendered)
            self.assertIn("old two", rendered)

    def test_trace_tail_colorizes_known_labels_only_for_display(self) -> None:
        rendered = colorize_line("user:\n")

        self.assertIn("\033[", rendered)
        self.assertIn("user:", rendered)
        self.assertEqual(colorize_line("plain body text\n"), "plain body text\n")
        self.assertEqual(colorize_line("D:\\Projects\\mira\\.mira\\_logs\\mira.log\n"), "D:\\Projects\\mira\\.mira\\_logs\\mira.log\n")

    def test_trace_tail_colors_entire_current_block(self) -> None:
        colorizer = TerminalColorizer()

        header = colorizer.colorize("user:\n")
        body = colorizer.colorize("hello from user\n")
        next_header = colorizer.colorize("mira:\n")
        next_body = colorizer.colorize("hello from mira\n")

        self.assertIn("\033[", header)
        self.assertIn("\033[", body)
        self.assertIn("hello from user", body)
        self.assertNotEqual(body.split("hello", 1)[0], next_body.split("hello", 1)[0])
        self.assertIn("mira:", next_header)

    def test_terminal_colorizer_strips_back_to_plain_transcript(self) -> None:
        colorizer = TerminalColorizer()
        plain = "mira:\nhello\nthinking:\nvisible reasoning\n"

        rendered = colorizer.colorize(plain)

        self.assertIn("\033[", rendered)
        self.assertEqual(strip_ansi(rendered), plain)

    def test_trace_tail_thinking_body_is_bright_not_dimmed(self) -> None:
        colorizer = TerminalColorizer()

        header = colorizer.colorize("thinking:\n")
        body = colorizer.colorize("visible reasoning\n")

        self.assertIn("\033[38;2;130;144;154mthinking:", header)
        self.assertIn("\033[38;2;184;194;201m", body)
        self.assertNotIn("\033[2m", body)

    def test_trace_tail_keeps_body_plain_before_first_header(self) -> None:
        colorizer = TerminalColorizer()

        self.assertEqual(colorizer.colorize("plain body text\n"), "plain body text\n")

    def test_trace_tail_maps_mira_bubble_labels_to_colors(self) -> None:
        self.assertTrue(color_for_label("user"))
        self.assertTrue(color_for_label("mira"))
        self.assertTrue(color_for_label("thinking"))
        self.assertTrue(color_for_label("warning"))
        self.assertTrue(color_for_label("error"))
        self.assertTrue(color_for_label("subagent - worker"))
        self.assertTrue(color_for_label("read_file"))
        self.assertEqual(color_for_label("ordinary sentence"), "")

    def test_trace_tail_enable_console_colors_failure_is_non_fatal(self) -> None:
        real_import = builtins.__import__

        def import_without_ctypes(name, *args, **kwargs):
            if name == "ctypes":
                raise ImportError("no ctypes")
            return real_import(name, *args, **kwargs)

        with (
            patch.object(sys, "platform", "win32"),
            patch("builtins.__import__", side_effect=import_without_ctypes),
        ):
            enable_console_colors()


if __name__ == "__main__":
    unittest.main()
