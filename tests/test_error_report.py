"""Tests for automatic error reports and diagnostics logging."""

from __future__ import annotations

import logging
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from io import StringIO
from contextlib import redirect_stdout

from logging.handlers import RotatingFileHandler

from runtime.diagnostics import get_diagnostics_logger, open_trace_window, setup_diagnostics_logging
from runtime.error_report import error_report_path, write_error_report
from runtime.trace_tail import main as trace_tail_main


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


if __name__ == "__main__":
    unittest.main()
