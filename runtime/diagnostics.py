"""Bounded diagnostics logging and optional trace window support."""

from __future__ import annotations

import logging
import subprocess
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOGGER_NAME = "mira"
LOG_MAX_BYTES = 2 * 1024 * 1024
LOG_BACKUP_COUNT = 3


def setup_diagnostics_logging(workspace: Path) -> Path:
    """Configure a bounded diagnostics log for optional live tracing."""
    log_path = workspace.expanduser().resolve() / ".mira" / "_logs" / "mira.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = get_diagnostics_logger()
    resolved = str(log_path.resolve())
    for handler in logger.handlers:
        if isinstance(handler, RotatingFileHandler) and str(Path(handler.baseFilename).resolve()) == resolved:
            return log_path

    handler = RotatingFileHandler(
        log_path,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.info("diagnostics logging started")
    return log_path


def get_diagnostics_logger() -> logging.Logger:
    """Return MIRA's diagnostics logger."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not any(isinstance(handler, logging.NullHandler) for handler in logger.handlers):
        logger.addHandler(logging.NullHandler())
    return logger


def open_trace_window(log_path: Path) -> bool:
    """Open a separate cmd.exe window that tails the diagnostics log."""
    if sys.platform != "win32":
        return False
    try:
        subprocess.Popen(  # noqa: S603 - intentional local cmd.exe trace window.
            ["cmd.exe", "/k", sys.executable, "-u", "-m", "runtime.trace_tail", str(log_path)],
            close_fds=True,
            creationflags=subprocess.CREATE_NEW_CONSOLE,
        )
    except OSError:
        return False
    return True
