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
    script = (
        "import pathlib,time,sys;"
        "p=pathlib.Path(sys.argv[1]);"
        "print('MIRA Trace');print(str(p));"
        "p.parent.mkdir(parents=True,exist_ok=True);p.touch(exist_ok=True);"
        "f=p.open('r',encoding='utf-8',errors='replace');"
        "f.seek(0,2);"
        "\nwhile True:\n"
        "    line=f.readline()\n"
        "    if line:\n"
        "        print(line,end='',flush=True)\n"
        "    else:\n"
        "        time.sleep(0.25)\n"
    )
    try:
        subprocess.Popen(  # noqa: S603 - intentional local cmd.exe trace window.
            ["cmd.exe", "/c", "start", "MIRA Trace", sys.executable, "-u", "-c", script, str(log_path)],
            close_fds=True,
        )
    except OSError:
        return False
    return True
