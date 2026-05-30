"""Environment-based configuration loading."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from config.llm import load_llm_config


def _int_env(name: str, default: int) -> int:
    """Read an integer environment variable, falling back on invalid input."""
    value = os.getenv(name)
    if not value:
        return default

    try:
        return int(value)
    except ValueError:
        return default


def load_config(workspace: Path) -> dict[str, Any]:
    """Load all runtime configuration from the environment and defaults."""
    dotenv_path = workspace / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path)
    else:
        load_dotenv()

    return {
        "workspace": str(workspace),
        **load_llm_config(os.environ),
        "tool_output_chars": _int_env("MIRA_TOOL_OUTPUT_CHARS", 240),
        "session_dir": os.getenv("MIRA_SESSION_DIR", str(workspace / ".mira" / "sessions")),
    }
