"""Shared paths for bundled and project resources."""

from __future__ import annotations

from pathlib import Path

DEFAULT_ROUTE = "/mira-defaults"
PROJECT_DIR = ".mira"

MEMORIES_DIR = "memories"
SKILLS_DIR = "skills"
SUBAGENTS_DIR = "subagents"
TOOLS_DIR = "tools"

DEFAULTS_ROOT = Path(__file__).parents[1] / "default_resources"


def default_dir(resource_dir: str) -> Path:
    return DEFAULTS_ROOT / resource_dir


def project_dir(workspace: Path, resource_dir: str) -> Path:
    return workspace / PROJECT_DIR / resource_dir


def default_virtual_dir(resource_dir: str) -> str:
    return f"{DEFAULT_ROUTE}/{resource_dir}"


def project_virtual_dir(resource_dir: str) -> str:
    return f"/{PROJECT_DIR}/{resource_dir}"
