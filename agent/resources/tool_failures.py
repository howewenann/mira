"""Structured failures from optional project tool files."""

from __future__ import annotations

import linecache
import traceback
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

ToolFailureFingerprint = tuple[str, str, str, str]


@dataclass(frozen=True)
class ToolLoadFailure:
    """One project tool file that could not be imported or inspected."""

    identifier: str
    source_path: Path
    display_path: str
    exception_type: str
    message: str
    line_number: int | None
    source_line: str
    traceback_text: str
    missing_module: str
    suggested_requirement: str


def tool_failure_fingerprint(
    failure: ToolLoadFailure,
    workspace: Path,
) -> ToolFailureFingerprint:
    """Return stable fields for an explicit reload's before-and-after comparison."""
    try:
        source = (
            failure.source_path.expanduser()
            .resolve()
            .relative_to(workspace.expanduser().resolve())
            .as_posix()
        )
    except (OSError, RuntimeError, ValueError):
        source = failure.display_path
    return (
        source,
        failure.exception_type,
        failure.missing_module,
        failure.message,
    )


def tool_load_failure(path: Path, workspace: Path, error: BaseException) -> ToolLoadFailure:
    """Capture a stable, source-focused description of a loading exception."""
    source_path = path.expanduser().resolve()
    workspace = workspace.expanduser().resolve()
    location_path, line_number, source_line = relevant_source_location(source_path, workspace, error)
    try:
        display_path = location_path.relative_to(workspace).as_posix()
    except ValueError:
        display_path = str(location_path)
    exception_type = type(error).__name__
    message = str(error)
    missing_module = ""
    if isinstance(error, ModuleNotFoundError) and isinstance(error.name, str):
        missing_module = error.name.split(".", 1)[0]
    fingerprint = "\0".join(
        (str(source_path), exception_type, message, str(line_number or ""), source_line)
    )
    return ToolLoadFailure(
        identifier=sha256(fingerprint.encode("utf-8")).hexdigest()[:20],
        source_path=source_path,
        display_path=display_path,
        exception_type=exception_type,
        message=message,
        line_number=line_number,
        source_line=source_line,
        traceback_text="".join(traceback.format_exception(error)),
        missing_module=missing_module,
        suggested_requirement=missing_module,
    )


def relevant_source_location(
    source_path: Path,
    workspace: Path,
    error: BaseException,
) -> tuple[Path, int | None, str]:
    """Prefer the deepest traceback frame that points into the workspace."""
    selected: tuple[Path, int] | None = None
    current = error.__traceback__
    while current is not None:
        candidate = Path(current.tb_frame.f_code.co_filename).expanduser().resolve()
        try:
            candidate.relative_to(workspace)
        except ValueError:
            pass
        else:
            selected = (candidate, current.tb_lineno)
        current = current.tb_next

    if selected is not None:
        candidate, number = selected
        return candidate, number, linecache.getline(str(candidate), number).strip()

    if isinstance(error, SyntaxError):
        number = error.lineno
        return source_path, number, str(error.text or "").strip()
    return source_path, None, ""


def missing_requirements(failures: list[ToolLoadFailure]) -> list[str]:
    """Return unique initial pip requirements in stable file order."""
    requirements: list[str] = []
    for failure in failures:
        requirement = failure.suggested_requirement
        if requirement and requirement not in requirements:
            requirements.append(requirement)
    return requirements


def one_shot_warning(failures: list[ToolLoadFailure]) -> str:
    """Return one grouped terminal warning for optional project tool files."""
    if not failures:
        return ""
    count = len(failures)
    lines = [f"Warning: {count} project tool file{'s' if count != 1 else ''} could not be loaded.", ""]
    for failure in failures:
        location = failure.display_path
        if failure.line_number:
            location = f"{location}:{failure.line_number}"
        lines.extend((location, f"  {failure.exception_type}: {failure.message}", ""))
    requirements = missing_requirements(failures)
    if requirements:
        lines.extend((f"Missing modules: {' '.join(requirements)}", ""))
    lines.extend(
        (
            "Normal @tool dependencies run in MIRA's environment.",
            "",
            "To use the configured project environment, see:",
            ".mira/examples/tools/project_runtime_tool.py",
        )
    )
    return "\n".join(lines)
