"""Default regex grep tool for MIRA."""

from __future__ import annotations

import re
from typing import Literal

from deepagents.backends.utils import format_grep_matches
from langchain.tools import tool

OutputMode = Literal["files_with_matches", "content", "count"]


def get_tools(project_backend: object) -> list[object]:
    """Return tools bound to the project filesystem backend."""

    @tool(
        "grep",
        description=(
            "Search files in the project workspace using a regular expression. "
            "Use this when literal grep is too limited."
        ),
    )
    def grep(
        pattern: str,
        path: str = "/",
        glob: str | None = None,
        output_mode: OutputMode = "files_with_matches",
    ) -> str:
        """Search project files with a regular expression."""
        if path.startswith("/mira-defaults"):
            return "Error: regex grep only searches the project workspace, not /mira-defaults."

        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return f"Invalid regex pattern: {exc}"

        paths_or_error = searchable_paths(project_backend, path, glob)
        if isinstance(paths_or_error, str):
            return paths_or_error

        matches = []
        for file_path, content in downloaded_text(project_backend, paths_or_error):
            for line_number, line in enumerate(content.splitlines(), 1):
                if regex.search(line):
                    matches.append({"path": file_path, "line": line_number, "text": line})

        return format_grep_matches(matches, output_mode)

    return [grep]


def searchable_paths(project_backend: object, path: str, glob: str | None) -> list[str] | str:
    """Find project files that should be searched."""
    try:
        result = project_backend.glob(glob or "**/*", path=path)
    except ValueError as exc:
        return f"Error searching path '{path}': {exc}"

    if result.error:
        return f"Error searching path '{path}': {result.error}"

    paths = [item["path"] for item in result.matches or []]
    if paths or glob or path in {"", "/"}:
        return paths

    response = project_backend.download_files([path])[0]
    if response.error is None:
        return [path]
    if response.error == "is_directory":
        return []
    return f"Error searching path '{path}': {response.error}"


def downloaded_text(project_backend: object, paths: list[str]) -> list[tuple[str, str]]:
    """Download and decode text files, skipping unreadable or binary files."""
    responses = project_backend.download_files(paths)
    texts = []
    for response in responses:
        if response.error or response.content is None:
            continue
        try:
            texts.append((response.path, response.content.decode("utf-8")))
        except UnicodeDecodeError:
            continue
    return texts
