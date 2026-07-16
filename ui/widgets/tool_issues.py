"""Focused repair modal for unavailable project tool files."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from dataclasses import dataclass

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, LoadingIndicator, Static

from agent.resources.tool_failures import ToolLoadFailure, missing_requirements


@dataclass(frozen=True)
class PipInstallResult:
    requirements: list[str]
    returncode: int
    stdout: str
    stderr: str
    launch_error: str = ""

    @property
    def succeeded(self) -> bool:
        return not self.launch_error and self.returncode == 0


class ToolIssuesScreen(ModalScreen[None]):
    """Show all current tool-file failures and repairable imports together."""

    BINDINGS = [Binding("escape", "close", "Close")]

    def __init__(self, failures: list[ToolLoadFailure]) -> None:
        super().__init__()
        self.failures = list(failures)
        self.installing = False
        self.install_details = ""

    def compose(self) -> ComposeResult:
        with Vertical(id="tool-issues-dialog"):
            yield Static("CUSTOM TOOLS UNAVAILABLE", id="tool-issues-title")
            with VerticalScroll(id="tool-issues-scroll"):
                yield Static(self.summary_text(), id="tool-issues-summary")
            yield Static("Packages to install:", id="tool-issues-package-label")
            yield Input(" ".join(missing_requirements(self.failures)), id="tool-issues-packages")
            yield LoadingIndicator(id="tool-issues-loading")
            with Horizontal(id="tool-issues-actions"):
                yield Button("Install All and Reload", id="tool-issues-install", variant="primary")
                yield Button("Close", id="tool-issues-close")

    def on_mount(self) -> None:
        self.query_one("#tool-issues-loading", LoadingIndicator).display = False
        self._sync_controls()

    def update_failures(self, failures: list[ToolLoadFailure]) -> None:
        self.failures = list(failures)
        self.install_details = ""
        if self.is_mounted:
            self.query_one("#tool-issues-summary", Static).update(self.summary_text())
            self.query_one("#tool-issues-packages", Input).value = " ".join(missing_requirements(self.failures))
            self._sync_controls()

    def summary_text(self) -> str:
        repairable = [failure for failure in self.failures if failure.missing_module]
        other = [failure for failure in self.failures if not failure.missing_module]
        sections: list[str] = []
        if repairable:
            noun = "file is" if len(repairable) == 1 else "files are"
            sections.append(f"{len(repairable)} {noun} missing packages:\n")
            sections.extend(failure_text(failure) for failure in repairable)
        if other:
            noun = "file has" if len(other) == 1 else "files have"
            sections.append(f"{len(other)} {noun} another error:\n")
            sections.extend(failure_text(failure) for failure in other)
        sections.append(
            "@tool runs inside MIRA.\n\n"
            "To use the configured project environment instead, use\n"
            "@project_tool and place project-only imports inside the\n"
            "function.\n\n"
            "Example:\n.mira/examples/tools/project_runtime_tool.py"
        )
        if self.install_details:
            sections.append(self.install_details)
        return "\n".join(sections).strip()

    @on(Button.Pressed, "#tool-issues-close")
    def close_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.action_close()

    @on(Button.Pressed, "#tool-issues-install")
    def install_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if self.installing:
            return
        try:
            requirements = parse_requirements(self.query_one("#tool-issues-packages", Input).value)
        except ValueError as error:
            self.install_details = f"Install not started: {error}"
            self.query_one("#tool-issues-summary", Static).update(self.summary_text())
            return
        self.installing = True
        self.install_details = "Installing packages into MIRA's environment..."
        self.query_one("#tool-issues-summary", Static).update(self.summary_text())
        self._sync_controls()
        self.install_requirements(requirements)

    @work(thread=True, exclusive=True, exit_on_error=False, group="tool-issues-install")
    def install_requirements(self, requirements: list[str]) -> None:
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "pip", "install", *requirements],
                capture_output=True,
                text=True,
                shell=False,
                check=False,
            )
            result = PipInstallResult(
                requirements=requirements,
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
        except BaseException as error:
            result = PipInstallResult(requirements, -1, "", "", f"{type(error).__name__}: {error}")
        self.app.call_from_thread(self._install_finished, result)

    async def _install_finished(self, result: PipInstallResult) -> None:
        if result.succeeded:
            await self.app.reload_after_tool_install(self, result)
            return
        self.installing = False
        if result.launch_error:
            details = result.launch_error
        else:
            output = []
            if result.stdout.strip():
                output.append(f"stdout:\n{result.stdout.strip()}")
            if result.stderr.strip():
                output.append(f"stderr:\n{result.stderr.strip()}")
            details = "\n\n".join(output) or "pip failed"
        self.install_details = f"Pip installation failed:\n{details}"
        self.query_one("#tool-issues-summary", Static).update(self.summary_text())
        self._sync_controls()

    def _sync_controls(self) -> None:
        repairable = bool(missing_requirements(self.failures))
        self.query_one("#tool-issues-packages", Input).disabled = self.installing or not repairable
        self.query_one("#tool-issues-install", Button).disabled = self.installing or not repairable
        self.query_one("#tool-issues-close", Button).disabled = self.installing
        self.query_one("#tool-issues-loading", LoadingIndicator).display = self.installing

    def action_close(self) -> None:
        if not self.installing:
            self.dismiss()


def parse_requirements(value: str) -> list[str]:
    try:
        requirements = shlex.split(value, posix=os.name != "nt")
    except ValueError as error:
        raise ValueError(f"invalid package list: {error}") from error
    if not requirements:
        raise ValueError("enter at least one package requirement")
    if os.name == "nt":
        requirements = [strip_matching_quotes(item) for item in requirements]
    forbidden = {"|", "||", "&", "&&", ";", "<", ">", ">>"}
    if any(item in forbidden or "\n" in item or "\r" in item for item in requirements):
        raise ValueError("shell operators are not package requirements")
    if any(item.startswith("-") for item in requirements):
        raise ValueError("pip options are not accepted here")
    return requirements


def strip_matching_quotes(value: str) -> str:
    """Remove the quotes retained by Windows-style shlex tokenization."""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def failure_text(failure: ToolLoadFailure) -> str:
    location = failure.display_path
    if failure.line_number:
        location = f"{location}:{failure.line_number}"
    lines = [f"• {location}"]
    if failure.source_line:
        lines.append(f"  {failure.source_line}")
    lines.append(f"  {failure.exception_type}: {failure.message}")
    return "\n".join(lines) + "\n"
