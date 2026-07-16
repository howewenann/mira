"""Sanitized tables for runtime inspection commands."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from rich.console import Group
from rich.table import Table
from rich.text import Text

from config.runtime import RuntimeSnapshot

def runtime_report(snapshot: RuntimeSnapshot) -> Group:
    """Build the model and connection report used by ``/runtime``."""
    sections = [runtime_table(snapshot)]
    if snapshot.warnings:
        sections.extend([Text(""), warnings_text(snapshot)])
    return Group(*sections)


def runtime_table(snapshot: RuntimeSnapshot) -> Table:
    """Build the allowlisted runtime, transport, and launch-option table."""
    table = _detail_table("Runtime")
    table.add_row("Model", snapshot.model_name)
    if snapshot.provider:
        table.add_row("Provider", snapshot.provider)
    table.add_row("Connection", "Direct" if snapshot.direct_effective else "Standard")
    if snapshot.endpoint:
        table.add_row("Endpoint", snapshot.endpoint)
    if snapshot.direct_effective:
        table.add_row("Proxy environment", "Ignored")
        table.add_row("TLS verification", "Disabled")
    table.add_row("-d / --direct", "enabled" if snapshot.direct_requested else "disabled")
    return table


def warnings_text(snapshot: RuntimeSnapshot) -> Text:
    """Render sanitized runtime warnings below the runtime table."""
    lines = ["Warning", *(f"  {warning}" for warning in snapshot.warnings)]
    return Text("\n".join(lines))


def tools_table(tools: Sequence[Mapping[str, str]], *, planning: bool) -> Table:
    """Build current-stage tool metadata with an explicit empty state."""
    mode_name = "planning" if planning else "action"
    table = Table(title=f"Tools ({mode_name})", title_style="bold cyan")
    table.add_column("Tool", style="cyan", no_wrap=True)
    table.add_column("Source", no_wrap=True)
    table.add_column("Replaces", no_wrap=True)
    table.add_column("Path")
    table.add_column("Runtime", no_wrap=True)
    table.add_column("Environment", no_wrap=True)
    table.add_column("Description")
    if not tools:
        table.add_row("none loaded", "-", "-", "-", "-", "-", "-")
        return table

    for tool in tools:
        table.add_row(
            str(tool.get("name") or "-"),
            str(tool.get("source") or "-"),
            str(tool.get("replaces") or "-"),
            str(tool.get("path") or "-"),
            str(tool.get("runtime") or "MIRA"),
            str(tool.get("environment") or "-"),
            str(tool.get("description") or "-"),
        )
    return table


def resources_table(title: str, items: Sequence[Mapping[str, str]]) -> Table:
    """Build one normalized resource section with an explicit empty state."""
    table = Table(title=title, title_style="bold cyan")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Source", no_wrap=True)
    table.add_column("Replaces", no_wrap=True)
    table.add_column("Path")
    if not items:
        table.add_row("none loaded", "-", "-", "-")
        return table

    for item in items:
        table.add_row(
            str(item.get("name") or "-"),
            str(item.get("source") or "-"),
            str(item.get("replaces") or "-"),
            str(item.get("path") or "-"),
        )
    return table


def _detail_table(title: str) -> Table:
    table = Table(title=title, title_style="bold cyan", show_header=False)
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value")
    return table
