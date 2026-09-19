"""Runtime-only grouped MCP lifecycle activity for the chat viewport."""

from __future__ import annotations

from time import monotonic
from typing import Any

from rich.text import Text
from textual.widgets import Static

from ui.shared.terminal.spinners import SPINNER_FRAMES


class MCPActivityCell(Static):
    """Update one lifecycle batch in place without creating chat messages."""

    def __init__(self, snapshot: dict[str, Any]) -> None:
        self.snapshot = snapshot
        self._spinner = 0
        super().__init__(self._build_activity_text(), classes="message mcp-activity")
        self.border_title = _batch_title(snapshot)

    def update_snapshot(self, snapshot: dict[str, Any]) -> None:
        self.snapshot = snapshot
        self.border_title = _batch_title(snapshot)
        self.update(self._build_activity_text())

    def tick(self) -> None:
        if self.snapshot.get("finished"):
            return
        self._spinner = (self._spinner + 1) % len(SPINNER_FRAMES)
        self.update(self._build_activity_text())

    def _build_activity_text(self) -> Text:
        snapshot = self.snapshot
        counts = dict(snapshot.get("counts") or {})
        if snapshot.get("successful"):
            return _success_text(snapshot, counts)

        text = Text()
        kind = str(snapshot.get("kind") or "")
        if kind != "restart":
            usable = int(counts.get("usable") or 0)
            configured = int(counts.get("configured") or 0)
            if snapshot.get("finished"):
                problems = int(counts.get("failed") or 0) + int(counts.get("partial") or 0)
                problems += int(counts.get("approval") or 0)
                symbol = "⚠" if problems else "✓"
                style = "bold yellow" if problems else "bold green"
                text.append(
                    f"{symbol} MCP {_batch_label(snapshot)} · "
                    f"{usable}/{configured} available",
                    style=style,
                )
                if problems:
                    text.append(f" · {problems} {_noun(problems, 'problem')}", style=style)
            else:
                text.append(f"MCP {_activity_heading(snapshot)}", style="bold #78d5cf")
                text.append(f"    {usable} / {configured} available", style="#b8c7cc")

        rows = [dict(row) for row in snapshot.get("servers") or ()]
        for index, row in enumerate(rows):
            if text.plain:
                text.append("\n\n" if index == 0 else "\n")
            elif index:
                text.append("\n")
            _append_row(text, row, snapshot, self._spinner)
        return text


def _append_row(
    text: Text,
    row: dict[str, Any],
    snapshot: dict[str, Any],
    spinner: int,
) -> None:
    name = str(row.get("name") or "server")
    stage = str(row.get("stage") or "")
    status = str(row.get("status") or "")
    if stage == "active":
        frame = SPINNER_FRAMES[spinner % len(SPINNER_FRAMES)]
        text.append(f"{frame} {name}", style="bold #8fb9e8")
        latest = str(row.get("latest_stderr_line") or "").strip()
        if latest:
            text.append(f"    {latest}", style="#d6e2e5")
        else:
            started_at = row.get("started_at") or snapshot.get("started_at") or monotonic()
            elapsed = max(0.0, monotonic() - float(started_at))
            verb = "Restarting" if snapshot.get("kind") == "restart" or status == "Restarting" else "Starting"
            text.append(f"    {verb}… {elapsed:.1f}s", style="#b8c7cc")
        return
    if stage == "waiting":
        text.append(f"· {name}", style="#789097")
        text.append("    Waiting", style="dim")
        return
    if stage == "disabled":
        text.append(f"– {name}", style="dim")
        text.append("    Disabled", style="dim")
        return
    if status == "Available":
        text.append(f"✓ {name}", style="bold green")
        text.append("    Ready", style="#b8c7cc")
        return
    if status == "Partially available":
        text.append(f"! {name}", style="bold yellow")
        text.append("    Ready with issues", style="yellow")
    elif status == "Approval required":
        text.append(f"! {name}", style="bold yellow")
        text.append("    Approval required", style="yellow")
    else:
        text.append(f"x {name}", style="bold red")
        text.append("    Failed", style="red")
    error = str(row.get("error") or "").strip()
    if error:
        text.append(f"\n  {error}", style="#ffc2c2")


def _success_text(snapshot: dict[str, Any], counts: dict[str, Any]) -> Text:
    if snapshot.get("kind") == "restart":
        rows = [dict(row) for row in snapshot.get("servers") or ()]
        row = rows[0] if rows else {}
        name = str(row.get("name") or snapshot.get("active_server") or "MCP server")
        tools = int(row.get("tool_count") or 0)
        prompts = int(row.get("prompt_count") or 0)
        resources = int(row.get("resource_count") or 0)
        return Text.assemble(
            (f"✓ {name} · Ready", "bold green"),
            "\n  ",
            (_capability_summary(tools, prompts, resources), "#b8c7cc"),
        )
    usable = int(counts.get("usable") or 0)
    configured = int(counts.get("configured") or 0)
    tools = int(counts.get("tools") or 0)
    prompts = int(counts.get("prompts") or 0)
    resources = int(counts.get("resources") or 0)
    return Text(
        f"✓ MCP {usable}/{configured} available · "
        f"{_capability_summary(tools, prompts, resources)}",
        style="bold green",
    )


def _capability_summary(tools: int, prompts: int, resources: int) -> str:
    return " · ".join(
        (
            f"{tools} {_noun(tools, 'tool')}",
            f"{prompts} {_noun(prompts, 'prompt')}",
            f"{resources} {_noun(resources, 'resource')}",
        )
    )


def _activity_heading(snapshot: dict[str, Any]) -> str:
    return "RELOAD" if snapshot.get("kind") == "reload" else "STARTUP"


def _batch_label(snapshot: dict[str, Any]) -> str:
    return {
        "startup": "startup",
        "reload": "runtime reload",
        "restart": "restart",
    }.get(str(snapshot.get("kind") or ""), "activity")


def _batch_title(snapshot: dict[str, Any]) -> str:
    return f"mcp {_batch_label(snapshot)}"


def _noun(count: int, singular: str) -> str:
    return singular if count == 1 else singular + "s"


__all__ = ["MCPActivityCell"]
