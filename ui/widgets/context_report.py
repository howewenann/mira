"""Compact modal for inspecting current context composition."""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Grid, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Collapsible, Static

from runtime.context_report import ContextReport, ContextReportRow as ReportRow
from runtime.context_report import current_context_values, estimated_token_text, share_text


class ContextReportRow(Grid):
    """A three-column row whose label cannot displace numeric values."""

    def __init__(self, label: str, value: str, percent: str, *, id: str | None = None,
                 classes: str = "") -> None:
        super().__init__(id=id, classes=f"context-report-row {classes}".strip())
        self.label = label
        self.value = value
        self.percent = percent

    def compose(self) -> ComposeResult:
        yield Static(self.label, classes="context-report-label", markup=False)
        yield Static(self.value, classes="context-report-value", markup=False)
        yield Static(self.percent, classes="context-report-percent", markup=False)


class ContextReportDisclosure(Collapsible):
    """A native disclosure whose aggregate metrics stay in the shared grid."""

    def __init__(
        self,
        row: ReportRow,
        total: int,
        *children: ContextReportRow | Collapsible,
        id: str,
        classes: str,
        header_classes: str,
    ) -> None:
        super().__init__(
            *children,
            title=row.label,
            collapsed=True,
            id=id,
            classes=f"context-report-disclosure {classes}",
        )
        self.metrics = estimated_token_text(row.tokens), share_text(row.tokens, total)
        self.header_classes = header_classes

    def compose(self) -> ComposeResult:
        with Grid(
            classes=f"context-report-row context-report-disclosure-header {self.header_classes}"
        ):
            yield self._title
            for metric, column in zip(self.metrics, ("value", "percent"), strict=True):
                yield Static(
                    metric,
                    classes=f"context-report-{column} context-report-primary",
                    markup=False,
                )
        with self.Contents():
            yield from self._contents_list


class ContextReportTools(ContextReportDisclosure):
    """The top-level Tools disclosure."""

    def __init__(self, row: ReportRow, total: int) -> None:
        super().__init__(
            row,
            total,
            *tool_detail_widgets(row, total),
            id="context-report-row-tools",
            classes="context-report-contributor context-report-tools",
            header_classes="context-report-tools-header",
        )


class ContextReportMCP(ContextReportDisclosure):
    """The nested MCP disclosure."""

    def __init__(self, row: ReportRow, total: int) -> None:
        servers = tuple(
            detail_row(server, total, level=2, prefix="server") for server in row.children
        )
        super().__init__(
            row,
            total,
            *servers,
            id="context-report-tool-mcp",
            classes="context-report-child context-report-mcp",
            header_classes="context-report-mcp-header context-report-level-1",
        )


class ContextReportScreen(ModalScreen[None]):
    """Show live context occupancy and a compact estimated breakdown."""

    AUTO_FOCUS = ""
    BINDINGS = [Binding("escape", "close", "Close")]

    def __init__(self, report: ContextReport) -> None:
        super().__init__()
        self.report = report

    def compose(self) -> ComposeResult:
        used_limit, usage = current_context_values(self.report)
        total = self.report.share_total or 0
        with Vertical(id="context-report-dialog"):
            with Horizontal(classes="context-report-top-row"):
                yield Static("Context Report", id="context-report-title")
                yield Button("x", id="context-report-close", classes="panel-close")
            with VerticalScroll(id="context-report-scroll"):
                yield ContextReportRow("", "Used / Limit", "Usage",
                                       id="context-report-current-header",
                                       classes="context-report-header")
                yield ContextReportRow(
                    "Current context",
                    used_limit,
                    usage,
                    id="context-report-current",
                    classes="context-report-measured",
                )
                yield Static("Estimated contributors", id="context-report-section-title")
                yield ContextReportRow("", "Tokens", "Share",
                                       id="context-report-contributor-header",
                                       classes="context-report-header")
                for row in self.report.rows:
                    if row.label == "Tools":
                        yield ContextReportTools(row, total)
                    else:
                        yield contributor_row(row, total)
                yield Static(
                    "Approximate component sizes. Current context is MIRA's live measurement.",
                    id="context-report-note",
                )

    def on_mount(self) -> None:
        """Open as a passive report without selecting a control."""
        self.call_after_refresh(self._clear_initial_focus)

    def _clear_initial_focus(self) -> None:
        self.set_focus(None)

    @on(Button.Pressed, "#context-report-close")
    def close_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss()

    def action_close(self) -> None:
        self.dismiss()


def contributor_row(row: ReportRow, total: int) -> ContextReportRow:
    return ContextReportRow(
        row.label,
        estimated_token_text(row.tokens),
        share_text(row.tokens, total),
        id=f"context-report-row-{safe_id(row.label)}",
        classes="context-report-contributor context-report-primary",
    )


def tool_detail_widgets(row: ReportRow, total: int) -> tuple[ContextReportRow | Collapsible, ...]:
    widgets: list[ContextReportRow | Collapsible] = []
    for child in row.children:
        if child.label == "MCP":
            widgets.append(ContextReportMCP(child, total))
        else:
            widgets.append(detail_row(child, total, level=1, prefix="tool"))
    return tuple(widgets)


def detail_row(row: ReportRow, total: int, *, level: int, prefix: str) -> ContextReportRow:
    return ContextReportRow(
        row.label,
        estimated_token_text(row.tokens),
        share_text(row.tokens, total),
        id=f"context-report-{prefix}-{safe_id(row.label)}",
        classes=f"context-report-child context-report-level-{level} context-report-estimated",
    )


def safe_id(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "-" for character in value).strip("-")


__all__ = ["ContextReportRow", "ContextReportScreen", "ContextReportTools"]
