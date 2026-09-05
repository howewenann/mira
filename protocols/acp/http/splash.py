"""Centered MIRA startup presentation for the ACP HTTP server."""

from rich.align import Align
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from config.branding import (
    MIRA_CYAN,
    MIRA_HINT,
    MIRA_LABEL,
    MIRA_TITLE,
    MIRA_VALUE,
    VERSION,
    blocky_wordmark,
)

MAX_PANEL_WIDTH = 78


def http_splash(listen: str, *, terminal_width: int) -> Align:
    """Build a centered server panel sized for the available terminal."""
    panel_width = min(MAX_PANEL_WIDTH, max(20, terminal_width - 2))
    wordmark = blocky_wordmark()
    wordmark_width = max(len(line) for line in wordmark.splitlines())

    # The blocky logo needs room to remain legible. Narrow terminals still get
    # the same MIRA identity without wrapping the ASCII art into fragments.
    logo_text = wordmark if panel_width >= wordmark_width + 8 else "MIRA"
    logo = Text(logo_text, style=MIRA_CYAN, justify="center")
    title = Text(VERSION, style=MIRA_TITLE, justify="center")

    metadata = Table.grid(padding=(0, 2))
    metadata.add_column(style=MIRA_LABEL, no_wrap=True)
    metadata.add_column(style=MIRA_VALUE)
    metadata.add_row("transport", "ACP Streamable HTTP")
    metadata.add_row("endpoint", f"http://{listen}/acp")
    metadata.add_row("access", "loopback only")
    metadata.add_row("status", "ready")

    contents = Group(
        Align.center(logo),
        Text(),
        Align.center(title),
        Text(),
        Align.center(metadata),
        Text(),
        Text("Ctrl+C to stop", style=MIRA_HINT, justify="center"),
    )
    panel = Panel(
        contents,
        width=panel_width,
        padding=(1, 2),
        border_style=MIRA_CYAN,
    )
    return Align.center(panel)


def print_http_splash(listen: str) -> None:
    """Print the centered ready panel after the listener has been observed."""
    console = Console()
    console.print(http_splash(listen, terminal_width=console.width))


__all__ = ["http_splash", "print_http_splash"]
