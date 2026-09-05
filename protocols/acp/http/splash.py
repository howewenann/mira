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
PANEL_SPACING_ROWS = 3


def http_splash(listen: str, *, terminal_width: int) -> Align:
    """Build a centered server panel sized for the available terminal."""
    panel_width = min(MAX_PANEL_WIDTH, max(20, terminal_width - 2))
    wordmark = blocky_wordmark()
    wordmark_width = max(len(line) for line in wordmark.splitlines())

    # The blocky logo needs room to remain legible. Narrow terminals still get
    # the same MIRA identity without wrapping the ASCII art into fragments.
    use_wordmark = panel_width >= wordmark_width + 8
    logo_text = wordmark if use_wordmark else "MIRA"
    logo = Text(logo_text, style=MIRA_CYAN, no_wrap=True)
    title = Text(VERSION, style=MIRA_TITLE, justify="center")

    metadata = Table.grid(padding=(0, 2))
    metadata.add_column(style=MIRA_LABEL, no_wrap=True)
    metadata.add_column(style=MIRA_VALUE)
    metadata.add_row("transport", "ACP Streamable HTTP")
    metadata.add_row("endpoint", f"http://{listen}/acp")
    metadata.add_row("access", "loopback only")
    metadata.add_row("status", "ready")

    contents = Group(
        Align.center(logo, pad=False),
        Text(),
        Align.center(title, pad=False),
        Text(),
        Align.center(metadata, pad=False),
        Text(),
        Text("Ctrl+C to stop", style=MIRA_HINT, justify="center"),
    )
    panel = Panel(
        contents,
        height=20 if use_wordmark else 17,
        width=panel_width,
        padding=(1, 2),
        border_style=MIRA_CYAN,
    )
    # Padding an Align to the terminal width leaves trailing spaces on every
    # rendered row. Windows terminals can reflow those rows after a focus or
    # font-metric change, making an already-printed panel appear to grow.
    return Align.center(panel, pad=False)


def print_http_splash(listen: str) -> None:
    """Print the centered ready panel after the listener has been observed."""
    # Hypercorn's startup logger also writes to stderr. Sharing the stream keeps
    # the panel and its following native INFO line in deterministic order.
    console = Console(stderr=True)
    for _ in range(PANEL_SPACING_ROWS):
        console.print()
    console.print(http_splash(listen, terminal_width=console.width))
    for _ in range(PANEL_SPACING_ROWS):
        console.print()


__all__ = ["http_splash", "print_http_splash"]
