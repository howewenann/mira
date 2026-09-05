"""MIRA-branded startup presentation for the ACP HTTP server."""

from rich.console import Console
from rich.text import Text

from config.branding import MIRA_HINT, append_label, branded_header


def http_splash_text(listen: str) -> Text:
    """Build server-only startup details for one validated listen address."""
    text = branded_header()
    append_label(text, "transport", "ACP Streamable HTTP")
    append_label(text, "endpoint", f"http://{listen}/acp")
    append_label(text, "access", "loopback only")
    append_label(text, "status", "ready")
    text.append("\nCtrl+C to stop", style=MIRA_HINT)
    return text


def print_http_splash(listen: str) -> None:
    """Print the ready banner after the HTTP listener has been observed."""
    Console().print(http_splash_text(listen), highlight=False)


__all__ = ["http_splash_text", "print_http_splash"]
