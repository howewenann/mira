"""Plain-terminal consumer of the shared MIRA Core API."""

from ui.shared.adapter import RendererAdapter


class TerminalFrontend(RendererAdapter):
    """Connect typed Core events and requests to the one-shot renderer."""


__all__ = ["TerminalFrontend"]
