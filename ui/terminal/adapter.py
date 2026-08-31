"""Plain-terminal consumer of the supported MIRA API."""

from ui.shared.adapter import RendererAdapter


class TerminalFrontend(RendererAdapter):
    """Connect typed Core events and requests to the one-shot renderer."""


__all__ = ["TerminalFrontend"]
