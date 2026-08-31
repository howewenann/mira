"""Textual consumer of the supported MIRA API."""

from ui.shared.adapter import RendererAdapter


class TextualFrontend(RendererAdapter):
    """Connect typed Core events and requests to the Textual application."""


__all__ = ["TextualFrontend"]
