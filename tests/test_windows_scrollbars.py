"""Tests for Windows-safe Textual scrollbar rendering."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from rich.color import Color
from rich.segment import Segment
from textual.scrollbar import ScrollBar, ScrollBarRender

from ui.windows_scrollbars import (
    SolidScrollBarRender,
    configure_scrollbars_for_platform,
)

FRACTIONAL_BARS = set(
    ScrollBarRender.VERTICAL_BARS + ScrollBarRender.HORIZONTAL_BARS
) - {" "}


def render_segments(renderer: type[ScrollBarRender]) -> list[Segment]:
    """Render a positioned vertical scrollbar with track, thumb, and page regions."""
    rendered = renderer.render_bar(
        size=12,
        virtual_size=100,
        window_size=25,
        position=30,
        vertical=True,
        back_color=Color.parse("#122023"),
        bar_color=Color.parse("#5bb8b1"),
    )
    return list(rendered.segments)


class WindowsScrollbarTests(unittest.TestCase):
    """Keep unsupported scrollbar glyphs out of the Windows display path."""

    def test_textual_renderer_uses_fractional_blocks(self) -> None:
        emitted = "".join(segment.text for segment in render_segments(ScrollBarRender))

        self.assertTrue(FRACTIONAL_BARS.intersection(emitted))

    def test_solid_renderer_emits_no_fractional_blocks(self) -> None:
        emitted = "".join(segment.text for segment in render_segments(SolidScrollBarRender))

        self.assertFalse(FRACTIONAL_BARS.intersection(emitted))

    def test_solid_renderer_preserves_thumb_and_mouse_actions(self) -> None:
        segments = render_segments(SolidScrollBarRender)
        actions = {
            segment.style.meta.get("@mouse.down")
            for segment in segments
            if segment.style is not None and segment.style.meta
        }
        thumb = [
            segment
            for segment in segments
            if segment.style is not None
            and segment.style.meta
            and segment.style.meta.get("@mouse.down") == "grab"
        ]

        self.assertEqual(actions, {"scroll_up", "grab", "scroll_down"})
        self.assertTrue(thumb)
        self.assertTrue(all(segment.style.reverse for segment in thumb))

    def test_windows_selects_solid_renderer(self) -> None:
        with patch.object(ScrollBar, "renderer", ScrollBarRender):
            configure_scrollbars_for_platform("win32")

            self.assertIs(ScrollBar.renderer, SolidScrollBarRender)

    def test_mira_app_configures_windows_renderer_at_startup(self) -> None:
        from ui.app import MiraApp

        with (
            patch.object(ScrollBar, "renderer", ScrollBarRender),
            patch("ui.app.sys.platform", "win32"),
            patch("ui.app.driver_class_for_platform", return_value=None),
        ):
            MiraApp(config={})

            self.assertIs(ScrollBar.renderer, SolidScrollBarRender)

    def test_non_windows_leaves_renderer_unchanged(self) -> None:
        class ExistingRenderer(ScrollBarRender):
            pass

        with patch.object(ScrollBar, "renderer", ExistingRenderer):
            configure_scrollbars_for_platform("linux")

            self.assertIs(ScrollBar.renderer, ExistingRenderer)


if __name__ == "__main__":
    unittest.main()
