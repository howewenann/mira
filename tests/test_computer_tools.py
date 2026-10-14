"""Computer tools use mocked desktop operations; no test touches the real screen."""

from __future__ import annotations

import base64
import sys
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call as mock_call, patch

from PIL import Image, ImageDraw
from deepagents.middleware.unsupported_content import UnsupportedContentMiddleware
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from agent.execution import MiraContext, create_mira_context
from agent.factory import _write_interrupts, effective_excluded_tools
from agent.planning.policy import PLAN_DISABLED_TOOLS
from agent.middleware.model_compatibility import ModelCompatibilityMiddleware
from agent.resources import build_resources
from agent.resources.defaults.tools.computer import click, click_cell, press_key, screenshot, scroll, type_text
from config.settings import DEFAULT_SETTINGS, set_tool_always_allow, tool_always_allow
from core.execution.streams.tools import tool_output_text


def call(tool, args: dict, call_id: str = "computer-call", context=None) -> ToolMessage:
    return tool.func(**args, runtime=SimpleNamespace(tool_call_id=call_id, context=context))


class ComputerToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.desktop = SimpleNamespace(
            screenshot=Mock(return_value=Image.new("RGB", (80, 60), "red")),
            size=Mock(return_value=(80, 60)),
            click=Mock(),
            write=Mock(),
            press=Mock(),
            hotkey=Mock(),
            scroll=Mock(),
            KEYBOARD_KEYS=("ctrl", "c", "enter"),
            FAILSAFE=True,
        )
        self.module_patch = patch.dict(sys.modules, {"pyautogui": self.desktop})
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)

    def test_screenshot_is_in_memory_and_uses_deepagents_image_block(self) -> None:
        with patch("pathlib.Path.write_bytes", side_effect=AssertionError("disk write")):
            with patch("pathlib.Path.open", side_effect=AssertionError("disk open")):
                with patch("pathlib.Path.mkdir", side_effect=AssertionError("directory creation")):
                    result = call(screenshot, {})

        self.assertEqual(result.status, "success")
        self.assertEqual(result.tool_call_id, "computer-call")
        self.assertIn("80x60", result.content_blocks[0]["text"])
        image_block = result.content_blocks[1]
        self.assertEqual(image_block["type"], "image")
        self.assertEqual(image_block["mime_type"], "image/png")
        image = Image.open(BytesIO(base64.b64decode(image_block["base64"])))
        self.assertGreater(image.width, 80)
        self.assertGreater(image.height, 60)
        self.assertIn(f"Annotated image: {image.width}x{image.height} pixels", result.content_blocks[0]["text"])
        self.assertIn(f"image ({image.width - 80}, {image.height - 60})", result.content_blocks[0]["text"])
        self.assertTrue(self.desktop.FAILSAFE)
        self.assertEqual(tool_output_text(result), result.content_blocks[0]["text"] + "\n[image attached]")
        self.assertNotIn(image_block["base64"], tool_output_text(result))
        self.assertEqual(screenshot.args["save_debug"]["default"], False)

    def test_debug_screenshot_saves_exact_cropped_png_with_unique_names(self) -> None:
        screen = Image.new("RGB", (640, 480), "white")
        self.desktop.size.return_value = screen.size
        self.desktop.screenshot.side_effect = lambda *, region: screen.crop(
            (region[0], region[1], region[0] + region[2], region[1] + region[3])
        )
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            context = SimpleNamespace(workspace=workspace)
            debug_dir = workspace / ".mira" / "_debug" / "screenshots"
            args = {"region": [73, 127, 210, 190], "save_debug": True}
            first = call(screenshot, args, context=context)
            first_files = list(debug_dir.glob("*.png"))
            self.assertEqual(len(first_files), 1)
            second = call(screenshot, {**args, "grid": False}, context=context)
            second_files = list(debug_dir.glob("*.png"))
            self.assertEqual(len(second_files), 2)

            newest = next(path for path in second_files if path != first_files[0])
            for result, saved_path in ((first, first_files[0]), (second, newest)):
                self.assertEqual(result.status, "success")
                self.assertTrue(saved_path.is_absolute())
                self.assertRegex(saved_path.name, r"^screenshot-\d{8}T\d{12}Z-[0-9a-f]{32}\.png$")
                self.assertIn(str(saved_path), result.content_blocks[0]["text"])
                self.assertIn(str(saved_path), tool_output_text(result))
                png_bytes = base64.b64decode(result.content_blocks[1]["base64"])
                self.assertEqual(saved_path.read_bytes(), png_bytes)
                image = Image.open(BytesIO(png_bytes))
                if result is first:
                    ruler_left, ruler_top = image.width - 210, image.height - 190
                    self.assertGreater(ruler_left, 0)
                    self.assertGreater(ruler_top, 0)
                    self.assertNotEqual(image.getpixel((ruler_left + 50, ruler_top + 38)), (255, 255, 255))
                else:
                    self.assertEqual(image.size, (210, 190))
                    self.assertEqual(image.getpixel((27, 38)), (255, 255, 255))
            self.desktop.screenshot.assert_has_calls([mock_call(region=(73, 127, 210, 190))] * 2)
        self.desktop.click.assert_not_called()

    def test_debug_screenshot_requires_workspace_context(self) -> None:
        result = call(screenshot, {"save_debug": True})
        self.assertEqual(result.status, "error")
        self.assertIn("requires a MIRA workspace", result.text)

    def test_screenshot_grid_uses_original_pixel_positions_and_can_be_disabled(self) -> None:
        original = Image.new("RGB", (450, 450), "white")
        original.putpixel((237, 237), (7, 111, 203))
        self.desktop.screenshot.side_effect = original.copy
        self.desktop.size.return_value = original.size
        labels = []
        draw_text = ImageDraw.ImageDraw.text

        def record_text(draw, xy, value, *args, **kwargs):
            bbox = draw.textbbox(xy, value, font=kwargs["font"], stroke_width=kwargs["stroke_width"])
            labels.append((value, xy, bbox, kwargs["fill"], kwargs["stroke_fill"]))
            return draw_text(draw, xy, value, *args, **kwargs)

        plain = call(screenshot, {"grid": False})
        self.assertEqual(plain.content_blocks[1]["type"], "image")
        plain_image = Image.open(BytesIO(base64.b64decode(plain.content_blocks[1]["base64"])))
        self.assertEqual(plain_image.size, (450, 450))
        self.assertEqual(plain_image.getpixel((100, 237)), (255, 255, 255))
        self.assertEqual(plain_image.getpixel((237, 237)), (7, 111, 203))
        expected_labels = ["(0,0)"]
        expected_labels += [f"X:{x}" for x in (100, 200, 300, 400)]
        expected_labels += [f"Y:{y}" for y in (100, 200, 300, 400)]
        for spacing in (25, 50, 100):
            with self.subTest(spacing=spacing):
                labels.clear()
                args = {"grid_mode": "coordinates"}
                if spacing != 100:
                    args["grid_spacing"] = spacing
                with patch.object(ImageDraw.ImageDraw, "text", record_text):
                    annotated = call(screenshot, args)

                self.assertEqual(annotated.status, plain.status)
                self.assertIn(plain.content_blocks[0]["text"], annotated.content_blocks[0]["text"])
                self.assertEqual(annotated.content_blocks[1]["type"], "image")
                self.assertEqual(annotated.content_blocks[1]["mime_type"], "image/png")
                grid_image = Image.open(BytesIO(base64.b64decode(annotated.content_blocks[1]["base64"])))
                ruler_left, ruler_top = grid_image.width - original.width, grid_image.height - original.height
                self.assertGreater(ruler_left, 0)
                self.assertGreater(ruler_top, 0)
                self.assertIn(f"image ({ruler_left}, {ruler_top})", annotated.content_blocks[0]["text"])
                self.assertEqual(grid_image.getpixel((ruler_left + 237, ruler_top + 237)), (7, 111, 203))
                for coordinate in range(spacing, 450, spacing):
                    self.assertNotEqual(grid_image.getpixel((ruler_left + coordinate, ruler_top + 237)), (255, 255, 255))
                    self.assertNotEqual(grid_image.getpixel((ruler_left + 237, ruler_top + coordinate)), (255, 255, 255))
                    self.assertEqual(grid_image.getpixel((ruler_left + coordinate + 1, ruler_top + 237)), (255, 255, 255))
                self.assertEqual(grid_image.getpixel((ruler_left + 100, ruler_top + 237)), grid_image.getpixel((ruler_left + 200, ruler_top + 237)))
                self.assertEqual(grid_image.getpixel((ruler_left + 237, ruler_top + 100)), grid_image.getpixel((ruler_left + 237, ruler_top + 200)))
                if spacing < 100:
                    self.assertLess(grid_image.getpixel((ruler_left + 100, ruler_top + 237))[0], grid_image.getpixel((ruler_left + spacing, ruler_top + 237))[0])
                    self.assertLess(grid_image.getpixel((ruler_left + 237, ruler_top + 100))[0], grid_image.getpixel((ruler_left + 237, ruler_top + spacing))[0])
                else:
                    self.assertEqual(grid_image.getpixel((ruler_left + 50, ruler_top + 237)), (255, 255, 255))
                if spacing > 25:
                    self.assertEqual(grid_image.getpixel((ruler_left + 25, ruler_top + 237)), (255, 255, 255))
                self.assertEqual([value for value, _, _, _, _ in labels], expected_labels)
                for x, (_, (label_x, label_y), bbox, _, _) in zip((100, 200, 300, 400), labels[1:5]):
                    self.assertTrue(ruler_left + x <= label_x <= ruler_left + x + 4)
                    self.assertEqual(label_y, 4)
                    self.assertLessEqual(bbox[3], ruler_top)
                for y, (_, xy, bbox, _, _) in zip((100, 200, 300, 400), labels[5:]):
                    self.assertEqual(xy, (4, ruler_top + y + 4))
                    self.assertLessEqual(bbox[2], ruler_left)
                self.assertLessEqual(labels[0][2][2], ruler_left)
                self.assertLessEqual(labels[0][2][3], ruler_top)
                self.assertTrue(all(fill == "white" and stroke == "black" for _, _, _, fill, stroke in labels))
        self.assertEqual(screenshot.args["grid"]["default"], True)
        self.assertEqual(screenshot.args["grid_mode"]["default"], "categorical")
        self.assertIsNone(screenshot.args["grid_spacing"]["default"])
        self.assertIsNone(screenshot.args["region"]["default"])
        self.desktop.click.assert_not_called()

    def test_categorical_full_screen_grid_centres_labels_outside_capture(self) -> None:
        self.desktop.size.return_value = (300, 200)
        self.desktop.screenshot.return_value = Image.new("RGB", (300, 200), "white")
        labels = []
        draw_text = ImageDraw.ImageDraw.text

        def record_text(draw, xy, value, *args, **kwargs):
            bbox = draw.textbbox(xy, value, font=kwargs["font"], stroke_width=kwargs["stroke_width"])
            labels.append((value, bbox))
            return draw_text(draw, xy, value, *args, **kwargs)

        with patch.object(ImageDraw.ImageDraw, "text", record_text):
            result = call(screenshot, {})

        self.assertEqual(result.status, "success")
        self.desktop.screenshot.assert_called_once_with()
        image = Image.open(BytesIO(base64.b64decode(result.content_blocks[1]["base64"])))
        ruler_left, ruler_top = image.width - 300, image.height - 200
        self.assertEqual([value for value, _ in labels], ["A", "B", "C", "1", "2"])
        for (_, bbox), centre in zip(labels[:3], (50, 150, 250)):
            self.assertAlmostEqual((bbox[0] + bbox[2]) / 2, ruler_left + centre, delta=2)
            self.assertLessEqual(bbox[3], ruler_top)
        for (_, bbox), centre in zip(labels[3:], (50, 150)):
            self.assertAlmostEqual((bbox[1] + bbox[3]) / 2, ruler_top + centre, delta=2)
            self.assertLessEqual(bbox[2], ruler_left)
        self.assertNotEqual(image.getpixel((ruler_left + 100, ruler_top + 37)), (255, 255, 255))
        self.assertEqual(image.getpixel((ruler_left + 99, ruler_top + 37)), (255, 255, 255))
        self.assertIn("Categorical grid: 3 columns and 2 rows of 100-pixel cells", result.content_blocks[0]["text"])

    def test_categorical_crop_labels_are_relative_to_capture(self) -> None:
        self.desktop.size.return_value = (2560, 1440)
        self.desktop.screenshot.side_effect = lambda *, region: Image.new("RGB", (region[2], region[3]), "white")
        labels = []
        draw_text = ImageDraw.ImageDraw.text

        def record_text(draw, xy, value, *args, **kwargs):
            labels.append(value)
            return draw_text(draw, xy, value, *args, **kwargs)

        with patch.object(ImageDraw.ImageDraw, "text", record_text):
            reference = call(screenshot, {"region": [2200, 0, 300, 200], "grid_spacing": 50})
        self.assertEqual(reference.status, "success")
        self.assertEqual(labels, ["A", "B", "C", "D", "E", "F", "1", "2", "3", "4"])
        self.assertIn("origin (2200, 0), size 300x200", reference.content_blocks[0]["text"])
        self.assertIn("Categorical grid: 6 columns and 4 rows", reference.content_blocks[0]["text"])
        self.desktop.screenshot.assert_called_once_with(region=(2200, 0, 300, 200))

        label_bounds = []

        def record_shifted_text(draw, xy, value, *args, **kwargs):
            label_bounds.append((value, draw.textbbox(xy, value, font=kwargs["font"], stroke_width=kwargs["stroke_width"])))
            return draw_text(draw, xy, value, *args, **kwargs)

        with patch.object(ImageDraw.ImageDraw, "text", record_shifted_text):
            shifted = call(screenshot, {"region": [2203, 7, 125, 75], "grid_spacing": 50})
        self.assertEqual(shifted.status, "success")
        self.assertEqual([value for value, _ in label_bounds], ["A", "B", "C", "1", "2"])
        image = Image.open(BytesIO(base64.b64decode(shifted.content_blocks[1]["base64"])))
        ruler_left, ruler_top = image.width - 125, image.height - 75
        self.assertAlmostEqual(sum(label_bounds[2][1][:3:2]) / 2, ruler_left + 112.5, delta=2)
        self.assertAlmostEqual(sum(label_bounds[4][1][1::2]) / 2, ruler_top + 62.5, delta=2)
        self.assertNotEqual(image.getpixel((ruler_left + 50, ruler_top + 37)), (255, 255, 255))
        self.assertEqual(image.getpixel((ruler_left + 47, ruler_top + 37)), (255, 255, 255))
        self.assertNotEqual(image.getpixel((ruler_left + 37, ruler_top + 50)), (255, 255, 255))
        self.assertEqual(image.getpixel((ruler_left + 37, ruler_top + 43)), (255, 255, 255))

    def test_cropped_screenshot_uses_absolute_grid_and_in_memory_image(self) -> None:
        screen = Image.new("RGB", (640, 480), "white")
        screen.putpixel((111, 165), (7, 111, 203))
        self.desktop.size.return_value = screen.size
        self.desktop.screenshot.side_effect = lambda *, region: screen.crop(
            (region[0], region[1], region[0] + region[2], region[1] + region[3])
        )
        labels = []
        draw_text = ImageDraw.ImageDraw.text

        def record_text(draw, xy, value, *args, **kwargs):
            bbox = draw.textbbox(xy, value, font=kwargs["font"], stroke_width=kwargs["stroke_width"])
            labels.append((value, xy, bbox))
            return draw_text(draw, xy, value, *args, **kwargs)

        with patch("pathlib.Path.write_bytes", side_effect=AssertionError("disk write")):
            with patch.object(ImageDraw.ImageDraw, "text", record_text):
                result = call(screenshot, {"region": [73, 127, 210, 190], "grid_mode": "coordinates"})

        self.assertEqual(result.status, "success")
        self.desktop.screenshot.assert_called_once_with(region=(73, 127, 210, 190))
        self.assertIn("origin (73, 127), size 210x190", result.content_blocks[0]["text"])
        self.assertIn("Original screen: 640x480", result.content_blocks[0]["text"])
        self.assertEqual(result.content_blocks[1]["type"], "image")
        self.assertEqual(result.content_blocks[1]["mime_type"], "image/png")
        image = Image.open(BytesIO(base64.b64decode(result.content_blocks[1]["base64"])))
        ruler_left, ruler_top = image.width - 210, image.height - 190
        self.assertGreater(ruler_left, 0)
        self.assertGreater(ruler_top, 0)
        self.assertIn(f"Annotated image: {image.width}x{image.height} pixels", result.content_blocks[0]["text"])
        self.assertEqual(image.getpixel((ruler_left + 38, ruler_top + 38)), (7, 111, 203))
        self.assertEqual(image.getpixel((ruler_left, ruler_top + 38)), (255, 255, 255))
        self.assertNotEqual(image.getpixel((ruler_left + 27, ruler_top + 38)), (255, 255, 255))  # screen x=100
        self.assertNotEqual(image.getpixel((ruler_left + 77, ruler_top + 38)), (255, 255, 255))  # screen x=150
        self.assertLess(image.getpixel((ruler_left + 27, ruler_top + 38))[0], image.getpixel((ruler_left + 77, ruler_top + 38))[0])
        self.assertNotEqual(image.getpixel((ruler_left + 38, ruler_top + 73)), (255, 255, 255))  # screen y=200
        self.assertNotEqual(image.getpixel((ruler_left + 38, ruler_top + 23)), (255, 255, 255))  # screen y=150
        self.assertLess(image.getpixel((ruler_left + 38, ruler_top + 73))[0], image.getpixel((ruler_left + 38, ruler_top + 23))[0])
        self.assertEqual([value for value, _, _ in labels], ["X:100", "X:200", "Y:200", "Y:300"])
        self.assertEqual(labels[0][1], (ruler_left + 31, 4))
        self.assertEqual(labels[2][1], (4, ruler_top + 77))
        self.assertTrue(all(bbox[3] <= ruler_top for _, _, bbox in labels[:2]))
        self.assertTrue(all(bbox[2] <= ruler_left for _, _, bbox in labels[2:]))
        self.assertNotIn(result.content_blocks[1]["base64"], tool_output_text(result))
        self.desktop.click.assert_not_called()

    def test_cropped_screenshot_validates_screen_edges_and_plain_view(self) -> None:
        screen = Image.new("RGB", (640, 480), "white")
        self.desktop.size.return_value = screen.size
        self.desktop.screenshot.side_effect = lambda *, region: screen.crop(
            (region[0], region[1], region[0] + region[2], region[1] + region[3])
        )
        edge = call(screenshot, {"region": [590, 430, 50, 50], "grid": False})
        self.assertEqual(edge.status, "success")
        image = Image.open(BytesIO(base64.b64decode(edge.content_blocks[1]["base64"])))
        self.assertEqual(image.size, (50, 50))
        self.assertEqual(image.getpixel((10, 20)), (255, 255, 255))
        corner = call(screenshot, {"region": [639, 479, 1, 1]})
        self.assertEqual(corner.status, "success")
        corner_image = Image.open(BytesIO(base64.b64decode(corner.content_blocks[1]["base64"])))
        self.assertGreater(corner_image.width, 1)
        self.assertGreater(corner_image.height, 1)
        self.desktop.screenshot.reset_mock()
        for region in ([-1, 0, 10, 10], [0, -1, 10, 10], [0, 0, 0, 10],
                       [0, 0, 10, 0], [630, 0, 11, 10], [0, 470, 10, 11],
                       [640, 0, 1, 1], [0, 0, 10], [True, 0, 10, 10]):
            with self.subTest(region=region):
                self.assertEqual(call(screenshot, {"region": region}).status, "error")
        self.desktop.screenshot.assert_not_called()
        self.desktop.click.assert_not_called()

    def test_ruler_labels_fit_outside_high_coordinate_crop(self) -> None:
        self.desktop.size.return_value = (12000, 12000)
        self.desktop.screenshot.return_value = Image.new("RGB", (120, 120), "white")
        labels = []
        draw_text = ImageDraw.ImageDraw.text

        def record_text(draw, xy, value, *args, **kwargs):
            labels.append((value, draw.textbbox(xy, value, font=kwargs["font"], stroke_width=kwargs["stroke_width"])))
            return draw_text(draw, xy, value, *args, **kwargs)

        with patch.object(ImageDraw.ImageDraw, "text", record_text):
            result = call(screenshot, {"region": [11880, 11880, 120, 120], "grid_mode": "coordinates"})

        self.assertEqual(result.status, "success")
        image = Image.open(BytesIO(base64.b64decode(result.content_blocks[1]["base64"])))
        ruler_left, ruler_top = image.width - 120, image.height - 120
        self.assertEqual([value for value, _ in labels], ["X:11900", "Y:11900"])
        self.assertLessEqual(labels[0][1][3], ruler_top)
        self.assertLessEqual(labels[1][1][2], ruler_left)
        self.assertEqual(image.getpixel((ruler_left + 1, ruler_top + 1)), (255, 255, 255))

    def test_cropped_screenshot_rejects_capture_geometry_mismatch(self) -> None:
        self.desktop.size.return_value = (640, 480)
        self.desktop.screenshot.return_value = Image.new("RGB", (10, 10))
        result = call(screenshot, {"region": [600, 440, 40, 40]})
        self.assertEqual(result.status, "error")
        self.assertIn("DPI scaling", result.text)

    def test_langgraph_tool_node_keeps_image_in_tool_message(self) -> None:
        self.desktop.screenshot.return_value = Image.new("RGB", (20, 15), "red")
        graph = StateGraph(MessagesState, context_schema=MiraContext)
        graph.add_node("tools", ToolNode([screenshot]))
        graph.add_edge(START, "tools")
        graph.add_edge("tools", END)
        request = AIMessage(content="", tool_calls=[{
            "name": "screenshot",
            "args": {"region": [10, 5, 20, 15], "save_debug": True},
            "id": "screen-1",
        }])
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            context = create_mira_context({}, (), workspace)
            output = graph.compile().invoke({"messages": [request]}, context=context)
            saved = list((workspace / ".mira" / "_debug" / "screenshots").glob("*.png"))
            self.assertEqual(len(saved), 1)
            self.assertIn(str(saved[0]), output["messages"][-1].content_blocks[0]["text"])
        message = output["messages"][-1]
        self.assertIsInstance(message, ToolMessage)
        self.assertEqual(message.tool_call_id, "screen-1")
        self.assertIn("origin (10, 5), size 20x15", message.content_blocks[0]["text"])
        self.assertEqual(message.content_blocks[1]["type"], "image")
        self.desktop.screenshot.assert_called_once_with(region=(10, 5, 20, 15))
        restored = JsonPlusSerializer().loads_typed(JsonPlusSerializer().dumps_typed(message))
        self.assertEqual(restored.content_blocks, message.content_blocks)

    def test_image_reaches_vision_model_and_text_only_model_gets_notice(self) -> None:
        result = call(screenshot, {})
        image = result.content_blocks[1]
        cases = (
            ({"image_inputs": True}, True),
            ({"image_inputs": False}, False),
            ({"image_inputs": True, "image_tool_message": False}, False),
        )
        for profile, image_expected in cases:
            with self.subTest(profile=profile):
                model = SimpleNamespace(profile=profile)
                request = ModelRequest(model=model, messages=[result], tools=[])
                converted = ModelCompatibilityMiddleware(Path("."))._normalize_request(request)
                filtered = UnsupportedContentMiddleware()._filter_request(converted)
                outbound = filtered.messages[0].content
                if image_expected:
                    self.assertEqual(outbound[1]["type"], "image_url")
                    self.assertTrue(outbound[1]["image_url"]["url"].startswith("data:image/png;base64,"))
                else:
                    self.assertNotIn(image["base64"], str(outbound))
                    self.assertIn("does not support image content", str(outbound))
                self.assertEqual(result.content_blocks[1], image)

    def test_click_validates_primary_monitor_and_dpi_geometry(self) -> None:
        self.assertEqual(call(click, {"x": 79, "y": 59, "button": "right"}).status, "success")
        self.desktop.click.assert_called_once_with(x=79, y=59, button="right")
        self.assertEqual(call(click, {"x": 80, "y": 0}).status, "error")
        self.assertEqual(self.desktop.click.call_count, 1)
        self.desktop.size.return_value = (40, 30)
        self.assertIn("DPI scaling", call(click, {"x": 10, "y": 10}).text)
        self.assertIn("DPI scaling", call(screenshot, {}).text)
        self.assertEqual(self.desktop.click.call_count, 1)

    def test_click_cell_uses_reference_region_and_partial_cell_centres(self) -> None:
        self.desktop.size.return_value = (2560, 1440)
        self.desktop.screenshot.return_value = SimpleNamespace(size=(2560, 1440))
        reference = call(click_cell, {"cell": "B1", "region": [2200, 0, 300, 200], "grid_spacing": 50})
        self.assertEqual(reference.status, "success")
        self.assertEqual(reference.text, "Clicked cell B1 at (2275, 25).")
        self.desktop.click.assert_called_once_with(x=2275, y=25, button="left")
        self.assertEqual(
            call(click_cell, {"cell": "C2", "region": [2200, 0, 125, 75], "grid_spacing": 50}).text,
            "Clicked cell C2 at (2312, 62).",
        )
        self.desktop.click.assert_called_with(x=2312, y=62, button="left")
        self.desktop.size.return_value = (80, 60)
        self.desktop.screenshot.return_value = SimpleNamespace(size=(80, 60))
        self.assertEqual(call(click_cell, {"cell": "D3", "grid_spacing": 25}).text, "Clicked cell D3 at (77, 55).")
        self.desktop.click.assert_called_with(x=77, y=55, button="left")

    def test_click_cell_validates_ids_bounds_and_dpi_before_clicking(self) -> None:
        self.desktop.size.return_value = (300, 200)
        self.desktop.screenshot.return_value = SimpleNamespace(size=(300, 200))
        for cell in ("", "A0", "A01", "a1", "1A", "A-1", "D1", "A3", "AA1"):
            with self.subTest(cell=cell):
                self.assertEqual(call(click_cell, {"cell": cell, "grid_spacing": 100}).status, "error")
        for region in ([290, 0, 20, 20], [-1, 0, 20, 20], [0, 0, 0, 20]):
            with self.subTest(region=region):
                self.assertEqual(call(click_cell, {"cell": "A1", "grid_spacing": 50, "region": region}).status, "error")
        self.assertEqual(call(click_cell, {"cell": "A1", "grid_spacing": 75}).status, "error")
        self.desktop.screenshot.return_value = SimpleNamespace(size=(299, 200))
        self.assertIn("DPI scaling", call(click_cell, {"cell": "A1", "grid_spacing": 50}).text)
        self.desktop.click.assert_not_called()

    def test_click_cell_supports_multi_letter_columns(self) -> None:
        self.desktop.size.return_value = (2800, 100)
        self.desktop.screenshot.return_value = SimpleNamespace(size=(2800, 100))
        self.assertEqual(call(click_cell, {"cell": "AA1", "grid_spacing": 100}).text, "Clicked cell AA1 at (2650, 50).")
        self.desktop.click.assert_called_once_with(x=2650, y=50, button="left")
        self.desktop.screenshot.return_value = Image.new("RGB", (2800, 100), "white")
        labels = []
        draw_text = ImageDraw.ImageDraw.text

        def record_text(draw, xy, value, *args, **kwargs):
            labels.append(value)
            return draw_text(draw, xy, value, *args, **kwargs)

        with patch.object(ImageDraw.ImageDraw, "text", record_text):
            result = call(screenshot, {})
        self.assertEqual(result.status, "success")
        self.assertEqual(labels[25:28], ["Z", "AA", "AB"])

    def test_keyboard_and_scroll_actions_and_validation(self) -> None:
        self.assertEqual(call(type_text, {"text": "Hello"}).status, "success")
        self.desktop.write.assert_called_once_with("Hello")
        self.assertEqual(call(type_text, {"text": "café"}).status, "error")
        self.desktop.write.assert_called_once()
        self.assertEqual(call(press_key, {"keys": "ctrl+c"}).status, "success")
        self.desktop.hotkey.assert_called_once_with("ctrl", "c")
        self.assertEqual(call(press_key, {"keys": "enter"}).status, "success")
        self.desktop.press.assert_called_once_with("enter")
        self.assertEqual(call(press_key, {"keys": "bad"}).status, "error")
        self.assertEqual(call(scroll, {"direction": "down", "amount": 3}).status, "success")
        self.desktop.scroll.assert_called_once_with(-3)
        self.assertEqual(call(scroll, {"direction": "up", "amount": 0}).status, "error")

    def test_pyautogui_errors_do_not_escape_or_disable_failsafe(self) -> None:
        self.desktop.screenshot.side_effect = OSError("capture unavailable")
        result = call(screenshot, {})
        self.assertEqual(result.status, "error")
        self.assertIn("capture unavailable", result.text)
        self.desktop.screenshot.side_effect = None
        self.desktop.screenshot.return_value = Image.new("RGB", (80, 60))
        self.desktop.click.side_effect = RuntimeError("fail-safe triggered")
        self.assertIn("fail-safe triggered", call(click, {"x": 1, "y": 1}).text)
        self.assertIn("fail-safe triggered", call(click_cell, {"cell": "A1", "grid_spacing": 25}).text)
        self.assertTrue(self.desktop.FAILSAFE)

    def test_registration_and_existing_hitl_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            resources = build_resources(Path(directory), create_examples=False)
        names = {tool.name for tool in resources.tools}
        self.assertTrue({"screenshot", "click", "click_cell", "type_text", "press_key", "scroll"} <= names)
        self.assertTrue(tool_always_allow(DEFAULT_SETTINGS, "screenshot"))
        plan_excluded = effective_excluded_tools({"settings": DEFAULT_SETTINGS}, PLAN_DISABLED_TOOLS, False, planning=True)
        self.assertNotIn("screenshot", plan_excluded)
        self.assertTrue({"click", "click_cell", "type_text", "press_key", "scroll"} <= set(plan_excluded))
        interrupts = _write_interrupts({"settings": DEFAULT_SETTINGS})
        self.assertNotIn("screenshot", interrupts)
        for name in ("click", "click_cell", "type_text", "press_key", "scroll"):
            self.assertIn(name, interrupts)
            allowed = set_tool_always_allow(DEFAULT_SETTINGS, name, True)
            self.assertNotIn(name, _write_interrupts({"settings": allowed}))
        self.assertIn("write_file", interrupts)


if __name__ == "__main__":
    unittest.main()
