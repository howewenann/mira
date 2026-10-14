"""Primary-monitor computer tools backed by PyAutoGUI."""

from __future__ import annotations

import base64
import re
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage


def _computer_error(name: str, runtime: ToolRuntime, error: Exception) -> ToolMessage:
    return ToolMessage(
        content=f"Error: {name} failed: {error}",
        name=name,
        tool_call_id=runtime.tool_call_id,
        status="error",
    )


def _region_bounds(region: list[int] | None, screen_width: int, screen_height: int) -> tuple[int, int, int, int]:
    if region is None:
        return 0, 0, screen_width, screen_height
    if (
        not isinstance(region, (list, tuple))
        or len(region) != 4
        or any(type(value) is not int for value in region)
    ):
        raise ValueError("region must be [left, top, width, height] in screen pixels")
    left, top, width, height = region
    if (
        left < 0 or top < 0 or width <= 0 or height <= 0
        or left + width > screen_width or top + height > screen_height
    ):
        raise ValueError(f"region {region} is outside the primary monitor ({screen_width}x{screen_height})")
    return left, top, width, height


def _column_label(index: int) -> str:
    label = ""
    while True:
        index, remainder = divmod(index, 26)
        label = chr(ord("A") + remainder) + label
        if index == 0:
            return label
        index -= 1


@tool("screenshot")
def screenshot(
    runtime: ToolRuntime[Any, dict],
    grid: bool = True,
    grid_spacing: Literal[25, 50, 100] | None = None,
    region: list[int] | None = None,
    save_debug: bool = False,
    grid_mode: Literal["categorical", "coordinates"] = "categorical",
) -> ToolMessage:
    """Capture the primary monitor or a region [left, top, width, height].

    Use the default categorical grid: identify the cell containing the target's
    centre, then call click_cell with that cell and the same region and spacing.
    Do not estimate screen coordinates. Crop again for small targets spanning
    cells, and verify important clicks with another screenshot. Set grid_mode
    to "coordinates" to compare the older absolute-pixel ruler. Both grids
    add ruler margins outside the unscaled captured desktop pixels.
    The default spacing is 100 pixels for the full screen and 50 for a region;
    set grid_spacing to 25, 50, or 100, or grid=False for a plain image.
    Set save_debug=True only when a saved screenshot is explicitly requested.
    """
    try:
        import pyautogui

        pyautogui.FAILSAFE = True
        screen_width, screen_height = pyautogui.size()
        if region is None:
            left = top = 0
            image = pyautogui.screenshot()
            width, height = image.size
            if (width, height) != (screen_width, screen_height):
                raise ValueError(
                    f"screenshot is {width}x{height}, but mouse coordinates use "
                    f"{screen_width}x{screen_height}; check Windows DPI scaling"
                )
        else:
            left, top, width, height = _region_bounds(region, screen_width, screen_height)
            image = pyautogui.screenshot(region=(left, top, width, height))
            if image.size != (width, height):
                raise ValueError(
                    f"screenshot region is {image.width}x{image.height}, but requested "
                    f"{width}x{height}; check Windows DPI scaling"
                )
        if grid:
            from PIL import Image, ImageDraw, ImageFont

            spacing = grid_spacing if grid_spacing is not None else (50 if region is not None else 100)
            if spacing not in (25, 50, 100):
                raise ValueError("grid_spacing must be 25, 50, or 100 pixels")
            if grid_mode not in ("categorical", "coordinates"):
                raise ValueError("grid_mode must be 'categorical' or 'coordinates'")
            if image.mode != "RGB":
                image = image.convert("RGB")
            try:
                font = ImageFont.truetype("arialbd.ttf", 16)
            except OSError:
                font = ImageFont.load_default()
            measure = ImageDraw.Draw(image)
            if grid_mode == "categorical":
                column_count = (width + spacing - 1) // spacing
                row_count = (height + spacing - 1) // spacing
                y_labels = [str(row + 1) for row in range(row_count)]
            else:
                y_labels = [f"Y:{top + y}" for y in range(-top % 100, height, 100) if top + y != 0]
            ruler_left = max(
                64,
                max((measure.textbbox((0, 0), label, font=font, stroke_width=1)[2] for label in y_labels), default=0) + 8,
            )
            ruler_top = max(28, measure.textbbox((0, 0), "X:88888", font=font, stroke_width=1)[3] + 8)
            annotated = Image.new("RGB", (width + ruler_left, height + ruler_top), (32, 36, 42))
            annotated.paste(image, (ruler_left, ruler_top))
            draw = ImageDraw.Draw(annotated, "RGBA")
            label_style = {"font": font, "fill": "white", "stroke_width": 1, "stroke_fill": "black"}
            if grid_mode == "categorical":
                for x in range(spacing, width, spacing):
                    draw.line((ruler_left + x, ruler_top, ruler_left + x, annotated.height - 1), fill=(0, 180, 255, 45))
                for y in range(spacing, height, spacing):
                    draw.line((ruler_left, ruler_top + y, annotated.width - 1, ruler_top + y), fill=(0, 180, 255, 45))
                for column in range(column_count):
                    start, end = column * spacing, min((column + 1) * spacing, width)
                    label = _column_label(column)
                    label_x = ruler_left + (start + end) / 2 - draw.textlength(label, font=font) / 2
                    draw.text((label_x, 4), label, **label_style)
                for row in range(row_count):
                    start, end = row * spacing, min((row + 1) * spacing, height)
                    label = str(row + 1)
                    bbox = draw.textbbox((0, 0), label, font=font, stroke_width=1)
                    label_y = ruler_top + (start + end) / 2 - (bbox[1] + bbox[3]) / 2
                    draw.text(((ruler_left - draw.textlength(label, font=font)) / 2, label_y), label, **label_style)
            else:
                # Coordinate mode keeps grid lines aligned to absolute desktop pixels.
                for x in range(-left % spacing, width, spacing):
                    image_x = ruler_left + x
                    major = (left + x) % 100 == 0
                    draw.line((image_x, ruler_top, image_x, annotated.height - 1), fill=(0, 180, 255, 85 if major else 35))
                    if major:
                        draw.line((image_x, ruler_top - 5, image_x, ruler_top - 1), fill=(210, 225, 235, 180))
                for y in range(-top % spacing, height, spacing):
                    image_y = ruler_top + y
                    major = (top + y) % 100 == 0
                    draw.line((ruler_left, image_y, annotated.width - 1, image_y), fill=(0, 180, 255, 85 if major else 35))
                    if major:
                        draw.line((ruler_left - 5, image_y, ruler_left - 1, image_y), fill=(210, 225, 235, 180))
                if region is None:
                    draw.text((4, 4), "(0,0)", **label_style)
                for x in range(-left % 100, width, 100):
                    if left + x == 0:
                        continue
                    label = f"X:{left + x}"
                    label_x = max(4, min(ruler_left + x + 4, annotated.width - draw.textlength(label, font=font) - 4))
                    draw.text((label_x, 4), label, **label_style)
                for y in range(-top % 100, height, 100):
                    if top + y == 0:
                        continue
                    label = f"Y:{top + y}"
                    label_height = draw.textbbox((0, 0), label, font=font)[3]
                    label_y = max(ruler_top + 4, min(ruler_top + y + 4, annotated.height - label_height - 4))
                    draw.text((4, label_y), label, **label_style)
            image = annotated
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        png_bytes = buffer.getvalue()
        description = (
            f"Primary monitor screenshot: {width}x{height} pixels. Coordinates start at (0, 0)."
            if region is None
            else f"Primary monitor screenshot region: origin ({left}, {top}), size {width}x{height} pixels. "
            f"Original screen: {screen_width}x{screen_height} pixels. Coordinates are absolute screen pixels."
        )
        if grid:
            description += (
                f" Annotated image: {image.width}x{image.height} pixels; captured desktop pixels start at "
                f"image ({ruler_left}, {ruler_top})."
            )
            if grid_mode == "categorical":
                description += (
                    f" Categorical grid: {column_count} columns and {row_count} rows of {spacing}-pixel cells, "
                    "indexed from A1 at the capture's top-left. Use click_cell with the same region and grid_spacing."
                )
            else:
                description += " Ruler labels are absolute desktop coordinates."
        if save_debug:
            workspace = getattr(getattr(runtime, "context", None), "workspace", None)
            if workspace is None:
                raise ValueError("debug saving requires a MIRA workspace")
            debug_dir = Path(workspace).expanduser().resolve() / ".mira" / "_debug" / "screenshots"
            debug_dir.mkdir(parents=True, exist_ok=True)
            filename = f"screenshot-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}-{uuid4().hex}.png"
            debug_path = debug_dir / filename
            with debug_path.open("xb") as debug_file:
                debug_file.write(png_bytes)
            description += f" Debug screenshot saved to {debug_path}."
        return ToolMessage(
            content_blocks=[
                {"type": "text", "text": description},
                {"type": "image", "base64": base64.b64encode(png_bytes).decode("ascii"), "mime_type": "image/png"},
            ],
            name="screenshot",
            tool_call_id=runtime.tool_call_id,
            additional_kwargs={"read_file_path": "the primary monitor screenshot"},
        )
    except Exception as error:
        return _computer_error("screenshot", runtime, error)


@tool("click")
def click(x: int, y: int, runtime: ToolRuntime, button: Literal["left", "right", "middle"] = "left") -> ToolMessage:
    """Click a pixel on the primary monitor using screenshot coordinates."""
    try:
        import pyautogui

        pyautogui.FAILSAFE = True
        width, height = pyautogui.size()
        if pyautogui.screenshot().size != (width, height):
            raise ValueError("screenshot and mouse coordinates differ; check Windows DPI scaling")
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"({x}, {y}) is outside the primary monitor ({width}x{height})")
        pyautogui.click(x=x, y=y, button=button)
        return ToolMessage(content=f"Clicked {button} at ({x}, {y}).", name="click", tool_call_id=runtime.tool_call_id)
    except Exception as error:
        return _computer_error("click", runtime, error)


@tool("click_cell")
def click_cell(cell: str, grid_spacing: int, runtime: ToolRuntime, region: list[int] | None = None) -> ToolMessage:
    """Click the centre of a categorical screenshot cell such as B1.

    Use the cell containing the target's centre, with the same region and
    grid_spacing as the categorical screenshot. Do not calculate coordinates
    manually. Crop and rescreenshot small targets spanning cells, then verify
    important interactions with a new screenshot.
    """
    try:
        import pyautogui

        pyautogui.FAILSAFE = True
        screen_width, screen_height = pyautogui.size()
        left, top, width, height = _region_bounds(region, screen_width, screen_height)
        if type(grid_spacing) is not int or grid_spacing not in (25, 50, 100):
            raise ValueError("grid_spacing must be 25, 50, or 100 pixels")
        match = re.fullmatch(r"([A-Z]+)([1-9][0-9]*)", cell) if isinstance(cell, str) else None
        if match is None:
            raise ValueError("cell must be an uppercase column and positive row, such as B1")
        column = 0
        for letter in match.group(1):
            column = column * 26 + ord(letter) - ord("A") + 1
        column -= 1
        row = int(match.group(2)) - 1
        cell_left, cell_top = column * grid_spacing, row * grid_spacing
        if cell_left >= width or cell_top >= height:
            raise ValueError(f"cell {cell} is outside the captured region ({width}x{height})")
        if pyautogui.screenshot().size != (screen_width, screen_height):
            raise ValueError("screenshot and mouse coordinates differ; check Windows DPI scaling")
        cell_right = min(cell_left + grid_spacing, width)
        cell_bottom = min(cell_top + grid_spacing, height)
        x = left + (cell_left + cell_right) // 2
        y = top + (cell_top + cell_bottom) // 2
        pyautogui.click(x=x, y=y, button="left")
        return ToolMessage(content=f"Clicked cell {cell} at ({x}, {y}).", name="click_cell", tool_call_id=runtime.tool_call_id)
    except Exception as error:
        return _computer_error("click_cell", runtime, error)


@tool("type_text")
def type_text(text: str, runtime: ToolRuntime) -> ToolMessage:
    """Type printable ASCII text into the active application."""
    try:
        import pyautogui

        pyautogui.FAILSAFE = True
        if any(character not in "\n\t" and not 32 <= ord(character) <= 126 for character in text):
            raise ValueError("PyAutoGUI typing supports ASCII text only")
        pyautogui.write(text)
        return ToolMessage(content=f"Typed {len(text)} characters.", name="type_text", tool_call_id=runtime.tool_call_id)
    except Exception as error:
        return _computer_error("type_text", runtime, error)


@tool("press_key")
def press_key(keys: str, runtime: ToolRuntime) -> ToolMessage:
    """Press one key or a combination such as ctrl+c in the active application."""
    try:
        import pyautogui

        pyautogui.FAILSAFE = True
        parts = [part.strip().lower() for part in keys.split("+")]
        if not parts or any(part not in pyautogui.KEYBOARD_KEYS for part in parts):
            raise ValueError(f"unsupported key or combination: {keys}")
        if len(parts) == 1:
            pyautogui.press(parts[0])
        else:
            pyautogui.hotkey(*parts)
        return ToolMessage(content=f"Pressed {keys}.", name="press_key", tool_call_id=runtime.tool_call_id)
    except Exception as error:
        return _computer_error("press_key", runtime, error)


@tool("scroll")
def scroll(direction: Literal["up", "down"], runtime: ToolRuntime, amount: int = 3) -> ToolMessage:
    """Scroll the active application up or down by a positive number of steps."""
    try:
        import pyautogui

        pyautogui.FAILSAFE = True
        if not 1 <= amount <= 100:
            raise ValueError("amount must be between 1 and 100")
        pyautogui.scroll(amount if direction == "up" else -amount)
        return ToolMessage(content=f"Scrolled {direction} {amount} steps.", name="scroll", tool_call_id=runtime.tool_call_id)
    except Exception as error:
        return _computer_error("scroll", runtime, error)
