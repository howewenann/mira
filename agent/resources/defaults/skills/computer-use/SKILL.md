---
name: computer-use
description: Inspect and operate a visible Windows desktop application through MIRA's primary-monitor screenshot, grid-cell click, typing, key, and scroll tools. Use when the user asks to work with an on-screen interface or when visible UI state is needed to complete the task.
license: MIT
compatibility: designed for MIRA on Windows with a vision-capable Main model
---

# Computer Use

Use MIRA's atomic computer tools to work with the visible primary monitor. Keep actions within the user's request and use app-specific tools when they can perform the task directly.

## Inspect the screen

1. Call `screenshot` for a full-screen view before choosing a target. Its categorical grid has 100-pixel cells, with column letters above the captured pixels and row numbers to the left. Identify the cell containing the target's centre. For a closer view, call `screenshot(region=[left, top, width, height])` around the target; the crop defaults to 50-pixel cells. Cell names start at A1 in the crop's top-left corner. Set `grid_spacing` to 25, 50, or 100 for another density, or `grid=False` for an unannotated view. The annotated image is larger than the unscaled desktop capture because of its ruler margins.
2. Identify the intended control from what is actually visible. If several controls could match, use `ask_user` for a focused clarification. Do not invent coordinates when an image is unavailable.
3. A text-only model cannot inspect the screenshot. If the tool reports unsupported image content, explain that a vision-capable Main model with `image_inputs: true` is needed. If screenshot and mouse dimensions differ, stop and report the DPI scaling error; do not rescale or guess coordinates.

## Act through the tools

- Use `click_cell(cell, grid_spacing, region)` with the cell containing the target's centre. Pass the same `region` and `grid_spacing` used for that screenshot; omit `region` for a full-screen screenshot. Do not estimate or calculate screen coordinates manually. For a small target spanning multiple cells, take a closer cropped screenshot and choose its cell. Verify important interactions with a subsequent screenshot.
- `click(x, y)` remains available when exact screen coordinates are already known. Its default button is `left`; pass `button="right"` or `button="middle"` only when the task calls for it.
- Use `type_text(text)` for printable ASCII text in the active application. For control keys and shortcuts, use `press_key(keys)`, such as `enter` or `ctrl+a`. If the user needs non-ASCII text, explain the current typing limit instead of silently dropping characters.
- Use `scroll(direction, amount)` with `direction` set to `up` or `down` and a positive step count. The default amount is 3.
- Call the tools directly. Do not script an autonomous desktop loop or batch dependent actions in parallel. After navigation, scrolling, or another layout change, take a new screenshot before choosing another cell.

`screenshot` is read-only. Mouse and keyboard actions use MIRA's normal approval prompt unless the user has configured Always Allow. Respect a rejection and do not bypass approval through `execute`, `eval`, or another tool. Keep PyAutoGUI's fail-safe enabled and leave the user's interrupt control available.

## Verify and finish

Take a final screenshot when the result can be seen on screen. Report the observed outcome, any action that was rejected or interrupted, and any capture, vision, DPI, or primary-monitor limitation that prevented completion. Do not claim a click or other action succeeded solely because a tool call was attempted.
