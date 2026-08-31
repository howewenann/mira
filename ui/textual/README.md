# Purpose

MIRA's interactive Textual terminal UI.

## Owns

Slash commands, navigation, widgets, styles, focus, platform input, and rendering.

## Does not own

Turn semantics, Goal/Plan state transitions, persistence, or native stream decoding.

## Depends on

Textual, `mira`, `mira.api`, internal Core helpers, and narrow shared UI
primitives.

## Public surface

`MiraApp` and `TextualFrontend`.
