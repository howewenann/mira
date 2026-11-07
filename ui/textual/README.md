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

## Workspace navigation

MIRA starts with the left navigation rail and main Chat. The speech bubble
toggles real saved chat history; the folder toggles the workspace file tree.
The wrench opens Settings on General. History retains New, Rename, Pin/Unpin,
Delete, and session switching. Its row menu opens to the right of that row.

Select a file to open a read-only preview tab. Markdown is rendered; other text
uses a read-only editor. Binary files and files larger than 350 KB show a preview
limitation. Expand folders in the file tree to browse further.

Hover or select a tab to expose its actions. `›` docks it to the right, `‹`
returns it to the main pane, and `×` closes it. Chat is permanent. Inspector
opens from the existing inspection controls and remains live when moved.
The arrows scroll each tab strip; `⋯` lists its tabs. Escape or a click outside
closes a dropdown. The main `⋯` menu also opens Inspector and the BTW preview.

History/Files starts at 42 columns and resizes between 36 and 66; the dock starts
at 42 and resizes between 36 and 76. Each has its own border. The one-cell gaps
beside them light teal only while hovered or dragged. Narrow terminals hide the
sidebar and move docked tabs into main, returning them when space is restored.

The compact prompt retains Enter to submit, Shift+Enter for a newline, prompt
history, `/` and `@` completion, and the `[=]` resize grip. Add file uses the
existing Windows picker and upload path. The right-aligned dropdown selects a
configured model alias through the same runtime rebuild used by Models settings.
The footer shows the effective alias after activation succeeds; configuration
alone does not change it. Token, turn, and duration metrics remain visible.

BTW is a UI preview for a future release. It can be opened, docked, returned,
resized, and closed. Its composer, attachments, and model selection are disabled.
It runs no agent, submits no messages to Chat, and has no conversation or
checkpoint persistence. Its footer model remains unset.
