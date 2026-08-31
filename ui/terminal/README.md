# Purpose

MIRA's non-interactive `--prompt` interface.

## Owns

Plain terminal rendering and its MIRA API adapter.

## Does not own

Application orchestration or Textual widgets.

## Depends on

Rich, `mira.api`, and narrow shared terminal primitives.

## Public surface

`Renderer` and `TerminalFrontend`.
