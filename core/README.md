# Purpose

Headless MIRA application behavior over native DeepAgents and LangGraph primitives.

## Owns

Application orchestration, native turn execution, consumer events, context accounting, and diagnostics.

## Does not own

UI rendering, external protocol adapters, or durable session schemas.

## Depends on

`agent`, `session`, `config`, and upstream native runtime libraries.

## Public surface

External developers use `mira` and `mira.api`. This package remains MIRA's
implementation; `core.interface` is the internal boundary behind that facade.
