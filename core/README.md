# Purpose

Headless MIRA application behavior over native DeepAgents and LangGraph primitives.

## Owns

Application orchestration, native turn execution, consumer events, context accounting, and diagnostics.

## Does not own

UI rendering, external protocol adapters, or durable session schemas.

## Depends on

`agent`, `session`, `config`, and upstream native runtime libraries.

## Public surface

`core.application` and `core.api` are the supported headless entry points.
