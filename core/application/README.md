# Purpose

The headless MIRA application and session lifecycle.

## Owns

Startup, sessions, ACT/PLAN semantics, Goal/Plan workflows, cancellation, and snapshots.

## Does not own

Stream decoding, persistence schemas, or presentation.

## Depends on

`agent`, `core.execution`, `core.interface`, `session`, and `config`.

## Public surface

`MiraApplication` and `MiraSession` are supported externally through `mira`.
The remaining helpers in this package are internal implementation details.
