# Purpose

Human interfaces owned by MIRA.

## Owns

Textual, plain-terminal, and narrowly shared presentation primitives.

## Does not own

Headless orchestration, durable state, or external interoperability protocols.

## Depends on

`core.api` and interface-specific presentation libraries.

## Public surface

`ui.textual` and `ui.terminal`; a future Qt interface belongs in `ui.qt`.
