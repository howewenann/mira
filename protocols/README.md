# Purpose

External interoperability adapters for MIRA.

## Owns

Protocol-specific translation at the repository boundary.

## Does not own

Core behavior or MIRA-owned user interfaces.

## Depends on

`core.api`; Core never depends inward on this package.

## Public surface

Empty in Patch 1; the future ACP adapter belongs in `protocols/acp`.
