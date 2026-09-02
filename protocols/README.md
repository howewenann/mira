# Purpose

External interoperability adapters for MIRA.

## Owns

Protocol-specific translation at the repository boundary.

## Does not own

Core behavior or MIRA-owned user interfaces.

## Depends on

`mira.api`; Core never depends inward on this package.

## Public surface

`protocols.acp` implements the stock Agent Client Protocol `Agent` interface
over stdio. It consumes only MIRA's supported `mira` and `mira.api` facades;
DeepAgents remains behind the application boundary.
