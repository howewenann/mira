# Purpose

Internal consumer-facing interface between headless MIRA Core and supported
public consumers.

The supported external Python surface is exposed through `mira.api`.

## Owns

Typed semantic events, blocking requests, and session snapshots.

## Does not own

Textual, terminal, Qt, ACP, or raw graph-stream transport.

## Depends on

Python types and native-shaped LangChain-normalized payloads only.

## Supported access

Core implementation modules import `core.interface` directly. UI, protocol,
and external consumers import the same underlying types through `mira.api`.
`FrontendEmitter` remains internal Core-side plumbing and is not public API.
