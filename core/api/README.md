# Purpose

The shared contract between headless MIRA and its consumers.

## Owns

Typed semantic events, blocking requests, and session snapshots.

## Does not own

Textual, terminal, Qt, ACP, or raw graph-stream transport.

## Depends on

Python types and native-shaped LangChain-normalized payloads only.

## Public surface

`Frontend`, `FrontendEvent`, `FrontendRequest`, and `SessionSnapshot`.
