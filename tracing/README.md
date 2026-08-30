# Purpose

Optional tracing bootstrap, semantic spans, and diagnostic transcript mirroring.

## Owns

Tracing configuration bootstrap, processors, middleware spans, stream mirroring, and trace tailing.

## Does not own

Core execution or general UI behavior.

## Depends on

Optional tracing libraries, Core diagnostics, and shared terminal presentation primitives.

## Public surface

Tracing bootstrap functions and `TraceStream`.
