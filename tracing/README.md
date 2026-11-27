# Purpose

Optional tracing bootstrap, semantic spans, and diagnostic transcript mirroring.

## Owns

Tracing configuration bootstrap, processors, middleware spans, and trace tailing.

## Does not own

Core execution or general UI behavior.

## Depends on

Optional tracing libraries and Core diagnostics.

## Public surface

Tracing bootstrap functions. UI transcript mirroring is owned by
[`ui/shared/trace.py`](../ui/shared/trace.py).
