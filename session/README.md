# Purpose

Durable MIRA session state and persistence.

## Owns

Per-chat SQLite checkpoints, JSON display history, dashboards, Goal/Plan records, event recording, stores, and values.

## Does not own

Application workflows or presentation.

## Depends on

Core event projection where persistence forwards observable activity.

## Public surface

`SessionStore`, checkpoint helpers, and current Goal/Plan state functions.
