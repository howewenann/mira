# Purpose

Run and interpret native DeepAgents and LangGraph turns.

## Owns

Turn streaming, HITL resume, and projection of native stream records.

## Does not own

UI rendering, external protocols, or replacement graph/message abstractions.

## Depends on

DeepAgents, LangChain/LangGraph, `core.interface`, and session recording.

## Public surface

None. Owned Core modules call `runner.run_turn` and `turns.run_user_turn`.
