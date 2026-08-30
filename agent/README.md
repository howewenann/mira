# Purpose

DeepAgents-native construction and MIRA extensions.

## Owns

Agent factories, middleware, planning policy, resources, MCP integration, and tools.

## Does not own

Application orchestration, persistence, or presentation.

## Depends on

DeepAgents, LangChain/LangGraph, configuration, and narrow Core services.

## Public surface

`agent.factory` is the canonical construction entry point.
