# Purpose

DeepAgents-native construction and MIRA extensions.

## Owns

`factory.py` is the composition entry point, and `llm.py` constructs models.
The domain packages have focused ownership:

- `middleware/` contains generic DeepAgents and LangChain middleware extensions.
- `rubric/` contains the MIRA Rubric verification and grading feature.
- `planning/` contains formal Plan and Goal policy support.
- `subagents/` contains project subagent discovery and DeepAgents compilation.
- `tools/` contains tool specs, discovery, project-runtime tooling, and failures.
- `resources/` contains backends, memories, skills, and bundled defaults.
- `mcp/` contains MCP integration.

## Does not own

Application orchestration, persistence, or presentation.

## Depends on

DeepAgents, LangChain/LangGraph, configuration, and narrow Core services.

## Public surface

`agent.factory` is the canonical construction entry point.
