# Native LangGraph workflows with MIRA

MIRA supplies two things to ordinary LangGraph code:

- `application.workflows.agent(...)` builds a workflow-local specialization of
  an existing configured MIRA subagent.
- `application.workflows.context` supplies the current resolved MIRA tools and
  subagents through LangGraph's native runtime context.

MIRA does not provide a workflow DSL. Your code owns the `StateGraph`, state
schema, reducers, edges, branches, loops, `Send()` calls, and adapters.

Read the examples in this order:

1. `minimal.py` - the smallest agent-compatible graph.
2. `structured_agents.py` - independent text and structured variants.
3. `tools_and_agents.py` - deterministic nodes using runtime capabilities.

Each example starts a headless `MiraApplication`. Configure the workspace's
models and subagents before running it with `python <example>.py`. The examples
use native LangGraph `stream_mode="updates"` to print each completed node output
without enabling token-level streaming.
