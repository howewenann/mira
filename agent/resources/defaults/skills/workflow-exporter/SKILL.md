---
name: workflow-exporter
description: Export an existing MIRA Workflow from .mira/workflows/ to a separate standalone LangGraph and DeepAgents program without MIRA runtime dependencies. Use when asked to convert or port that project graph outside MIRA.
license: MIT
compatibility: designed for MIRA
---

# Workflow Exporter

Convert an existing MIRA Workflow into a standalone Python program that constructs and runs its LangGraph graph without starting MIRA. Keep the original `.mira/workflows/<name>.py` intact unless the user explicitly asks to replace it.

## Inspect the source

1. Read the actual Workflow file, its imports and local dependencies, and the relevant project models, tools, and configured agents. Read the smallest relevant managed example under `.mira/examples/workflows/` when a MIRA seam needs explanation. Use existing project conventions to choose a separate destination for the export.
2. Trace its input and state schemas, nodes, edges, conditional routes, reducers, loops, parallel work, `Send` calls, and domain-state transformations. Identify which parts already use ordinary LangGraph and which depend on MIRA. Do not convert by a fixed text-substitution table.
3. If a required model, tool, credential source, or output behavior cannot be inferred, ask one focused question. Otherwise choose explicit standalone dependencies and proceed.

## Export the graph

- Preserve the source graph's behavior and topology wherever possible. Keep its public inputs and state transformations recognizable; adapt only what removing MIRA requires.
- Replace each actual MIRA seam according to its use. For `mira.agent(...)`, construct an equivalent DeepAgents worker with an explicitly selected model, tools, prompt, and response format. Preserve node adapters that map domain state to agent messages and responses back to domain state. For `Runtime[MiraContext]`, `runtime.context.tools`, `runtime.context.agents`, and `context_schema=MiraContext`, supply explicit standalone dependencies through ordinary LangGraph context or node closures. Replace `context=mira.context` with that context, or omit it when unused.
- Replace the `workflow(mira)` factory and `MiraApplication.start(...)` harness with ordinary graph construction and an entry point that invokes or streams the compiled graph. Do not import MIRA or rely on its discovery, session, tool registry, approvals, or configured subagents at runtime. Do not add an exporter framework or a second runtime abstraction.
- Keep a small runnable example input and document the standalone model/tool configuration the program needs. Respect the source's approval-sensitive actions rather than silently treating an interrupt as a completed run.

## Verify and report

Check that the exported file compiles and has no MIRA imports or hidden MIRA runtime calls. Run a small safe standalone input when the required dependencies and credentials are available; inspect the final result, or report the exact interrupt or external blocker. Compare the output and meaningful graph structure with the source's intended behavior. Report the exported path, how to run it, required configuration, verification performed, and any behavior that could not be preserved.
