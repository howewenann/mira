---
name: workflow-creator
description: Create, edit, debug, and smoke-test launchable Python/LangGraph files in a MIRA project's .mira/workflows/ directory. Use for persistent /workflow__ commands, graph state and nodes, agents, public inputs, or launch behavior.
license: MIT
compatibility: designed for MIRA
---

# Workflow Creator

Build a native LangGraph workflow that MIRA can discover and launch. Keep the graph's state, nodes, edges, branches, reducers, loops, and parallelism in ordinary LangGraph; use MIRA APIs for configured capabilities and execution context.

## Inspect and decide

1. Read the request and relevant project code, especially existing `.mira/workflows/` files and any tools or subagents the Workflow should use. When updating a Workflow, preserve its working structure and project behavior; change only what the request needs.
2. Consult the version-matched managed examples under `.mira/examples/workflows/`. Read `.mira/examples/workflows/README.md` for orientation, then the smallest relevant file: `demo.py` for deterministic graph and launch basics, `minimal.py` for a domain-state agent node, `structured_agents.py` for specialized or structured agents, or `tools_and_agents.py` for MIRA runtime-context tools and configured agents. Do not load every example by default. Follow their short standalone-conversion comments at MIRA-specific seams. Adapt example code to this Workflow's discovery contract and safe smoke input; some examples demonstrate a concept without a launchable input schema or runnable harness.
3. Infer reasonable design choices from that context. If one unresolved choice would materially change behavior, use `ask_user` for one focused question with concise, mutually exclusive options. Continue after the answer; ask another only if a separate material choice remains. Ask nothing when the request is sufficiently specified.

## Author the Workflow

- Put project Workflows in direct `.py` files under `.mira/workflows/`. The file stem becomes `/workflow__<stem>`; nested files are not discovered. Do not add a manifest or a MIRA-specific graph format.
- Export exactly `def workflow(mira):` as a synchronous, side-effect-free graph-construction factory. Return a compiled `StateGraph`. Put execution work inside nodes, not in the factory or at import time.
- Give `StateGraph` a dedicated `input_schema` separate from its internal state schema. Its public input must be a top-level object with at least one named property. Choose useful domain inputs; MIRA exposes them as `name=value` launch arguments and validates their types. Internal state can carry intermediate results.
- For agent-based work, normally construct the relevant `mira.agent(...)` workers near the top of `workflow(mira)`. Use ordinary node functions that close over them. Explicitly turn Workflow state into agent input (for example, `HumanMessage` in `messages`) and map returned `messages` text or `structured_response` back into domain-shaped state. A MIRA agent may also be inserted directly as a LangGraph node when the graph state follows its `messages` contract.
- `mira.agent()` specializes a configured subagent (default `general-purpose`); it does not add that worker to `MiraContext.agents`. For nodes that intentionally need MIRA's runtime-bound tools or configured agents, use `Runtime[MiraContext]`, `runtime.context.tools` or `runtime.context.agents`, and `context_schema=MiraContext`, as in `tools_and_agents.py`. Do not force that pattern into a workflow that only needs local agent workers.
- Keep standalone conversion simple: graph logic stays ordinary LangGraph. Add a few concise comments at actual MIRA seams, such as `mira.agent(...)`, `MiraContext`, `context=mira.context`, or developer startup. Explain relevant substitutions: construct a DeepAgents worker with your own model/tools, supply or omit host context, or start the graph without MIRA. Do not create a second implementation or portability machinery.

## Include and run a developer smoke test

Keep a small permanent `if __name__ == "__main__":` harness in each generated Workflow, including a useful new harness when updating a file that lacks one. Follow `minimal.py`, `structured_agents.py`, or `tools_and_agents.py`: use `asyncio.run(main())`, start a headless `MiraApplication` with the project workspace, get `mira = application.workflows`, and always `await application.shutdown()` in `finally`. Those managed examples call their factories directly so they can run from `.mira/examples/`; a generated project Workflow must additionally look up its stem with `application.workflow_registry.specs.get(Path(__file__).stem)`. If absent, report the matching `application.workflow_registry.issues` entry (or that no entry was found); if present, build through `spec.factory(mira)`. Call the graph with a small, safe, representative public input using `await graph.ainvoke(..., context=mira.context)` and print the result. Do not implement your own discovery or input validator.

When `execute` is available, choose the command for the shell that **execute actually runs**, then run the harness with `MIRA_PYTHON`, the interpreter hosting MIRA:

- On Windows, `execute` runs through `cmd` even if MIRA was started from PowerShell. Use: `"%MIRA_PYTHON%" .mira/workflows/<name>.py`
- On POSIX, `execute` runs through a POSIX shell. Use: `"$MIRA_PYTHON" .mira/workflows/<name>.py`

Keep the quotes around the interpreter path. If variable expansion fails, correct the shell syntax and retry; **never fall back to bare `python`**. Bare `python` intentionally uses the configured project environment and may lack MIRA dependencies. Inspect the output, fix actual Workflow failures, and rerun. Never weaken or bypass normal approval behavior to complete a smoke run. If `execute` is unavailable, leave the harness ready to run and say the smoke test was not executed. Do not invent a new testing tool or hard-code a user-specific interpreter.

## Finish

Check the generated Python file before finishing: when it uses MIRA-specific agents, context, or startup, it must contain a few useful comments at those seams explaining standalone substitutions. The answer alone does not satisfy this check. Also check that the Workflow is discoverable and its public inputs match the intended launch contract. A `[interrupt]` for tool approval is a **partial smoke run**, even with exit code 0: report the exact node or approval boundary reached, and do not claim later nodes ran unless their output was observed. Claim an end-to-end pass only when the expected final Workflow result/state was observed. Report the file changed, launch command, smoke-test result or precise limitation, and any approval boundary reached.
