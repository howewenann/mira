# Advanced MIRA guide

This guide covers configuration and runtime details that are useful after the
basic workflow in the root README is running.

## Models and configuration

MIRA creates `.mira/models.yml` with a schema guide and commented examples.
Model profiles are ordered and named. `${NAME}` is the only supported
environment-reference syntax; unresolved, malformed, `${env:NAME}`, and
`$${NAME}` references are reported in Issues. Secondary model assignments can
inherit Main, and the usable context is the smaller of the Settings cap and
trustworthy provider/model metadata.

Workspace behavior is stored in `.mira/settings.yml`. The Settings screen owns
Git protection, tools and approvals, execution environments, dynamic eval
subagents, planning todos, rubric grading, model assignments, and tracing.
After editing resource files directly, use `/reload`; use `/reload-runtime`
when MCP connections or tracing must be recreated.

## Project resources

MIRA loads project customization from `.mira/`:

```text
.mira/
  models.yml
  settings.yml
  mcp/
    mcp.json         # active MCP configuration
    example.json     # inert examples
    schema.json      # supported contract
  prompts/           # recursive Mustache prompt files
  memories/          # always-on Markdown context
  skills/            # DeepAgents SKILL.md folders
  subagents/         # Python SUBAGENTS definitions
  tools/             # active custom tools
  examples/tools/    # inert tool examples
```

Project resources override built-ins with the same name. Prompt paths flatten
to commands with `__` between suffix-free path components, so
`prompts/review/python.md` becomes `/prompt__review__python`. Collisions are
excluded and reported in Issues. `/prompts`, `/memories`, `/skills`, `/tools`,
and `/subagents` show the active resources.

## MCP and reusable prompts

`.mira/mcp/mcp.json` is the active MCP configuration. `example.json` is never
loaded, while `schema.json` documents accepted keys. MCP string values use the
same `${NAME}` resolver as model profiles; templates remain unresolved on disk
and in approval previews, while a separate runtime copy supplies resolved
values to the connection.

Local stdio and remote Streamable HTTP servers require approval before first
use. Server and tool enablement, persistent approval, and Plan access live in
Settings. Browser OAuth is available from the MCP panel, with local token state
under `~/.mira/_state/mcp-tokens/`.

MCP tools use names such as `mcp__local__search`. Fixed resources appear in `@`
completion as `@mcp__<server>__<exact-uri>` and are read on demand. MCP prompts
appear as `/mcp__<server>__<prompt>`. Prompts with only required arguments use
positional values; if any argument is optional, pass supplied values as
`name=value` and quote values containing spaces.

## Tools and execution environments

Standard LangChain `@tool` functions run in MIRA's Python environment. A bad
project tool is isolated, omitted from the agent, and explained in Issues;
`/reload` retries it after repair. Ordinary tool exceptions become error
results the agent can inspect, while interrupts and turn cancellation retain
their native control flow.

Use `mira_tool_api.project_tool` when a function body must run in the configured
project Execute Environment. Keep project-only imports inside that function;
MIRA still imports the containing file for discovery. The project environment
does not need LangChain or MIRA installed. See the examples generated under
`.mira/examples/tools/`.

Enabling `execute` switches the project backend to a local shell backend.
Settings can use the system shell, a named Conda environment, a Conda prefix,
or a virtual environment, with an explicit allowlist for additional inherited
environment-variable names. Tool enablement, always-allow approval, Plan, PTC,
and Rubric access remain independent policies.

`write_file` creates or fully replaces a file; `edit_file` performs targeted
replacement. Recursive `delete` is action-only and follows the configured
approval policy.

## LangGraph workflows and execution context

A running application exposes `application.workflows`, a small build-time
facade. `mira.agent(...)` creates an independent workflow-local specialization
of an existing configured MIRA subagent; it does not mutate the reusable base.
The default base is `general-purpose`. Tool replacements use MIRA's existing
resolver, `tools=None` inherits, and `tools=[]` selects no tools. The model is
always inherited from the selected base. Configure a separate MIRA subagent and
select it by name when a workflow needs a different model.

MIRA Workflows are ordinary LangGraph graphs. Construct a workflow-local agent,
then call it from an ordinary node that maps domain state into agent input and
the agent result back into domain state:

```python
from typing import NotRequired, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from mira import MiraContext


class InputState(TypedDict):
    topic: str


class State(InputState):
    research: NotRequired[str]


def workflow(mira):
    researcher = mira.agent(name="researcher", system_prompt="Research carefully.")

    async def research(state: State) -> dict[str, str]:
        result = await researcher.ainvoke(
            {"messages": [HumanMessage(state["topic"])]}
        )
        return {"research": result["messages"][-1].text}

    graph = StateGraph(State, input_schema=InputState, context_schema=MiraContext)
    graph.add_node("research", research)
    graph.add_edge(START, "research")
    graph.add_edge("research", END)
    return graph.compile()


graph = workflow(application.workflows)
result = await graph.ainvoke(
    {"topic": "pineapples"}, context=application.workflows.context
)
```

The dedicated `input_schema` defines a discovered Workflow's launch arguments.
For a standalone LangGraph + DeepAgents version, replace the `mira.agent(...)`
construction with `create_deep_agent(model=model, ...)` and omit MIRA's
`context_schema` and execution `context`. The node's input/output mapping and
graph topology can stay the same. `mira.agent(...)` is a shallow MIRA worker
specialized from the configured subagent environment; a standalone
`create_deep_agent(...)` may have its own `task` tool and nested subagents. The
graph does not depend on that hierarchy.

Direct insertion, such as `graph.add_node("worker", mira.agent())`, also works
when the Workflow state already follows the agent's `messages` contract. The
node mapping above is better suited to domain-shaped state.

Nodes can use `Runtime[MiraContext]`; ordinary LangChain `@tool` functions use
their injected `ToolRuntime`. Both can access `runtime.context.tools["..."]`
and `runtime.context.agents["..."]`. This is a capability-bearing API for
trusted in-process Python. Direct calls through `context.tools` bypass a second
HITL prompt because they implement the already-invoked trusted operation, but
disabled tools, backend/filesystem rules, unavailable MCP capabilities, and
other hard MIRA restrictions still apply. Delegated agents retain native
DeepAgents task execution. Intermediate Python values pass directly between
capabilities without entering the parent model's messages; MIRA's trace export
policy also redacts these marked nested-call payloads.

MIRA does not provide a workflow DSL or automatic state mapping. User code
continues to own state, reducers, edges, branches, loops, parallelism, `Send()`,
and adapters for domain-shaped state. Workflows that use MIRA's context tools or
agents directly are intentionally more coupled to MIRA. Runnable examples are
under `.mira/examples/workflows/`.

## Plans, Goals, and sessions

Plan mode is a continuous read-only conversation that can present one durable
Plan. Goals retain only an Objective and Success Criteria, leaving the approach
to Act. Their `-show`, `-resume`, and `-clear` commands are listed in `/help`;
MIRA retains one current Plan or Goal.

Sessions live under `.mira/_sessions/` and can be resumed with `--resume` or
`--session <id>`. `/compact` asks the active DeepAgents summarization middleware
to compact older context immediately.

## Generic OTLP tracing

Install the optional tracing runtime with:

```bash
pip install "mira[tracing]"
```

Tracing enablement and selection stay in `.mira/settings.yml`:

```yaml
tracing:
  enabled: true
  profile: corporate
```

MIRA bootstraps `.mira/tracing.yml` once with Phoenix, MLflow, and LangSmith
examples. Each profile defines a complete OTLP/HTTP `endpoint`, a `headers`
mapping, and optional profile-wide `span_attributes`. Add any number of profiles
directly to that file:

```yaml
profiles:
  corporate:
    endpoint: https://example.com/otel/v1/traces
    headers:
      Authorization: "Bearer ${TRACE_TOKEN}"
      tenant: my-team
    span_attributes:
      deployment.environment: production
```

Existing registries are never overwritten or extended automatically.
Environment references stay literal in `tracing.yml` and the Settings preview.
MIRA resolves only the selected profile's in-memory runtime copy at the OTLP
boundary. Supply referenced values before starting or reloading MIRA.

LangSmith collects LangChain and DeepAgents runs in OTel-only mode so it remains
the sole owner of the trace tree. Before export, MIRA adds OpenInference span
kinds, structured inputs and outputs, model messages, tool attributes, and token
counts to those same spans. A standard OTel batch processor sends the enriched
tree to every selected OTLP/HTTP profile; there is no backend-specific
instrumentation. Phoenix is the recommended local viewer, while LangSmith and
custom compatible destinations are transport profiles in the same registry.

Reload Runtime closes the current OTel-only LangSmith client, drains and closes
its MIRA-owned OTel provider, then rebuilds the processor chain with the selected
profile's endpoint, headers, and span attributes. MIRA temporarily sets the
LangSmith callback activation values required by LangChain and restores the
launching process's previous values on reload, disable, or shutdown. If the
extra is missing, tracing is disabled for that runtime and Issues shows the
install command.
Exporter connection failures follow normal OpenTelemetry behavior and do not
disable the agent runtime.

## Safety, storage, and diagnostics

- MIRA checks Git protection before agent work starts in a workspace.
- Dangerous tools require approval unless explicitly allowed in Settings.
- Error reports are stored under `.mira/_errors/`; `/clear-errors` removes them.
- `--trace` opens MIRA's diagnostic transcript at `.mira/_logs/mira.log`; this
  is separate from OTLP agent tracing.
- `/runtime`, `/session`, `/tools`, `/memories`, `/skills`, and `/subagents`
  provide read-only runtime inspection without a model request.

For contributors, repository rules are in `AGENTS.md`, architecture rationale
is in `ARCHITECTURE_DECISIONS.md`, and user-driven scenarios are in
`tests/manual/prompts.md`.
