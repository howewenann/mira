# Advanced MIRA guide

This guide explains how to use MIRA's workspace features after completing the
[quick start](../README.md). For implementation choices and trade-offs, follow
the links to [Architecture Decisions](../ARCHITECTURE_DECISIONS.md).

## Table of Contents

- [Models, settings, and reloads](#models-settings-and-reloads)
  - [Choose models and context limits](#choose-models-and-context-limits)
  - [Apply changes safely](#apply-changes-safely)
- [Project resources](#project-resources)
  - [Customize memories, skills, subagents, and tools](#customize-memories-skills-subagents-and-tools)
  - [Use reusable local prompts](#use-reusable-local-prompts)
- [MCP servers, tools, resources, and prompts](#mcp-servers-tools-resources-and-prompts)
  - [Configure and approve a server](#configure-and-approve-a-server)
  - [Use discovered capabilities](#use-discovered-capabilities)
- [Tools and execution environments](#tools-and-execution-environments)
  - [Choose a custom tool runtime](#choose-a-custom-tool-runtime)
  - [Enable shell execution and understand approval](#enable-shell-execution-and-understand-approval)
- [Plans, Goals, and sessions](#plans-goals-and-sessions)
  - [Develop and run a Plan](#develop-and-run-a-plan)
  - [Create and run a Goal](#create-and-run-a-goal)
  - [Resume sessions and compact context](#resume-sessions-and-compact-context)
- [LangGraph Workflows](#langgraph-workflows)
  - [Build a Workflow with a MIRA agent](#build-a-workflow-with-a-mira-agent)
  - [Use the execution context](#use-the-execution-context)
- [OTLP tracing](#otlp-tracing)
  - [Configure an export profile](#configure-an-export-profile)
  - [Reload and troubleshoot tracing](#reload-and-troubleshoot-tracing)
- [Safety, storage, and diagnostics](#safety-storage-and-diagnostics)

## Models, settings, and reloads

Models and workspace policies are separate. MIRA creates `.mira/models.yml` for
ordered, named model profiles, and `.mira/settings.yml` for assignments and
behavior. Open `/settings` to manage models, tool policy, execution environments,
Git protection, dynamic eval subagents, planning todos, rubric grading, and
tracing. `/models` opens the model part of Settings directly.

### Choose models and context limits

Assign a Main model first. Rubric, Summarization, and raw-subagent assignments
can inherit Main when left null. An explicitly named but missing profile is an
Issue, not a fallback. `${NAME}` is the supported environment reference syntax
in model profile values; MIRA reports unresolved or malformed references,
including `${env:NAME}` and `$${NAME}`. The usable context limit is the smaller
of the Settings cap and trustworthy model/provider metadata. See
[configuration and resource ownership](../ARCHITECTURE_DECISIONS.md#configuration-and-project-resources)
for the precedence and reload rules.

### Apply changes safely

After changing `.mira/` resource files or settings on disk, run `/reload` to
reread the environment, workspace configuration, and non-MCP project resources
and rebuild agents. Run `/reload-runtime` when MCP connections or tracing must
also be recreated. Settings changes use the active application path; model
changes rebuild the Act/Plan agents without restarting MCP. Use `/runtime` to
inspect the effective model and connection, and `/issues` for configuration
errors. Neither inspection command asks the model a question.

## Project resources

MIRA loads bundled defaults and then active project resources. A project memory,
skill, subagent, or tool with the same name replaces its bundled counterpart.
Examples under `.mira/examples/` are inert until you copy them into an active
resource directory.

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
  workflows/         # launchable Python Workflow factories
  examples/tools/    # inert tool examples
```

### Customize memories, skills, subagents, and tools

Put always-on Markdown context in `memories/`, `SKILL.md` folders in `skills/`,
Python `SUBAGENTS = [...]` definitions in `subagents/`, and custom Python tools
in `tools/`. `/memories`, `/skills`, `/subagents`, and `/tools` show what was
loaded and which defaults were replaced. Project tool files load independently:
a failed file is omitted, appears in `/issues`, and is retried on `/reload`.
Newly discovered subagents are disabled by default except `general-purpose`;
use Settings to enable them. The [resource rules](../ARCHITECTURE_DECISIONS.md#resource-discovery-and-replacement)
describe collisions, replacement, and execution boundaries.

### Use reusable local prompts

Put Mustache prompt files under `.mira/prompts/`. Path components become a
slash command separated by `__`: `prompts/review/python.md` is available as
`/prompt__review__python`. Run `/prompts` to see the effective commands.
Colliding names are excluded and reported in Issues. This directory is for
prompt templates; it does not load a memory or tool.

## MCP servers, tools, resources, and prompts

MCP connects MIRA to separately configured servers. A server can advertise
tools, reusable prompts, and fixed resources; it need not advertise all three.
MIRA discovers supported capabilities when the connection starts, then exposes
the allowed results to the current mode. The [MCP architecture](../ARCHITECTURE_DECISIONS.md#mcp-integration)
explains lifecycle and trust boundaries.

### Configure and approve a server

Edit `.mira/mcp/mcp.json`, the active file. For example, these are a local stdio
server and a remote Streamable HTTP server:

```json
{
  "mcpServers": {
    "local": {
      "type": "stdio",
      "command": "python",
      "args": ["/absolute/path/to/server.py"],
      "env": {}
    },
    "remote": {
      "type": "http",
      "url": "https://example.com/mcp",
      "headers": {"Authorization": "Bearer ${REMOTE_MCP_TOKEN}"}
    }
  }
}
```

Replace the illustrative paths and endpoint with a real server. The shipped
`example.json` is never loaded; `schema.json` documents accepted keys. MCP
string values may reference `${NAME}`. MIRA keeps templates on disk and in the
approval preview, then resolves a separate copy in memory for the connection.
A missing variable fails that server, not the entire MIRA runtime. Use
`/reload-runtime` after editing server configuration or environment values.

Local and remote servers require launch/use approval before first use. The
approval preview shows the command or URL and environment or header names.
Allow approves that server configuration for this process; Always allow saves
the approval fingerprint in Settings. A changed configuration needs approval
again. Server and tool enablement and Plan access are separate Settings
controls. Use `/mcp` to inspect server state or restart a server. Remote HTTP
servers use FastMCP's browser OAuth flow when protected. With its current
default in-memory storage, OAuth credentials do not survive a MIRA restart.

### Use discovered capabilities

MCP tool names look like `mcp__local__search`; tool policy still controls
whether the agent may use them. Fixed resources appear in `@` completion as
`@mcp__<server>__<exact-uri>` and are read on demand. MCP prompts appear as
`/mcp__<server>__<prompt>`. For a prompt with only required arguments, pass
positional values. If it has any optional arguments, use `name=value` for all
supplied values and quote values with spaces. `/prompts` lists the available
prompt commands. Some tools can request a form or URL decision during use;
answering that request resumes the same interrupted turn.

## Tools and execution environments

MIRA distinguishes a tool's Python runtime, the project file backend, and the
human approval policy. Enabling one capability does not automatically allow
every tool to call it. See [tools and permissions](../ARCHITECTURE_DECISIONS.md#tools-permissions-and-execution)
for the boundaries.

### Choose a custom tool runtime

Ordinary LangChain `@tool` functions run in MIRA's Python environment. Use
`mira_tool_api.project_tool` when the function body needs the configured project
Execute Environment. Keep project-only imports inside the function: discovery
still imports the containing file in MIRA. The child environment does not need
MIRA or LangChain installed. Start with the generated examples under
`.mira/examples/tools/`, then place active tools in `.mira/tools/`.

If a project tool file fails to load, its tools are omitted and `/issues`
explains the failure. Repair the file and run `/reload`. Ordinary exceptions
from an enabled workspace tool become model-visible error results; interrupts
and turn cancellation retain their native control flow.

### Enable shell execution and understand approval

Enable `execute` in Settings to switch the project backend to a local shell.
Choose the system shell, a named Conda environment, a Conda prefix, or a virtual
environment. Additional inherited environment variables are an allowlist of
*names*; their values come from the current process. MIRA does not inherit the
entire host environment into shell commands.

Dangerous built-in tools and newly discovered project tools require human
approval unless their Settings policy explicitly allows them. Plan mode is
read-only even if shell execution is enabled. `write_file` creates or fully
replaces a file; `edit_file` changes targeted text; recursive `delete` is
action-only and follows its approval policy. Tool enablement, always-allow
approval, Plan, programmatic tool calling (PTC), and Rubric access are distinct
policies. Review the proposed call before approving it; cancelling a review
returns to the same question, while Alt+Q cancels the outer turn.

## Plans, Goals, and sessions

Use a **Plan** when you want to review an approach before implementation. Use a
**Goal** when you want to specify an outcome and its Success Criteria while
letting Act choose the approach. MIRA retains one current Plan *or* Goal, and
the proposed artifact is reviewed before it replaces existing formal work.
The [Plan and Goal architecture](../ARCHITECTURE_DECISIONS.md#plan-and-goal-architecture)
explains the stages, graph threads, and rubric behavior.

### Develop and run a Plan

```text
/plan Add a search command for saved sessions
```

`/plan` alone enters the same persistent read-only conversation; `/plan <prompt>`
also sends a request. Continue discussing requirements in that mode. When MIRA
prepares a formal Plan, it generates Success Criteria, presents the Plan bubble,
and waits for review. Choose **Revise** and provide feedback to update the
draft; choose **Implement** to switch to Act and start the approved work;
**Close** retains the Plan without running it; **Clear** discards the draft.
`/plan-show` recalls the retained Plan, `/plan-resume` runs an incomplete Plan
in Act, and `/plan-clear` removes the current Plan while leaving history.
Use `/act` to return to ordinary action mode without implementing a Plan.
Completed Plans must be reopened and deliberately implemented again.

### Create and run a Goal

```text
/goal Saved sessions can be found by title and opened from search results
```

`/goal <prompt>` prepares a criteria-only Goal on a separate read-only planning
thread without changing the current Act/Plan mode. Review its Objective and
Success Criteria, then choose **Revise**, **Implement**, **Close**, or **Clear**
in the Goal bubble. Implement starts one Act attempt. `/goal-show` reopens the
retained Goal, `/goal-resume` resumes an incomplete Goal, and `/goal-clear`
removes the current Goal without erasing its historical event. A completed Goal
must be reopened and explicitly implemented again. If rubric grading is
enabled, MIRA evaluates the exact criteria after execution; a cap reached or
paused attempt remains resumable.

### Resume sessions and compact context

MIRA saves sessions under `.mira/_sessions/`. `/new-chat` creates a new saved
chat without deleting the old one. Start MIRA with `--resume` to reopen the
most recent session or `--session <id>` for a specific session:

```bash
mira --resume
mira --session SESSION_ID
```

Pending approval or question interrupts are shown again on reopening. If work
stopped without a checkpointed answer, use explicit `/continue` only after
considering whether an external side effect already happened. `/compact`
immediately applies the active DeepAgents summarization policy to older
context; it does not erase the saved transcript. Use `/context-report` to
inspect current context pressure. See [sessions](../ARCHITECTURE_DECISIONS.md#sessions-and-persistence),
[handoff](../ARCHITECTURE_DECISIONS.md#conversation-handoff), and
[compaction](../ARCHITECTURE_DECISIONS.md#context-management-and-compaction).

## LangGraph Workflows

A MIRA Workflow is an ordinary LangGraph graph. Your Python code owns its
state, nodes, edges, branches, loops, reducers, and input/output mapping. The
running application exposes `application.workflows` for building a graph with
configured MIRA subagents and tools. This is useful when a repeatable task
needs structured intermediate state rather than one free-form agent turn.
Runnable examples are under `.mira/examples/workflows/`; see
[workflow ownership](../ARCHITECTURE_DECISIONS.md#workflows-and-subagents).

### Build a Workflow with a MIRA agent

`mira.agent(...)` creates an independent, workflow-local specialization of a
configured subagent; it does not change the reusable base. The default base is
`general-purpose`. The selected base determines the model. `tools=None`
inherits its tools, while `tools=[]` selects none. Tool replacements use MIRA's
normal resolver. Configure and select another subagent by name if the Workflow
needs a different model.

This graph maps a domain-shaped `topic` into an agent message, then maps the
agent response back into `research`:

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

```

Save this factory in `.mira/workflows/research.py`, then reload and launch it
from the TUI:

```text
/reload
/workflows
/workflow__research topic="pineapples"
```

The file name determines the command name; `/workflows` shows the required and
optional arguments from the graph's `input_schema`. From Python code that
already has a running `application`, the same factory can be invoked directly:

```python
graph = workflow(application.workflows)
result = await graph.ainvoke(
    {"topic": "pineapples"}, context=application.workflows.context
)
```

The dedicated `input_schema` defines a discovered Workflow's launch arguments.
The node does the state mapping explicitly; MIRA does not provide a Workflow
DSL or automatic domain-state adapter. User code also owns parallelism and
`Send()`. If the state already follows the
agent's `messages` contract, direct insertion such as
`graph.add_node("worker", mira.agent())` also works.

For a standalone LangGraph + DeepAgents version, replace `mira.agent(...)`
with `create_deep_agent(model=model, ...)` and omit MIRA's `context_schema` and
execution `context`. The graph topology and state mapping can remain the same.
A standalone DeepAgents agent may have its own `task` tool and nested subagents;
a `mira.agent(...)` worker is specialized from MIRA's configured environment.

### Use the execution context

Nodes can request `Runtime[MiraContext]`, and ordinary LangChain `@tool`
functions receive `ToolRuntime`. Both can use
`runtime.context.tools["..."]` and `runtime.context.agents["..."]`. This is a
capability-bearing API for trusted in-process Python. Direct context-tool
calls implement an already-invoked trusted operation and do not open a second
HITL prompt. Disabled tools, backend rules, unavailable MCP capabilities, and
other hard restrictions still apply. Delegated agents use native DeepAgents
task execution. Intermediate Python values can pass directly between
capabilities without entering the parent model messages; marked nested-call
payloads are redacted by MIRA's trace export policy. Workflows that use these
MIRA context capabilities are intentionally coupled to MIRA.

## OTLP tracing

Optional OTLP tracing exports agent runs to a selected OTLP/HTTP destination.
This differs from `--trace`, which opens a local diagnostic transcript. Install
the optional tracing runtime first:

```bash
pip install "mira[tracing]"
```

### Configure an export profile

MIRA creates `.mira/tracing.yml` once with Phoenix, MLflow, and LangSmith
examples. Add a complete profile there, then select it in `.mira/settings.yml`:

```yaml
# .mira/tracing.yml
profiles:
  corporate:
    endpoint: https://example.com/otel/v1/traces
    headers:
      Authorization: "Bearer ${TRACE_TOKEN}"
      tenant: my-team
    span_attributes:
      deployment.environment: production
```

```yaml
# .mira/settings.yml
tracing:
  enabled: true
  profile: corporate
```

Existing profile registries are not overwritten. Environment references stay
literal in the file and Settings preview; MIRA resolves the selected profile's
in-memory copy when it builds OTLP export. Supply referenced values before
startup or `/reload-runtime`. Phoenix is the recommended local viewer, while
LangSmith and compatible custom destinations use the same profile contract.

### Reload and troubleshoot tracing

`/reload-runtime` closes the current OTel-only LangSmith client and OTel
provider, then builds the selected exporter again. LangSmith supplies the
LangChain and DeepAgents trace tree; MIRA adds OpenInference span kinds,
structured inputs and outputs, model messages, tool attributes, and token
counts before standard OTel batch export. It temporarily sets the LangSmith
callback activation values and restores the previous process values on reload,
disable, or shutdown. Missing optional dependencies appear in Issues with the
install command; exporter connection errors follow OpenTelemetry behavior and
do not stop the agent. See [tracing design](../ARCHITECTURE_DECISIONS.md#tracing-and-diagnostics).

## Safety, storage, and diagnostics

MIRA reports missing Git protection as a startup Issue while keeping the
workspace usable; Settings can initialize a repository if you choose. Dangerous
tools require approval unless explicitly allowed. Error reports are written
under `.mira/_errors/` only for failures, and `/clear-errors` removes them.
`--trace` opens the local diagnostic transcript at `.mira/_logs/mira.log`; this
is separate from OTLP agent tracing. `/runtime`, `/session`, `/tools`,
`/memories`, `/skills`, and `/subagents` inspect current state without a model
request. Contributor rules are in [AGENTS.md](../AGENTS.md), design rationale
in [Architecture Decisions](../ARCHITECTURE_DECISIONS.md), and manual scenarios
in [tests/manual/prompts.md](../tests/manual/prompts.md).
