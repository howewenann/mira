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

MIRA bootstraps `.mira/tracing.yml` once with Phoenix and LangSmith profiles.
Each profile contains only a complete OTLP/HTTP `endpoint` and a `headers`
mapping. Add any number of profiles directly to that file:

```yaml
profiles:
  corporate:
    endpoint: https://example.com/otel/v1/traces
    headers:
      Authorization: "Bearer ${TRACE_TOKEN}"
      tenant: my-team
```

Existing registries are never overwritten or extended automatically.
Environment references stay literal in `tracing.yml` and the Settings preview.
MIRA resolves only the selected profile's in-memory runtime copy at the OTLP
boundary. Supply referenced values before starting or reloading MIRA.

LangChain and DeepAgents runs are instrumented once with OpenInference, which
adds AI-specific span kinds, structured inputs and outputs, messages, tools,
and token attributes to ordinary OpenTelemetry spans. A standard OTel batch
processor sends that same representation to every selected OTLP/HTTP profile;
there is no backend-specific instrumentation. Phoenix is the recommended local
viewer, while LangSmith and custom compatible destinations are transport
profiles in the same registry.

Reload Runtime detaches the current OpenInference instrumentor, drains and
closes its MIRA-owned OTel provider, then rebuilds both with the selected
profile's endpoint and headers. It follows the same full reload path as
`/reload-runtime`. If the extra is missing, tracing is disabled for that runtime
and Issues shows the install command. Exporter connection failures follow
normal OpenTelemetry behavior and do not disable the agent runtime.

MIRA does not enable native LangSmith tracing. If `LANGSMITH_TRACING` or a
supported legacy LangChain tracing switch is already enabled in the launching
environment, MIRA leaves it unchanged and shows a non-fatal duplicate-tracing
warning. Disable the external switch if the configured OpenInference path
should be the only trace source.

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
