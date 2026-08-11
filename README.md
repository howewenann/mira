# MIRA

Minimal Iterative Reasoning Agent.

MIRA is a small, educational general-purpose Python agent with a Textual terminal UI,
one-shot prompting, project-specific tools and context, planning, approvals,
and resumable sessions.

For implementation rationale, see
[ARCHITECTURE_DECISIONS.md](ARCHITECTURE_DECISIONS.md).

## Install

MIRA requires Python 3.11 or newer. Its agent runtime is pinned to
DeepAgents 0.7.3 and `langchain-quickjs` 0.3.5.

With Conda:

```powershell
conda env create -f environment.yml
conda activate mira
pip install -e .
```

Or install into an existing compatible Python environment:

```powershell
pip install -e .
```

## Configure

On first run MIRA creates `.mira/models.yml` with a schema guide and two
commented examples. Add named model profiles there, put secrets in `.env`, and
reference them as `${NAME}`. MIRA resolves only that syntax; unresolved,
malformed, `${env:NAME}`, and `$${NAME}` references appear in Issues.

Open `/models` to select Main and manage Rubric, Summarization, context limits,
and subagents. Fresh workspaces show Main as `unset`. Null secondary assignments
inherit Main and display the effective profile, such as `claude (default)`,
without storing a fake assignment. The usable context is the smaller of the
Settings cap and trustworthy provider/model metadata.

Workspace settings live in `.mira/settings.yml`. Use `/settings` in the TUI to
manage Git protection, tools and approvals, execution environments, dynamic
eval subagents, optional planning todos, and rubric grading.

## Run

Open the interactive TUI:

```powershell
mira
```

Run one prompt and exit:

```powershell
mira --prompt "summarize this project"
mira --file task.txt
mira --prompt "add focused tests" --rubric "The requested tests pass."
mira --file task.txt --rubric-file criteria.txt
```

Task text/file options and rubric text/file options are mutually exclusive.
Files must be readable, non-empty UTF-8 text. One-shot exits with `0` on
success, `1` on runtime failure, `2` on invalid input, and `3` when an
invocation rubric remains unsatisfied.

Useful startup options:

```text
mira --help
mira --resume
mira --session <session-id>
mira --workspace <path>
mira --trace
```

`--direct` disables proxy use and TLS verification for the current process. Use
it only with a trusted local endpoint.

## TUI Basics

| Action | Shortcut or command |
| --- | --- |
| Submit a prompt | Enter |
| Insert a newline | Shift+Enter |
| Copy selected chat or prompt text | Ctrl+C |
| Cancel active work or quit | Alt+Q |
| Complete a slash command | Type `/`, then use Up/Down and Enter |
| Reference a local file | Type `@`, then use Up/Down and Enter |
| Show key bindings, autocomplete, usage notes, and commands | `/help` |
| Change workspace settings | `/settings` |
| Manage model assignments | `/models` or the footer model button |
| Enter Plan mode or send its first prompt | `/plan`, `/plan <prompt>` |
| Return to action mode | `/act` |
| Create an outcome-focused Goal | `/goal <prompt>` |
| Compact older context | `/compact` |
| Reload configuration/resources and rebuild agents | `/reload` |
| Reload the full runtime including MCP | `/reload-runtime` |
| View configuration/resource Issues | `/issues` |
| Open MCP server status and controls | `/mcp` |
| List reusable local and MCP prompts | `/prompts` |

Inspection commands include `/runtime`, `/session`, `/tools`, `/memories`,
`/skills`, and `/subagents`. Destructive cleanup commands require confirmation
and are listed in `/help`.

Slash commands autocomplete at the start of the prompt. Active tool names,
enabled subagents, local project files, and MCP resources autocomplete after
`@` anywhere in the prompt. Selecting a TOOL or SUBA removes the temporary `@`;
a subagent inserts text such as `general-purpose subagent`. FILE and RSRC
entries remain `@` references. Paths containing spaces use quoted
mentions such as `@"docs/design notes.md"`. Local file references guide the
agent to use an explicitly requested or appropriate available read-only tool;
the file contents are not embedded into the prompt automatically.

Plan mode is a continuous read-only conversation that can present one durable
Plan. Goals retain an Objective and Success Criteria while leaving the approach
to Act. Use the corresponding `-show`, `-resume`, and `-clear` commands shown in
`/help`; MIRA retains only one current Plan or Goal.

## Project Resources

MIRA loads project customization from `.mira/`:

```text
.mira/
  models.yml        # ordered AnyLLM model profiles
  settings.yml
  mcp/
    mcp.json         # active MCP configuration
    example.json     # inert stdio and HTTP examples
    schema.json      # supported configuration contract
  prompts/           # recursive reusable Mustache prompt files
  memories/          # always-on Markdown context
  skills/            # DeepAgents SKILL.md folders
  subagents/         # Python SUBAGENTS definitions
  tools/             # active custom tools
  examples/tools/    # inert MIRA- and project-runtime examples
```

Prompt paths flatten to commands with `__` between suffix-free path components.
For example, `prompts/review/python.md` becomes `/prompt__review__python`.
Collisions are excluded and reported in Issues. Project resources override
built-in resources with the same name. Use
`/memories`, `/skills`, `/subagents`, and `/tools` to inspect what is active.
Run `/reload` after changing project resources or non-MCP model/settings
configuration.

### MCP and reusable prompts

MIRA bootstraps an empty active configuration at `.mira/mcp/mcp.json`.
`example.json` contains inert stdio and HTTP configurations to copy, and
`schema.json` documents the exact accepted keys. `example.json` is never
loaded. Run `/reload-runtime` after MCP configuration changes.

MCP string values use the same `${NAME}` resolver as model profiles. For
example, an HTTP header can use
`"Authorization": "Bearer ${MCP_TOKEN}"` without storing the resolved
secret in `mcp.json`. Start MIRA from an environment containing the variable,
or define it in the workspace `.env`; a missing variable fails only that MCP
server with a clear error.

Local stdio and remote Streamable HTTP servers require approval before first
use. Server and tool enablement, persistent approval, and Plan access live in
`/settings`. Standards-compliant browser OAuth is available from the MCP panel;
token state is stored locally under `~/.mira/_state/mcp-tokens/`.

MCP tools use names such as `mcp__local__search`. Fixed text resources appear
in `@` completion as `@mcp__<server>__<exact-uri>`; selecting one attaches its
identity, and the agent reads it on demand. MCP prompts appear as
`/mcp__<server>__<prompt>`. Any readable UTF-8 file recursively under
`.mira/prompts/` becomes a flattened `/prompt__...` command and uses Mustache variables
as required positional arguments. MCP prompts whose arguments are all required
also use positional values. If an MCP prompt has any optional argument, pass
every supplied value as `name=value`; quote values containing spaces. Command
lists keep the compact `<required> [optional]` signature in both cases.

Standard LangChain `@tool` functions run inside MIRA, so their imports must be
installed in MIRA's Python environment. A bad project tool file is isolated and
kept out of the agent; `/issues` shows the exact install command or project-tool
guidance, while `/reload` retries every failed file. See
`.mira/examples/tools/mira_runtime_tool.py`.
Ordinary exceptions raised while an enabled workspace tool runs become error
results the agent can inspect and recover from; graph interrupts and explicit
turn cancellation keep their native control flow.

Use `mira_tool_api.project_tool` when a function body must run in the configured
project Execute Environment. Keep project-only imports inside that function;
MIRA still imports the file to discover it. The project environment needs
neither LangChain nor MIRA installed. See
`.mira/examples/tools/project_runtime_tool.py`.

## Safety and Local Data

- MIRA checks for Git protection before allowing agent work in a workspace.
- Dangerous tools use human approval unless allowed explicitly in `/settings`.
- `write_file` creates or fully replaces a file; use `edit_file` for targeted
  changes. Recursive `delete` follows the configured approval policy.
- Sessions are stored under `.mira/_sessions/` and can be resumed with `-r` or
  `-s <session-id>`.
- Error reports are stored under `.mira/_errors/`; `/clear-errors` removes them.
- `--trace` opens a live diagnostic transcript at `.mira/_logs/mira.log`.

Do not commit `.env` files, credentials, session data, or diagnostic logs.

## Development

Repository guidance is in [AGENTS.md](AGENTS.md); manual scenarios are in
`tests/manual/`.
