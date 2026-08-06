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

Copy the values you need from `.env.example` into a workspace `.env`. MIRA
does not create or overwrite this file.

The default configuration targets LM Studio:

```dotenv
MIRA_LLM_PROVIDER=lmstudio
MIRA_LLM_MODEL=local-model
MIRA_LLM_BASE_URL=http://localhost:1234/v1
MIRA_LLM_API_KEY=lm-studio
MIRA_LLM_CONTEXT_TOKENS=32768
MIRA_TOOL_OUTPUT_CHARS=240
```

Common providers include `lmstudio`, `ollama`, `openai`, `anthropic`, `gemini`,
`groq`, and `openrouter`. Provider examples and optional generation settings
are documented in `.env.example`. Strict JSON `MIRA_LLM_MODEL_KWARGS` values
pass provider-specific generation controls to AnyLLM; optional
`MIRA_RUBRIC_LLM_*` values can select a separate rubric grader.

Workspace settings live in `.mira/settings.yml`. Use `/settings` in the TUI to
manage Git protection, tools and approvals, execution environments, dynamic
subagents, optional planning todos, and rubric grading.

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

`--prompt/-p` and `--file/-f` are mutually exclusive task inputs.
`--rubric` and `--rubric-file` are mutually exclusive, invocation-only rubric
inputs and require a task input. File inputs accept any readable, non-empty
UTF-8 text file regardless of extension; literal and file inputs cannot be
empty or whitespace-only. Invocation rubrics use the configured rubric
iteration cap without changing the saved workspace rubric setting.

One-shot exit codes are `0` for success (and rubric satisfaction when supplied),
`1` for runtime/provider/execution failure, `2` for invalid arguments or input
files, and `3` when the supplied rubric remains unsatisfied at the iteration
limit.

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
| Return focus to the prompt | Escape |
| Complete a slash command | Type `/`, then use Up/Down and Enter |
| Reference a local file | Type `@`, then use Up/Down and Enter |
| Show all commands | `/help` |
| Display a fresh MIRA splash | `/mira` |
| Change workspace settings | `/settings` |
| Start a new saved chat | `/new-chat` |
| Enter Plan mode or send its first prompt | `/plan`, `/plan <prompt>` |
| Return to action mode | `/act` |
| Show, resume, or clear the current Plan | `/plan-show`, `/plan-resume`, `/plan-clear` |
| Create an outcome-focused Goal | `/goal <prompt>` |
| Show, resume, or clear the current Goal | `/goal-show`, `/goal-resume`, `/goal-clear` |
| Compact older context | `/compact` |
| Reload configuration and resources | `/reload` |
| Repair unavailable custom tools | `/issues` |
| Open MCP server status and controls | `/mcp` |
| List reusable local and MCP prompts | `/prompts` |

Inspection commands include `/runtime`, `/session`, `/tools`, `/memories`,
`/skills`, and `/subagents`. Destructive cleanup commands require confirmation
and are listed in `/help`.

Slash commands autocomplete at the start of the prompt. Active tool names,
local project files, and MCP resources autocomplete after `@` anywhere in the
prompt. Selecting a tool removes the temporary `@` and inserts its plain name;
files and resources remain `@` references. Paths containing spaces use quoted
mentions such as `@"docs/design notes.md"`. Local file references guide the
agent to inspect files through its normal `read_file` tool—the file
contents are not embedded into the prompt automatically.

Plan mode is a continuous read-only conversation. Discuss and investigate
normally; when the work is decision-complete, MIRA generates Success Criteria
before presenting one durable Plan with Implement, Revise, and Close actions.
`/plan-show` reopens the exact retained Plan, `/plan-resume` continues incomplete
work, and `/plan-clear` removes it without erasing transcript history. Rubric
grading changes only execution-time evaluation, not Plan construction.

Plan and Goal are alternative durable formal-work artifacts: a Plan contains an
Objective, approach, and Success Criteria; a Goal contains only an Objective
and Success Criteria, leaving the approach to the Act agent. MIRA retains one
current Plan or Goal, never both. Replacing incomplete formal work requires
confirmation, and replacement occurs only after the new artifact is presented.

`/goal <prompt>` works with rubric grading on or off and presents Implement,
Revise, and Close actions. `/goal-show` reopens the exact Goal, `/goal-resume`
continues incomplete work, and `/goal-clear` removes it without deleting
history. Successful non-rubric attempts are agent-declared; rubric-enabled
attempts complete only when rubric-verified. A Goal never contains a hidden
implementation Plan; agents reopen it through `show_goal`.

## Project Resources

MIRA loads project customization from `.mira/`:

```text
.mira/
  settings.yml
  mcp/
    mcp.json         # active MCP configuration
    example.json     # inert stdio and HTTP examples
    schema.json      # supported configuration contract
  prompts/           # top-level reusable Mustache prompt files
  memories/          # always-on Markdown context
  skills/            # DeepAgents SKILL.md folders
  subagents/         # Python SUBAGENTS definitions
  tools/             # active custom tools
  examples/tools/    # inert MIRA- and project-runtime examples
```

Project resources override built-in resources with the same name. Use
`/memories`, `/skills`, `/subagents`, and `/tools` to inspect what is active.
Run `/reload` after changing project resources.

### MCP and reusable prompts

MIRA bootstraps an empty active configuration at `.mira/mcp/mcp.json`.
`example.json` contains inert stdio and HTTP configurations to copy, and
`schema.json` documents the exact accepted keys. `example.json` is never
loaded. Run `/reload` after changes.

MCP string values can read explicit process environment variables with
`${env:NAME}`. For example, an HTTP header can use
`"Authorization": "Bearer ${env:MCP_TOKEN}"` without storing the resolved
secret in `mcp.json`. Start MIRA from an environment containing the variable,
or define it in the workspace `.env`; a missing variable fails only that MCP
server with a clear error.

Both local stdio servers and remote Streamable HTTP endpoints require approval
before first use. `Allow` lasts for the current MIRA process, `Deny` leaves the
server enabled but unused for that process, and `Always allow` persists approval
for a hash of the exact configuration. A configuration change therefore
requires approval again. Server enablement and per-tool enable, approval, and
Plan-access choices live in `/settings`.

OAuth-protected HTTP servers need no authentication field. After normal server
approval, a standards-compliant OAuth challenge appears as `Login required` in
the MCP panel. Select `Login` there to open the browser. MIRA stores access,
refresh, expiry, and dynamic client state outside the project under
`~/.mira/_state/mcp-tokens/`, locally in plaintext. This first version supports
standards-compliant browser OAuth only, not provider-specific login or
device-code flows.

MCP tools use names such as `mcp__local__search`. Fixed text resources appear
in `@` completion as `@mcp__<server>__<exact-uri>`; selecting one attaches its
identity, and the agent reads it on demand. MCP prompts appear as
`/mcp__<server>__<prompt>`. Any readable top-level UTF-8 file under
`.mira/prompts/` becomes `/prompt__<filename-stem>` and uses Mustache variables
as required positional arguments. Double-quote arguments containing spaces.

Initial MCP support intentionally covers fixed text resources only. It does
not support resource templates, resource completion, binary materialisation,
subscriptions, or `listChanged`; local prompt discovery is not recursive.

Standard LangChain `@tool` functions run inside MIRA, so their imports must be
installed in MIRA's Python environment. A bad project tool file is isolated and
kept out of the agent; `/issues` can install all detected missing packages into
MIRA and reload, while `/reload` retries every failed file. See
`.mira/examples/tools/mira_runtime_tool.py`.

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

Use the shared development environment and run focused tests for changed code:

```powershell
conda run -n ai_agents python -m unittest tests.test_textual_app
conda run -n ai_agents python -m compileall agent cli config runtime session ui tests
git diff --check
```

Run the current checkout directly when smoke testing:

```powershell
conda run --no-capture-output -n ai_agents python -m cli.main
```

Repository guidance is in [AGENTS.md](AGENTS.md). Manual scenarios are in
`tests/manual/`.
