# MIRA

Minimal Iterative Reasoning Agent.

MIRA is a small, educational Python coding agent with a Textual terminal UI,
one-shot prompting, project-specific tools and context, planning, approvals,
and resumable sessions.

For implementation rationale, see
[ARCHITECTURE_DECISIONS.md](ARCHITECTURE_DECISIONS.md).

## Install

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
are documented in `.env.example`.

Workspace settings live in `.mira/settings.yml`. Use `/settings` in the TUI to
manage Git protection, tools and approvals, execution environments, dynamic
subagents, and optional rubric grading.

## Run

Open the interactive TUI:

```powershell
mira
```

Run one prompt and exit:

```powershell
mira --prompt "summarize this project"
mira --file prompt.md
```

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
| Show all commands | `/help` |
| Change workspace settings | `/settings` |
| Start a new saved chat | `/new-chat` |
| Enter planning mode | `/plan` |
| Return to action mode | `/act` |
| Create a graded goal | `/goal <prompt>` |
| Compact older context | `/compact` |
| Reload configuration and resources | `/reload` |
| Repair unavailable custom tools | `/issues` |

Inspection commands include `/runtime`, `/session`, `/tools`, `/memories`,
`/skills`, and `/subagents`. Destructive cleanup commands require confirmation
and are listed in `/help`.

Planning mode is read-only. Review the generated plan, then choose Implement,
Revise, or Discard. Rubric grading and `/goal` are opt-in through `/settings`.

## Project Resources

MIRA loads project customization from `.mira/`:

```text
.mira/
  settings.yml
  memories/          # always-on Markdown context
  skills/            # DeepAgents SKILL.md folders
  subagents/         # Python SUBAGENTS definitions
  tools/             # active custom tools
  examples/tools/    # inert MIRA- and project-runtime examples
```

Project resources override built-in resources with the same name. Use
`/memories`, `/skills`, `/subagents`, and `/tools` to inspect what is active.
Run `/reload` after changing project resources.

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
