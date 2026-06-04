# MIRA

Minimal Iterative Reasoning Agent v1.0.0.

MIRA is an educational Python CLI coding agent. The code is intentionally small
and direct so it is easy to read, change, and learn from.

## Project Philosophy

- Prioritise clarity over cleverness.
- Keep modules small and avoid unnecessary abstractions.
- Use LangChain and DeepAgents primitives where they fit.
- Make the code readable without comments; structure should explain intent.
- Prefer code a junior developer can trace, modify, and explain to someone else.

## Quick Start

With Conda, create a Python 3.12 environment before installing MIRA:

```powershell
conda env create -f environment.yml
conda activate mira
mira --help
mira
```

Or install into an existing compatible Python environment:

```powershell
pip install -e .
mira --help
mira
```

MIRA defaults to an LM Studio-compatible local endpoint at
`http://localhost:1234/v1`.
Running `mira` opens the Textual TUI. Use `mira --prompt "..."` for a one-shot
plain terminal run.

On startup, MIRA checks whether your workspace is covered by Git. If it is not,
MIRA asks whether to create a repository before the agent can run. If you choose
to continue without Git, that choice is remembered for the workspace.

## Configuration

MIRA reads configuration from environment variables and from a `.env` file in
your workspace. Start with `.env.example`, copy the values you want into your
own `.env`, and keep exactly one LLM provider active.

You can set values in your shell before running `mira`:

```powershell
$env:MIRA_LLM_PROVIDER = "lmstudio"
$env:MIRA_LLM_MODEL = "your-loaded-model-name"
$env:MIRA_LLM_BASE_URL = "http://localhost:1234/v1"
$env:MIRA_LLM_API_KEY = "lm-studio"
mira
```

Or put them in a `.env` file in the workspace directory:

```dotenv
MIRA_LLM_PROVIDER=lmstudio
MIRA_LLM_MODEL=your-loaded-model-name
MIRA_LLM_BASE_URL=http://localhost:1234/v1
MIRA_LLM_API_KEY=lm-studio
MIRA_TOOL_OUTPUT_CHARS=240
MIRA_SESSION_MAX_CHARS=40000
MIRA_SESSION_RECENT_MESSAGES=10
MIRA_SESSION_SUMMARY_MAX_CHARS=6000
```

`MIRA_LLM_PROVIDER` is the selector for `langchain-anyllm`. Common values
include `lmstudio`, `ollama`, `openai`, `anthropic`, `gemini`, `groq`, and
`openrouter`; use `anthropic` for Claude models. MIRA also accepts optional
generation values: `MIRA_LLM_TEMPERATURE`, `MIRA_LLM_MAX_TOKENS`, and
`MIRA_LLM_TOP_P`.

MIRA does not create or overwrite `.env`. If you already have one, use
`.env.example` as a reference and update your own file by hand. Old
`MIRA_LMSTUDIO_*` variables still work for LM Studio when no `MIRA_LLM_*`
provider config is present, but the `MIRA_LLM_*` names are preferred.

These values are loaded in `config/loader.py` and normalized in `config/llm.py`
before being passed to `ChatAnyLLM` in `agent/llm.py`.
`MIRA_TOOL_OUTPUT_CHARS` controls how many characters of each tool result are
shown in the terminal, including the final tool output shown for subagents.
Tool output is shown on one line; set the value to `0` to show full output.
For LM Studio, MIRA also asks the local LM Studio SDK for the loaded model's
context length so the TUI can show context pressure as a colored bar.

Session resume uses the same configured LLM to title sessions and compact long
sessions into durable continuation state. Titles are generated after the first
completed response, refreshed after the next early turn, and then updated
periodically as the session changes. `MIRA_SESSION_MAX_CHARS` controls when
stored messages become long enough to compact, `MIRA_SESSION_RECENT_MESSAGES`
controls how many recent messages stay verbatim, and
`MIRA_SESSION_SUMMARY_MAX_CHARS` caps the stored structured summary.

If you do not set them, MIRA uses:

```text
MIRA_LLM_PROVIDER=lmstudio
MIRA_LLM_MODEL=local-model
MIRA_LLM_BASE_URL=http://localhost:1234/v1
MIRA_LLM_API_KEY=lm-studio
MIRA_TOOL_OUTPUT_CHARS=240
MIRA_SESSION_MAX_CHARS=40000
MIRA_SESSION_RECENT_MESSAGES=10
MIRA_SESSION_SUMMARY_MAX_CHARS=6000
```

## How MIRA Works

MIRA is split into a few small pieces:

- `cli/` starts the app, loads a session, and chooses one-shot or TUI mode.
- `config/` reads `.env` and normalizes LLM settings.
- `agent/factory.py` builds the action agent and the planning agent.
- `agent/resources/` gathers default and project memories, skills, subagents, and tools.
- `runtime/runner.py` streams one agent turn and handles HITL approvals.
- `ui/app.py` is the Textual TUI shell for interactive mode.
- `ui/dialogs.py` contains the modal prompts used by the TUI.
- `ui/interrupts.py` normalizes approval and `ask_user` interrupt payloads.
- `ui/widgets/` contains the chat log, session list, prompt input, and status bar.
- `ui/repl.py` keeps slash-command and planning-mode state helpers.
- `ui/renderer.py` is the plain one-shot renderer used by `--prompt`.
- `session/` stores durable session JSON, resume context, and checkpoints.

## Project Folder Map

```text
mira/
  .env.example
  README.md
  pyproject.toml
  agent/
    factory.py              # builds DeepAgents agents
    llm.py                  # creates ChatAnyLLM
    plan_policy.py          # planning-mode rules
    default_resources/      # bundled MIRA defaults
    resources/              # loads memories, skills, subagents, and tools
    tools/                  # tool metadata helpers
  cli/
    main.py
    commands.py
  config/
    loader.py
    llm.py
  runtime/
    runner.py               # streams one agent turn
    *_events.py             # handles stream event types
  session/
  ui/
    app.py
    dialogs.py
    interrupts.py
    repl.py
    styles/
      mira.tcss
    widgets/
      chat_log.py
      prompt_box.py
      session_history.py
      status_bar.py
    renderer.py
  tests/
```

The main startup path is:

```text
cli/main.py -> cli/commands.py -> agent/factory.py -> agent/resources/ -> create_deep_agent(...)
```

Resource loading is kept one-step-per-file:

```text
agent/resources/__init__.py
  -> memories.py
  -> skills.py
  -> subagents.py
  -> tools.py
```

For example, the default regex `grep` starts in
`agent/default_resources/tools/regex_grep.py`, is loaded by
`agent/resources/tools.py`, is placed in the resource bundle by
`agent/resources/__init__.py`, and is passed to DeepAgents in
`agent/factory.py`.

## Project Resources

MIRA ships small default resources, then layers project resources from `.mira/`
on top. Project resources win when they use the same memory filename, skill
name, subagent name, or tool name.

During normal use, a project folder looks like this:

```text
your-project/
  .env
  .mira/
    _sessions/
    README.md
    git_safety.json
    memories/
      AGENTS.md
    skills/
      example-skill/
        SKILL.md
    subagents/
      example_subagent.py
    tools/
      example_tool.py
```

MIRA creates the `.mira` resource examples if they are missing and never
overwrites existing files. `_sessions/` stores durable session JSON, and
`git_safety.json` remembers when you choose to continue in a workspace without Git. Delete `git_safety.json` if
you want MIRA to ask again.

Use `.mira/memories/*.md` for always-on project context. The bundled default
memory is only `AGENTS.md`; `.mira/memories/AGENTS.md` replaces it. Additional
Markdown files in `.mira/memories/` are added as extra memories.

Use `.mira/skills/<skill>/SKILL.md` for DeepAgents skills. Skills need YAML
frontmatter with `name` and `description`; a project skill with the same `name`
as a default skill takes priority.

Use `.mira/subagents/*.py` for DeepAgents subagents. Each file should export
`SUBAGENTS = [...]`. MIRA accepts the same subagent objects DeepAgents accepts,
including dictionary specs, compiled subagents, and async subagents. A project
subagent with the same `name` as a default subagent takes priority.

Use `.mira/tools/*.py` for LangChain tools. Each file can export `TOOLS = [...]`
or `get_tools(project_backend) -> list[...]` for tools that need workspace file
access. MIRA includes a default regex-capable `grep` tool that replaces
DeepAgents' literal-only `grep`, plus `ask_user`, which lets the model pause
for a concrete multiple-choice user decision. A project tool with the same
`name` takes priority.

In the TUI, use `/memories`, `/skills`, `/subagents`, and `/tools` to inspect
the final resources MIRA loaded and see which project resources replaced
defaults.

## Session Resume

MIRA v1.0.0 stores each session in one JSON file under `.mira/_sessions/`. The
filename starts with a local timestamp and timezone offset so the folder sorts
by creation time, for example `20260602-171423+0800-a1b2c3d4.json`. The file
starts with a generated `title` so you can identify it quickly. MIRA refreshes
that title after early follow-up work and then periodically during longer
sessions. The file also stores the workspace, turn count, dashboard stats,
context policy, optional compacted summary, and recent messages.

Resume the latest session:

```powershell
mira --resume
mira -r
```

Resume a specific session id:

```powershell
mira --session 20260602-171423+0800-a1b2c3d4
mira -s 20260602-171423+0800-a1b2c3d4
```

Older UUID-style session files and UTC `Z` timestamp ids still work with
`--session`; new sessions use the local timestamped id by default.

Short sessions keep exact user and assistant messages. When stored message
content exceeds `MIRA_SESSION_MAX_CHARS` (default `40000`), MIRA asks the
configured LLM to compact older messages into structured continuation state and
keeps the most recent `MIRA_SESSION_RECENT_MESSAGES` messages (default `10`)
verbatim. The compacted state is capped by `MIRA_SESSION_SUMMARY_MAX_CHARS`
(default `6000`).

DeepAgents may manage its own runtime memory while MIRA is running, but the
session JSON is MIRA's durable source of truth after restart.

The Textual TUI shows a compact dashboard line above the chat:
mode, run state, model, context bar with percent and size, input/output tokens,
turns, and duration. The same dashboard values are stored in the session JSON
under `dashboard`.

## Plan Mode

Use `/plan` when you want MIRA to think through a change without editing files.
In planning mode, `write_file` and `edit_file` are hidden from the model and
blocked by filesystem permissions as a backstop.

```text
/plan
write a file called test.txt with the content 'hello world'
/plans
/act
write a file called test.txt with the content 'hello world'
```

`/plans` shows clean plans saved in memory for the current TUI session. When you run
`/act`, MIRA includes the latest saved plan in the next action request once,
then clears that pending plan context.

## Tool Calls And Subagents

When MIRA delegates work, the main agent calls the `task` tool. In the Textual
TUI, normal chat, tool calls, tool results, and subagent progress all stay in
the central scrollable log so their order is preserved. Each delegated worker
appears once under its own block with a readable suffix, for example
`subagent - general-purpose [luna]`. MIRA first shows a short delegation entry,
then each subagent block shows the request and an animated `RUNNING` status.
When it finishes, the block switches to `DONE` and shows that subagent's final
output. The final output follows `MIRA_TOOL_OUTPUT_CHARS`; set it to `0` when
you want the full result.

When MIRA is blocked on a specific user decision, it can call `ask_user` to show
a framed multiple-choice prompt. The final choice is always open-ended:
`Tell MIRA what to do differently`. Normal assistant replies are not converted
into prompts; if MIRA ends with a question, answer it in the next turn.
