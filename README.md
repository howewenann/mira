# MIRA

Minimal Iterative Reasoning Agent.

MIRA is an educational Python CLI coding agent. It is intentionally small and
direct so you can use it, inspect it, and adapt it without wading through a
large framework.

For design rationale and implementation notes, see
[ARCHITECTURE_DECISIONS.md](ARCHITECTURE_DECISIONS.md).

## Quick Start

With Conda:

```powershell
conda env create -f environment.yml
conda activate mira
pip install -e .
mira
```

Or install into an existing compatible Python environment:

```powershell
pip install -e .
mira
```

Running `mira` opens the Textual TUI. Run one prompt and exit with:

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
mira --direct
mira --trace
```

`--direct` is for trusted local setups that need direct LLM HTTP calls with
proxy environment variables ignored and TLS verification disabled.
`--trace` opens a separate diagnostics window for TUI runs.

In the TUI, `/clear-errors` deletes saved error reports for the current
workspace. Chat-clearing commands keep error reports unless this command is run.

## Configuration

MIRA reads environment variables and a workspace `.env` file. Start from
`.env.example`, copy the values you need into `.env`, and keep one provider
block active.

MIRA defaults to an LM Studio-compatible local endpoint:

```dotenv
MIRA_LLM_PROVIDER=lmstudio
MIRA_LLM_MODEL=local-model
MIRA_LLM_BASE_URL=http://localhost:1234/v1
MIRA_LLM_API_KEY=lm-studio
MIRA_LLM_CONTEXT_TOKENS=32768
MIRA_TOOL_OUTPUT_CHARS=240
```

Common `MIRA_LLM_PROVIDER` values include `lmstudio`, `ollama`, `openai`,
`anthropic`, `gemini`, `groq`, and `openrouter`. Optional generation settings
include `MIRA_LLM_TEMPERATURE`, `MIRA_LLM_MAX_TOKENS`, and `MIRA_LLM_TOP_P`.
`MIRA_LLM_CONTEXT_TOKENS` is the configured context-window cap; MIRA uses LM
Studio's loaded context when available and otherwise applies provider profile
limits or this configured cap to DeepAgents.

MIRA does not create or overwrite `.env`. Workspace settings such as Git
protection, dynamic eval subagents, and tool approval behavior live in
`.mira/settings.yml`; change them from the TUI with `/settings`. Dynamic
subagents are disabled by default, so QuickJS eval does not expose its
top-level `task()` helper unless enabled in System Settings. Dynamic response
schemas remain enabled by default for compatibility. Disable them when a local
model cannot reliably complete DeepAgents' structured-output tool protocol;
MIRA then keeps the same full subagents but compiles them ahead of time, so
Eval `responseSchema` requests are rejected before a child model starts.

MIRA's own Python runtime can be different from the environment used by the
agent's `execute` tool. In `/settings`, the Execute Environment section lets a
project run shell commands through the system shell, a Conda env name, a Conda
env path, or a venv path. Venv paths can point at the venv folder or its Python
executable, such as `.venv`, `.venv\Scripts\python.exe`, or `.venv/bin/python`.

When `MIRA_LLM_PROVIDER=lmstudio`, MIRA keeps the UI/provider identity as
`lmstudio:<model>` but sends chat turns through LM Studio's OpenAI-compatible
`/v1` endpoint so DeepAgents can use normal LangChain tool calling. Reasoning
blocks still depend on the fields emitted by the loaded LM Studio model through
that endpoint.

The additional env var field stores variable names only, not host values. Enter
comma-separated names such as `CUDA_HOME, HF_HOME, REQUESTS_CA_BUNDLE` when a
project tool needs them. Empty fields mean no extra names are allowed, and muted
placeholder examples in `/settings` are never saved or applied automatically.
MIRA intentionally does not support "inherit all environment" for `execute`.
When running files created through MIRA's file tools, `execute` uses the project
workspace as its shell working directory, so virtual paths such as `/tmp.py`
should be run as workspace-relative paths such as `python tmp.py`.

## Everyday Use

- Chat in the TUI by running `mira`.
- Use `mira -p "..."` for one-shot terminal output.
- Use `mira -f prompt.md` to read a Markdown file as a one-shot prompt.
- Resume the latest session with `mira -r`.
- Resume a specific session with `mira -s <session-id>`.
- Use `/help` in the TUI to see commands.
- Use `/new-chat` or the chat history `+ New` action to start a fresh saved
  session without deleting the current one.
- Use `/compact` in the TUI to summarize and archive older context immediately
  when changing topics. This explicit command bypasses the agent-selected
  compaction eligibility gate but still keeps DeepAgents' normal recent-context
  retention window.
- Use `/plan` for safe, read-only conversation and implementation planning.
  `write_file`, `edit_file`, `execute`, `task`, and `eval` are hidden from the
  planning agent. Explanations and read-only findings may remain normal chat;
  requests that imply implementation should proactively become plan bubbles
  without requiring a follow-up such as "show me the plan." Material decisions
  should use `ask_user` choices instead of open-ended chat questions. Explicit
  unresolved preferences are asked before direction-dependent research, and a
  selected answer remains part of the original implementation request until a
  structured plan is presented. Use
  Implement to run the plan, Revise to give targeted feedback with the previous
  plan kept in context, or Discard to close it. Plan bubbles include Summary,
  Key Changes, Test Plan, and Assumptions. Test Plan items should name the exact
  tests or checks to create/run and expected results. When a plan is
  implemented, MIRA should run the feasible planned checks after building, or
  say exactly why a check was skipped. If execute is unavailable, MIRA should
  still plan test files/checks and say they were not run. Recent plan bubbles
  are included in lightweight resume context so MIRA can answer follow-ups like
  "show me the previous plan" after switching modes or resuming a session.
  Implement, Revise, and Discard use the same compact one-row button treatment
  as prompt-panel choices. New plan bubbles focus Implement automatically;
  Left/Right moves between actions, `i`/`r`/`d` selects an action, Enter
  activates the focused action, and Escape returns to the prompt.
  Clicking an active plan bubble restores focus to its most recently focused
  action after the prompt or another control has been selected. Discarding a
  plan returns focus to the prompt.
- Use `/act` to return to normal action mode.
- Use `/reload` after changing `.env` or project resources to rebuild the
  active agents without restarting the TUI.

On startup, MIRA checks whether your workspace is covered by Git. If it is not,
MIRA asks whether to create a repository before the agent runs. If you choose to
continue without Git, that choice is remembered for the workspace.

When you approve repository creation, MIRA runs `git init <workspace>` directly
from startup code before building the agent. It does not use the agent's shell
tool approval flow, and it does not stage files or create an initial commit.

## Project Resources

MIRA ships a small default memory plus built-in tools, then layers project
resources from `.mira/` on top. Project resources win when they use the same
memory filename, skill name, subagent name, or tool name.

The overwrite rules are intentionally simple:

- Memories replace by Markdown filename, such as `AGENTS.md`.
- Skills replace by frontmatter `name`, falling back to the folder name.
- Subagents replace by each exported subagent `name`.
- Tools replace by LangChain tool `name`.

During normal use, a project can contain:

```text
your-project/
  .env
  .mira/
    _sessions/
    settings.yml
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

Use these folders to customize MIRA for a project:

- `.mira/memories/*.md`: always-on project context.
- `.mira/skills/<skill>/SKILL.md`: DeepAgents skills with YAML frontmatter.
- `.mira/subagents/*.py`: Python files exporting `SUBAGENTS = [...]`.
- `.mira/tools/*.py`: LangChain tools, including module-level `@tool` objects
  and optional `get_tools(project_backend)`.

In the TUI, use `/memories`, `/skills`, `/subagents`, and `/tools` to inspect
what MIRA loaded and which project resources replaced defaults.

## Features

- Textual TUI with chat, tool calls, tool results, subagent progress, session
  history with a flat new-chat action, and a compact status/dashboard line.
- Live subagent panel in the TUI for running task delegation. Eval-created
  subagents are grouped as `Group 1`, `Group 2`, and so on; raw eval ids are not
  shown. Rows keep fixed task, status, and elapsed-time columns as the terminal
  resizes, with long task text truncated using `...`. While work is active, the
  panel can collapse to an animated summary but cannot be dismissed; new
  subagent activity opens it again. The live chat suppresses separate task
  delegation bubbles and shows subagent progress in the panel instead.
- One-shot terminal mode for scripts or quick prompts.
- Git protection before agent startup.
- Human-in-the-loop approvals for write, edit, eval, task, execute, and project
  tools that need approval, with `/settings` toggles to enable or disable
  built-in dangerous tools and project tools.
- `ask_user` decision prompts prefer concise choices, show options as vertical
  buttons with an open-ended fallback, and scroll only when larger choice sets
  are explicitly needed. Prompt-panel focus highlights the selected button
  itself; ask_user choices span the row so keyboard selection is easy to scan.
  Planning mode uses this tool for every decision that needs a user answer.
- QuickJS eval can call the safe project exploration tools `ls`, `read_file`,
  `glob`, and `grep` through PTC; optional dynamic subagent delegation uses
  QuickJS' top-level `task()` helper when enabled in `/settings`, and
  write/edit/shell tools stay outside that bridge. The nested Response schemas
  setting can make all synchronous subagents text-only for dynamic Eval
  dispatch while retaining their normal tools and middleware.
- Project-specific `execute` environment selection for system shell, Conda, or
  venv commands without saving host env values.
- Planning mode that hides write, shell, eval, and delegation tools while still
  supporting read-only conversation. Implementation intent produces a
  structured plan bubble; resolved bubbles stay as inactive history and only
  the newest unresolved plan is actionable.
- Project-level memories, skills, subagents, and tools.
- Session resume from `.mira/_sessions/`.
- Context pressure display from DeepAgents' own summarization count, plus
  DeepAgents-backed automatic and on-demand context compaction. MIRA fills the
  provider identity omitted by ChatAnyLLM so DeepAgents can validate reported
  token usage; DeepAgents still decides when compaction is eligible. `In` and
  `Out` are cumulative provider token totals; `Ctx` is the latest DeepAgents
  context estimate.
- Default memory plus `grep`, `ask_user`, and `present_plan` tools for search,
  concrete user decisions, and structured plan review. `ask_user` should put
  only the direct question in its prompt text and keep answer choices in
  options; add `(Recommended)` only when there is a real default.

## Sessions

MIRA stores session JSON under `.mira/_sessions/`. Session ids start with a
local timestamp and timezone offset so they sort by creation time, for example:

```text
20260602-171423+0800-a1b2c3d4
```

MIRA keeps replayable user and assistant messages for the chat history UI.
Starting a new chat creates another saved session and switches to it; it does
not clear or overwrite the previous session.
Regular subagent task outputs are stored as session transcript events so they
can be restored and included in resume context. The live bottom subagents panel
itself is not stored or replayed when opening past sessions. Reopening or
resuming a past chat therefore shows durable transcript subagent blocks instead
of the live panel rows that were shown during the original run. Eval-created
subagent detail remains live telemetry, with durable history coming from the
parent eval tool call/result and assistant summary.
Recent structured plan bubbles are also included in model resume context as
compact plan summaries, without replaying raw reasoning or tool event noise.
DeepAgents manages runtime context compaction while MIRA is running, and MIRA
records visible compaction markers and archive paths in the session file. The
TUI `/compact` command uses the same DeepAgents summarization engine and writes
the same checkpoint event as runtime compaction, without adding a user or
assistant turn. It can compact before the agent-selected tool becomes eligible,
but it does not discard the recent context protected by DeepAgents' retention
policy. The status line's `Ctx` value is the latest DeepAgents summarization
count MIRA has observed. `In` and `Out` are cumulative provider usage totals
across every model call in the session, including repeated conversation
history, so `In + Out` is not expected to equal current context after multiple
turns. MIRA
normalizes ChatAnyLLM responses with the missing provider identity before they
enter agent state, allowing DeepAgents to trust reported usage without changing
its compaction thresholds.
Live summary-model output is identified through DeepAgents' structural
`lc_source="summarization"` metadata, so its internal reasoning stays out of the
transcript without guessing from words such as "compact" or "summary."

## Error Reports And Trace

When an unexpected error reaches MIRA's handled one-shot, TUI, or top-level
boundaries, MIRA writes a copy-pasteable report under `.mira/_errors/` before
showing or re-raising the error. Successful runs do not create error reports.

```text
.mira/
  _sessions/
    <session-id>.json
  _errors/
    latest_error.txt
    <session-id>/
      <YYYYMMDD-HHMMSS+ZZZZ-ffffff>.txt
```

The timestamped report is the specific failure artifact. `latest_error.txt` is
only a convenience copy of the most recent report. TUI error messages include
the timestamped report path, and one-shot mode preserves the normal terminal
traceback after writing the report. Error reports are durable diagnostics; use
`/clear-errors` in the TUI to delete them explicitly.

For a live plain-text mirror during TUI runs, start MIRA with `-t` or
`--trace`. Trace mode opens a separate Windows command window that tails a
bounded rotating log at `.mira/_logs/mira.log`. It uses the same plain terminal
spacing as `mira -p`, with sidecar-only color, and shows startup progress, user
prompts, assistant text, tool calls and results, subagent lifecycle, system
messages, tracebacks, and error report paths. The trace log is overwritten for
each trace session and is not durable ordered history; session JSON remains the
authoritative transcript. The trace window is optional diagnostics, and error
reports are written whether or not trace mode is enabled.

## Development

Use the shared Conda environment for checks:

```powershell
conda run -n ai_agents python -m compileall agent cli config runtime session ui
```

Prefer current-checkout commands while developing:

```powershell
conda run -n ai_agents python -m cli.main -p "hello"
```

Run focused tests for changed areas, then `git diff --check` before finishing.
