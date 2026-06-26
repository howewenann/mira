# MIRA

Minimal Iterative Reasoning Agent v1.0.0.

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
```

Useful startup options:

```text
mira --help
mira --resume
mira --session <session-id>
mira --workspace <path>
mira --direct
```

`--direct` is for trusted local setups that need direct LLM HTTP calls with
proxy environment variables ignored and TLS verification disabled.

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
protection and tool approval behavior live in `.mira/settings.yml`; change them
from the TUI with `/settings`.

## Everyday Use

- Chat in the TUI by running `mira`.
- Use `mira -p "..."` for one-shot terminal output.
- Resume the latest session with `mira -r`.
- Resume a specific session with `mira -s <session-id>`.
- Use `/help` in the TUI to see commands.
- Use `/plan` when you want MIRA to think through a change without editing
  files.
  Implementation-ready planning turns appear as plan bubbles. Use Implement to
  run the plan, Revise to give targeted feedback with the previous plan kept in
  context, or Discard to close it. Plan bubbles include Summary, Key Changes,
  Test Plan, and Assumptions; if execute is unavailable, MIRA should still plan
  test files/checks and say they were not run. Recent plan bubbles are included
  in lightweight resume context so MIRA can answer follow-ups like "show me the
  previous plan" after switching modes or resuming a session.
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

MIRA ships small default resources, then layers project resources from `.mira/`
on top. Project resources win when they use the same memory filename, skill
name, subagent name, or tool name.

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
  history, and a compact status/dashboard line.
- One-shot terminal mode for scripts or quick prompts.
- Git protection before agent startup.
- Human-in-the-loop approvals for write, edit, eval, task, execute, and project
  tools that need approval.
- Planning mode that hides and blocks project write tools, with explicit plan
  bubbles for implementation-ready plans. Resolved plan bubbles stay as inactive
  history; only the newest unresolved plan is actionable.
- Project-level memories, skills, subagents, and tools.
- Session resume from `.mira/_sessions/`.
- Context pressure display from DeepAgents' own summarization count, plus
  DeepAgents-backed context compaction. `In` and `Out` are cumulative provider
  token totals; `Ctx` is the latest DeepAgents context estimate.
- Default `grep`, `ask_user`, and `present_plan` tools for search, concrete
  user decisions, and structured plan review.

## Sessions

MIRA stores session JSON under `.mira/_sessions/`. Session ids start with a
local timestamp and timezone offset so they sort by creation time, for example:

```text
20260602-171423+0800-a1b2c3d4
```

MIRA keeps replayable user and assistant messages for the chat history UI.
Recent structured plan bubbles are also included in model resume context as
compact plan summaries, without replaying raw reasoning or tool event noise.
DeepAgents manages runtime context compaction while MIRA is running, and MIRA
records visible compaction markers and archive paths in the session file. The
status line's `Ctx` value is the latest DeepAgents summarization count MIRA has
observed. `In` and `Out` are cumulative provider usage totals across every
model call in the session, including repeated conversation history, so `In +
Out` is not expected to equal current context after multiple turns.

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
