# MIRA Architecture Decisions

This is MIRA's living design-rationale document. When answering a question about
why MIRA behaves a certain way, read this file first, then verify the current
code. When code changes alter one of these decisions, update this file in the
same change.

## Project Shape

**Decision:** MIRA stays small, direct, and educational.

**Why:** The project is meant to be readable by people learning how a coding
agent is assembled. Small modules and plain control flow are preferred over
clever abstractions.

**Where to check:** `AGENTS.md`, `cli/`, `agent/factory.py`,
`runtime/runner.py`, `ui/app.py`.

**Update this when:** A new abstraction, framework layer, or package boundary
changes how a reader should trace the system.

## DeepAgents And LangGraph Ownership

**Decision:** MIRA uses DeepAgents and LangGraph native behavior for agent
construction, tool calls, subagents, HITL resume, backend routing, permissions,
and runtime context compaction.

**Why:** MIRA should show how the underlying agent stack works instead of
reimplementing it. Local workarounds are kept narrow and tested when a real
library or provider edge case is confirmed.

**Where to check:** `agent/factory.py`, `agent/compaction.py`,
`agent/context_overflow.py`, `runtime/runner.py`, `session/checkpoint.py`.

**Update this when:** MIRA takes ownership of behavior that DeepAgents or
LangGraph used to handle, or when a workaround becomes part of the normal path.

## Startup Flow

**Decision:** Startup builds runtime state in one path: CLI command, Git guard,
config, session store, model metadata, action agent, planning agent, and UI or
one-shot renderer.

**Why:** A single startup shape keeps TUI and one-shot mode consistent. The Git
guard runs before sessions and agents so MIRA does not begin work in an
unprotected workspace by accident.

**Git guard behavior:** When Git protection is enabled, startup first checks
the resolved workspace with `git -C <workspace> rev-parse
--is-inside-work-tree`, with a parent `.git` marker check as a fallback. If the
workspace is not covered by Git and the user approves initialization, MIRA runs
`git init <workspace>` directly through `subprocess.run(...)` in
`cli/git_guard.py`. This happens before agent construction, so it is outside
the normal agent tool/HITL approval path. The initializer only creates the
repository; it does not stage files or create an initial commit.

**Where to check:** `cli/main.py`, `cli/commands.py`, `cli/git_guard.py`,
`config/loader.py`, `config/metadata.py`.

**Update this when:** Startup order changes, a new runtime mode is added, or Git
protection is moved later in the flow.

## Configuration And Settings

**Decision:** Provider configuration comes from environment variables and
workspace `.env`; user-facing workspace settings live in `.mira/settings.yml`.
LM Studio remains the default user-facing provider, but MIRA constructs its
LangChain chat model through AnyLLM's OpenAI-compatible transport so DeepAgents
tool calls use LM Studio's `/v1` server path instead of the native
`lmstudio-python` SDK path. The display identity remains `lmstudio:<model>`.

**Why:** LLM provider details are environment-specific, while Git protection and
tool approval choices are workspace behavior. Keeping these separate makes
settings easier to inspect and safer to change from the TUI.

**Where to check:** `config/loader.py`, `config/llm.py`, `agent/llm.py`,
`config/settings.py`, `ui/widgets/settings_panel.py`.

**Update this when:** A setting moves between `.env` and `.mira/settings.yml`,
new provider variables are introduced, or `/settings` changes what it controls.

## Execute Backend

**Decision:** `execute` is special. When enabled, MIRA switches the project
backend from `FilesystemBackend` to `LocalShellBackend`; when disabled, MIRA
uses the filesystem backend. MIRA keeps `inherit_env=False` and provides a small
allowlisted host environment. Project settings can select the system shell, a
Conda env name, a Conda env prefix, or a venv for `execute` commands without
persisting host env values.

**Why:** Shell execution changes the capability surface of the project backend.
Keeping it behind a setting and HITL approval preserves the normal safer path
while allowing trusted local workflows. The allowlist gives Windows tools enough
path context for standard locations such as `%SystemDrive%`, `%ProgramData%`,
`%AppData%`, and `%LocalAppData%`, but avoids exposing the full user environment
or secrets. Extra variables in `.mira/settings.yml` are names only; values are
read from the current process environment at runtime. Conda modes wrap commands
with `conda run`, and venv mode prepares `PATH` and `VIRTUAL_ENV` for the local
shell backend. MIRA adds custom execute-tool prompt guidance so file-tool
virtual paths such as `/tmp.py` are run from the project shell as
workspace-relative paths instead of host-root absolute paths.

**Where to check:** `agent/resources/__init__.py`, `config/settings.py`,
`agent/factory.py`, `agent/middleware.py`, `ui/app.py`.

**Update this when:** `execute` is exposed by a different backend, default
approval behavior changes, or shell environment inheritance changes.

## Project Resources

**Decision:** MIRA loads bundled defaults first, then project `.mira/` resources.
Project resources replace defaults by memory filename, skill name, subagent
name, or tool name.

**Why:** Defaults make MIRA useful immediately, while project resources let a
workspace customize behavior without editing package files. Name-based
replacement makes overrides explicit and easy to inspect.

**How it works at a high level:**

- Defaults live under `agent/default_resources/` and are mounted read-only at
  `/mira-defaults/...`.
- Project resources live under the workspace's `.mira/` folder and are mounted
  at `/.mira/...`.
- `build_resources()` loads memories, skills, subagents, and tools, then passes
  the final lists to `create_deep_agent(...)`.
- Metadata keeps `source` and `replaces` fields so `/memories`, `/skills`,
  `/subagents`, `/tools`, and `/settings` can show what happened.

**Overwrite rules:**

- Memories load from `*.md` and replace by filename. A project
  `.mira/memories/AGENTS.md` replaces the bundled default `AGENTS.md`; extra
  Markdown files are added as additional memories.
- Skills load from folders containing `SKILL.md`. MIRA display metadata keys
  them by YAML frontmatter `name`, falling back to the folder name. DeepAgents
  receives default skill sources first and project skill sources second, so a
  duplicate skill name follows DeepAgents' later-source-wins behavior.
- Subagents load from Python files exporting `SUBAGENTS = [...]` and replace by
  each subagent's `name`.
- Tools load from module-level LangChain `@tool` objects, optional `TOOLS`, and
  optional `get_tools(project_backend)`. Duplicate tool names inside one file
  keep the first tool. Across layers, project tools replace defaults by tool
  name. A project tool can also replace a known DeepAgents built-in tool name,
  which is shown as `replaces: built-in` when no MIRA default tool already
  occupied that name.
- Disabled project tools stay in metadata for the settings UI but are not
  exposed to the agent.

**Where to check:** `agent/resources/`, `agent/default_resources/`,
`tests/test_resources.py`.

**Update this when:** Resource locations, overwrite rules, display metadata, or
supported export shapes change.

## Tools And HITL

**Decision:** Dangerous built-in tools require approval by default. Project
tools can be enabled or disabled through settings and remain visible in
metadata even when disabled. QuickJS programmatic tool calling is limited to
`ls`, `read_file`, `glob`, and `grep`; subagent delegation uses QuickJS'
top-level `task()` helper, while destructive file tools, shell execution, and
interrupt/control-flow tools stay outside that bridge.

**Why:** Approval prompts make file edits, eval, subagent delegation, and shell
execution transcript-compatible and user-controlled. Keeping disabled project
tools in metadata lets the settings UI manage them without exposing them to the
model.

**Where to check:** `config/settings.py`, `agent/factory.py`,
`agent/middleware.py`, `agent/resources/__init__.py`, `ui/interrupts.py`,
`runtime/runner.py`.

**Update this when:** Approval defaults, interrupt payload handling, or
settings-panel tool behavior changes.

## Planning Mode

**Decision:** Planning mode has a separate agent with project write tools hidden
from the model and blocked by filesystem permissions as a backstop. Structured
plans are created only through the `present_plan` tool and shown as ephemeral
plan bubbles with explicit Implement, Revise, and Discard actions.
Every structured plan includes Summary, Key Changes, Test Plan, and
Assumptions. Planning prompts include an exact content template so Summary names
goal, context, and success criteria; Key Changes name concrete implementation
steps; Test Plan names exact test artifacts, commands/checks, and expected
results; and Assumptions records explicit defaults. When an approved plan is
implemented, MIRA treats the Test Plan as required follow-through: feasible
checks should run after building, while skipped checks must be named with a
reason. Todo/checklist use is encouraged during implementation for multi-step
plans, but it does not replace running the planned checks. When execute is
unavailable, MIRA still plans the test artifacts and says the tests were not
run.
Revise opens a focused feedback prompt, then sends the previous structured plan
and the user's feedback through planning mode so the replacement plan keeps
context. Resolved plan bubbles remain inactive transcript history; only the
newest unresolved plan is actionable.
Recent structured plan events are included in lightweight model resume context
so mode switches and resumed sessions can answer plan follow-ups. Raw reasoning,
tool-call, and tool-result events remain excluded from normal model history.

**Why:** Users need a mode where MIRA can reason about a change without editing
files. Hiding write tools improves model behavior; permissions provide a safety
fallback. Plan execution should be an explicit user action, not an automatic
side effect of leaving planning mode.

**Where to check:** `agent/factory.py`, `agent/plan_policy.py`, `ui/app.py`,
`ui/repl.py`, `tests/test_plan_mode.py`, `tests/test_textual_app.py`.

**Update this when:** Planning mode gains or loses tools, changes how plan
bubbles are presented, resolved, or replayed into model context, or changes its
filesystem permissions.

## Textual TUI And One-Shot Output

**Decision:** The Textual TUI is the primary interactive experience. One-shot
terminal output uses a separate renderer.

**Why:** The TUI can preserve chat order, tool calls, tool results, subagent
progress, settings, and session history in one place. The one-shot renderer
stays simpler for scripts and quick prompts.

TUI-only commands that need live app state stay in `ui/app.py`; for example,
`/settings` persists workspace settings before rebuilding agents, while
`/reload` reloads `.env`, current settings, and project resources before
rebuilding agents without restarting the session.

**Where to check:** `ui/app.py`, `ui/widgets/`, `ui/renderer.py`,
`runtime/*_events.py`, `tests/test_textual_app.py`.

**Update this when:** Rendering responsibility moves, a new UI mode appears, or
tool/subagent events are projected differently.

## Sessions And Compaction

**Decision:** MIRA stores durable session JSON for replayable UI history, while
DeepAgents handles runtime context counting and compaction.

**Why:** Session files should be stable user-facing history after restart.
Runtime compaction is agent-execution behavior and belongs to DeepAgents. MIRA
installs a named `MiraSummarizationMiddleware` subclass built from DeepAgents'
summarization defaults, then observes that middleware's `_count_tokens` result
so the UI can show context pressure. MIRA does not run a parallel dashboard
counter, compute provider prompt tokens, or decide when to compact. Provider
`In` and `Out` usage are cumulative per-call totals, not current context
occupancy.

**Where to check:** `agent/compaction.py`, `session/store.py`,
`session/context.py`, `session/recorder.py`, `session/dashboard.py`,
`runtime/context_usage.py`, `runtime/compaction_filter.py`.

**Update this when:** Session JSON shape changes, compaction ownership changes,
or replay context starts depending on a new source of truth.

## Context Metadata

**Decision:** MIRA resolves model context metadata before turns, sets the
DeepAgents context profile, and shows observed DeepAgents context pressure in
the UI.

**Why:** Providers expose context limits differently. MIRA normalizes the
effective limit so the model profile, dashboard, and overflow handling agree.
Provider `In` and `Out` usage are cumulative session totals and are not used as
current-context occupancy; after multiple turns they include repeated
conversation history and are not expected to add up to the latest `Ctx` value.

**Where to check:** `config/metadata.py`, `cli/commands.py`, `ui/app.py`,
`agent/context_overflow.py`.

**Update this when:** New providers need special metadata handling, context
fallback rules change, or the dashboard changes how context is reported.
