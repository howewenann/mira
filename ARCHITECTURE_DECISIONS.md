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

## Error Reports And Trace Diagnostics

**Decision:** MIRA treats the TUI as a friendly display layer, the trace
sidecar as an optional live plain-text stream, and automatic error reports as
durable failure artifacts.

**Why:** Some TUI exceptions are intentionally caught and rendered as concise
messages, so uncaught-exception hooks are not enough. Reports are written at
the boundaries that already catch one-shot and TUI turn failures, with a small
top-level backup for unexpected escaping exceptions. Reports use the current
session id whenever one exists and are only created when an exception happens.
They are durable diagnostics, not chat history, so `/clear-chat` and
`/clear-all-chats` leave them in place; `/clear-errors` is the explicit TUI
command for deleting `.mira/_errors/`.
The trace sidecar mirrors visible TUI activity such as startup progress, user
prompts, coalesced assistant text, tool calls, tool results, subagent
lifecycle, and system messages through a bounded current-session diagnostics
log. Trace transcript formatting is shared with one-shot terminal output, while
sidecar color remains display-only in the log tailer. The trace sidecar is not
the authoritative ordered transcript; saved session JSON is. It remains
optional, so MIRA keeps running if that window cannot open or is closed, and
normal non-trace TUI runs do not create live trace logs solely for successful
activity.

**Where to check:** `runtime/error_report.py`, `runtime/diagnostics.py`,
`runtime/trace_stream.py`, `ui/terminal_transcript.py`, `cli/commands.py`,
`ui/app.py`.

**Update this when:** Error artifact layout, reporting boundaries, diagnostic
log behavior, or trace-window behavior changes.

## Configuration And Settings

**Decision:** Provider configuration comes from environment variables and
workspace `.env`; user-facing workspace settings live in `.mira/settings.yml`.
Immutable launch options are process-local and overlay freshly loaded values to
form the effective runtime configuration. `/reload` reloads environment and
workspace configuration, then reapplies the same launch options. Launch options
are never recovered from the previous effective config or persisted in settings
or sessions. `--direct` is currently the only launch option in this layer;
trace-window state remains separate.
The active runtime state is derived from effective configuration: model
metadata, model identity, action/planning agents, normalized resource
projections, and a sanitized runtime snapshot. Focused read-only commands expose
one section at a time without loading configuration, constructing a model, or
checking connectivity: `/runtime`, `/tools`, `/memories`, `/skills`, and
`/subagents`. Launch-scoped flags are displayed as rows in the Runtime table so
their process scope remains visible beside the effective connection state.
`/reload` builds replacement configuration, metadata, both agents, resource
projections, and the snapshot before replacing active runtime references, then
displays a short confirmation. Endpoint display is allowlisted and strips URL
credentials, query strings, and fragments; API keys and arbitrary config values
never enter the snapshot or inspection output.
LM Studio remains the default user-facing provider, but MIRA constructs its
LangChain chat model through AnyLLM's OpenAI-compatible transport so DeepAgents
tool calls use LM Studio's `/v1` server path instead of the native
`lmstudio-python` SDK path. The display identity remains `lmstudio:<model>`.

**Why:** LLM provider details are environment-specific, while Git protection and
tool approval choices are workspace behavior. Dynamic eval subagents are also
workspace behavior: they let JavaScript eval spawn subagents through
QuickJS' top-level `task()` helper, so MIRA keeps them disabled by default and
requires an explicit System Settings toggle. Dynamic response schemas default
on for compatibility, but can be disabled independently for models that do not
reliably complete the synthetic structured-output tool protocol. In that mode,
MIRA materializes every raw synchronous subagent with its inherited model,
tools, middleware, skills, permissions, and interrupts, then passes it to
DeepAgents as a `CompiledSubAgent`. A compiled `general-purpose` replaces the
auto-added raw one by name. DeepAgents therefore rejects a dynamic
`responseSchema` before starting the child while ordinary text delegation and
static response formats continue to work. Keeping these choices in workspace
settings makes them inspectable without changing QuickJS or installed packages.

**Where to check:** `config/loader.py`, `config/runtime.py`, `config/llm.py`,
`agent/llm.py`, `config/settings.py`, `cli/commands.py`, `ui/app.py`,
`ui/runtime_snapshot.py`, `agent/subagent_compilation.py`,
`ui/widgets/settings_panel.py`.

**Update this when:** A value moves between reloadable and launch-scoped
configuration, a setting moves between `.env` and `.mira/settings.yml`, new
provider variables are introduced, or `/settings` changes what it controls.

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
Bundled defaults stay minimal: default memory plus MIRA's built-in project
tools. Project resources replace defaults by memory filename, skill name,
subagent name, or tool name.

**Why:** Defaults make MIRA useful immediately without prescribing a skill or
subagent style. Project resources let a workspace customize behavior without
editing package files. Name-based replacement makes overrides explicit and easy
to inspect.

**How it works at a high level:**

- Defaults live under `agent/default_resources/` and are mounted read-only at
  `/mira-defaults/...`; only default memory and built-in tools are shipped
  there by default.
- Project resources live under the workspace's `.mira/` folder and are mounted
  at `/.mira/...`.
- `build_resources()` loads memories, skills, subagents, and tools, then passes
  the final lists to `create_deep_agent(...)`.
- Metadata keeps `source` and `replaces` fields so `/tools`, `/memories`,
  `/skills`, `/subagents`, and `/settings` can show what happened.

**Overwrite rules:**

- Memories load from `*.md` and replace by filename. A project
  `.mira/memories/AGENTS.md` replaces the bundled default `AGENTS.md`; extra
  Markdown files are added as additional memories.
- Skills load from project folders containing `SKILL.md`. MIRA display metadata
  keys them by YAML frontmatter `name`, falling back to the folder name. If
  bundled default skills are added later, DeepAgents receives default skill
  sources first and project skill sources second, so a duplicate skill name
  follows DeepAgents' later-source-wins behavior.
- Subagents load from Python files exporting `SUBAGENTS = [...]` and replace by
  each subagent's `name`. MIRA does not ship an opinionated default subagent.
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

**Decision:** Dangerous built-in tools require approval by default. Built-in
dangerous tools and project tools can be enabled or disabled through settings;
disabled project tools remain visible in metadata even when they are not
exposed to the agent. QuickJS programmatic tool calling is limited to
`ls`, `read_file`, `glob`, and `grep`; dynamic subagent delegation uses
QuickJS' top-level `task()` helper only when the System Settings toggle is
enabled, while destructive file tools, shell execution, and interrupt/control
flow tools stay outside that bridge.
MIRA also owns display of `ask_user` interrupts: the interrupt payload keeps
`question` and `options` separate, the TUI shows the question once, and options
render as vertical choice buttons with the open-ended fallback last. Larger
choice sets are allowed for compatibility and explicit user requests, but the
default tool prompt asks the agent to prefer 1-3 concise options. Prompt-panel
keyboard focus styles the selected button itself, not a parent row; ask_user
option buttons fill their row to provide a full-row selection feel without
extra focus bookkeeping.

**Why:** Approval prompts make file edits, eval, subagent delegation, and shell
execution transcript-compatible and user-controlled. Keeping disabled project
tools in metadata lets the settings UI manage them without exposing them to the
model, while disabled built-ins are hidden through the same excluded-tools path
MIRA uses for mode-specific tool visibility. `ask_user` stays a normal
LangGraph interrupt/resume path; MIRA only formats the prompt surface so user
decisions remain readable in narrow terminals.

**Where to check:** `config/settings.py`, `agent/factory.py`,
`agent/middleware.py`, `agent/resources/__init__.py`, `ui/interrupts.py`,
`runtime/runner.py`.

**Update this when:** Approval defaults, interrupt payload handling, or
settings-panel tool behavior changes.

## Planning Mode

**Decision:** Planning mode has a separate agent with `write_file`, `edit_file`,
`execute`, `task`, and `eval` hidden from the model; filesystem permissions also
deny writes as a backstop. This keeps shell execution, programmatic evaluation,
and delegation paths out of planning mode rather than relying on their normal
action-mode approval behavior. Planning mode still supports ordinary read-only
conversation, explanations, findings, brainstorming, and existing-plan recall.
Requests with implementation intent must produce a structured plan through the
`present_plan` tool once decision-complete, without requiring the user to ask
for the plan explicitly. Material questions must use `ask_user` choices rather
than an open-ended assistant message. At the start of each turn, the planning
prompt requires a semantic classification: safe conversation may end in prose,
while implementation intent must end through `ask_user` or `present_plan`.
Material decisions are defined generically: reasonable interpretations or
choices that produce meaningfully different outcomes, scope, audience,
priorities, behavior, presentation, constraints, resources, compatibility, or
risk. MIRA separates discoverable facts from preferences and resolves the
latter before dependent research. `ask_user` is an intermediate step whose
answer preserves implementation intent until `present_plan`, and user-supplied
alternatives remain separate choices. This policy applies to any planned
change, not only software work.
Each wrapped planning request repeats this terminal contract after the user's
text so lengthy repository research does not displace it from the model's final
output decision.
This uses the existing planning-agent call, not punctuation/regex
classification or an extra model-judge request.
Structured plans are shown as ephemeral plan bubbles with explicit Implement,
Revise, and Discard actions. Their compact one-row styling and visible keyboard
shortcut labels intentionally mirror prompt-panel choices without changing
PromptPanel itself. An active plan focuses Implement after mounting, Left/Right
wraps through its action row, and Escape returns focus to the prompt; shortcuts
remain local to the plan controls so typing cannot resolve a plan accidentally.
Clicking the active plan bubble restores its most recently focused action.
Discarding a plan resolves the bubble in place and returns keyboard focus to
the prompt so the next request can be typed immediately.
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

**Why:** Users need a mode where MIRA can converse and investigate safely while
still recognizing when an implementation-ready plan is the useful outcome.
Hiding every tool path that can mutate or delegate improves model behavior;
permissions provide a filesystem safety fallback. Prompt-level semantic policy
avoids brittle text heuristics. Plan execution remains an explicit user action,
not an automatic side effect of leaving planning mode.

**Where to check:** `agent/factory.py`, `agent/planning/policy.py`, `ui/app.py`,
`ui/repl.py`, `tests/test_plan_mode.py`, `tests/test_textual_app.py`.

**Update this when:** Planning mode gains or loses tools, changes how plan
bubbles are presented, resolved, or replayed into model context, or changes its
filesystem permissions.

## Goal-Driven Rubric Grading

**Decision:** Rubric grading is an opt-in workspace setting under
`system.rubric`, disabled by default with a three-iteration cap. Valid caps are
1 through 20. Only the action agent receives DeepAgents' `RubricMiddleware`,
using the active model, no grader tools, and the configured cap. The planning
agent and the criteria service never receive that middleware. Changing either
rubric setting rebuilds both agents and resolves any pending rubric proposal as
`settings changed` so its displayed cap cannot become stale.

MIRA keeps three invocation states distinct. Planning and rubric-disabled
action turns omit rubric state. Ordinary rubric-enabled action turns send
`rubric=None` to clear any checkpointed criteria. An approved goal or plan
sends its Markdown criteria. The middleware's injected revision messages stay
inside DeepAgents state and are not projected as user-authored session events.

**Proposal lifecycle:** `/goal <prompt>` is TUI-only because it requires an
interactive review bubble. A separate `GoalCriteriaService` creates or revises
Markdown criteria with a fresh instance of MIRA's configured model outside the
action graph. Goal proposals and rubric-enabled plan proposals store their id,
kind, original objective, structured resolved decisions, deterministically
derived effective objective, criteria, optional structured plan, iteration
cap, and status as explicit proposal events.

Rubric-enabled planning adds one hidden control interrupt, `prepare_goal`. The
planning agent calls it only after read-only research and material `ask_user`
decisions are complete. A focused `PlanningStageMiddleware` owns a checkpointed
`research | finalize` state using LangChain's middleware state schema. It keeps
the compiled tool registry stable while filtering each model request: research
hides `present_plan`; after the interrupt, the runner resumes with a native
LangGraph `Command` that also updates the stage to `finalize`; finalization
exposes only `present_plan` and sets the provider-portable `required` tool
choice. Requiring a call is deterministic with one exposed tool and avoids the
named-tool object rejected by some OpenAI-compatible providers. The wrapped
research request repeats a stage-specific `ask_user`/`prepare_goal` terminal
contract instead of the legacy `present_plan` reminder. MIRA then
generates criteria and resumes the same planning thread to produce the plan
alone. Revision first sends feedback to criteria revision, then starts a new
planning thread directly in finalization with the same feedback, revised
criteria, original objective, and previous plan. This avoids graph rebuilding,
message parsing, duplicate plan tools, and prompt-only ordering. Disabled
planning does not install the stage middleware and keeps the legacy prompt,
tools, plan event, rendering, and action handoff unchanged.

Criteria generation and plan finalization can each involve a silent model call.
The TUI reuses its transient animated waiting block with phase-specific labels
for Definition-of-Done drafting/revision and plan drafting; these blocks are not
persisted as transcript events.

**Streaming and persistence:** One custom-event dispatcher independently
routes QuickJS Eval subagent events and DeepAgents rubric start/end events.
Rubric passes are displayed one-based and include pass counts, failed criteria,
gaps, and terminal verdicts. For DeepAgents 0.6.12, MIRA reads completed
checkpoint `_rubric_status` because the final streamed event can still say
`needs_revision` when the cap was reached; newer terminal statuses are accepted
directly. Starts are transient. Completed evaluations are durable rubric
events, never tools. TUI results update in place, while one-shot and trace
surfaces emit concise blocks. Rubric colors are centralized as `#C58FD6` for
headers/borders and `#F1DCF5` for body text and are isolated to rubric UI.

**Why:** Users can agree on observable completion conditions before work while
DeepAgents continues to own iterative grading and revision. Separating the
objective, decisions, criteria, and plan avoids flattening recoverable state or
making the planning agent grade its own output. Opt-in construction preserves
the existing workflow when disabled.

**Where to check:** `agent/planning/criteria.py`, `agent/planning/proposals.py`,
`agent/factory.py`, `agent/middleware.py`, `agent/default_resources/tools/prepare_goal.py`, `runtime/rubric_events.py`,
`runtime/runner.py`, `session/context.py`, `session/recorder.py`, `ui/app.py`,
`ui/repl.py`, and `ui/widgets/chat_log.py`.

**Update this when:** Rubric ownership, criteria prompts, proposal persistence,
terminal-status handling, or the goal/plan review lifecycle changes.

## Textual TUI And One-Shot Output

**Decision:** The Textual TUI is the primary interactive experience. One-shot
terminal output uses a separate renderer. One-shot mode accepts literal prompt
text through `--prompt/-p` or explicit Markdown prompt files through
`--file/-f`.

**Why:** The TUI can preserve chat order, tool calls, tool results, subagent
progress, settings, and session history in one place. The one-shot renderer
stays simpler for scripts and quick prompts.

On Windows, MIRA pins Textual 8.2.7 and selects a narrow Windows driver adapter.
Textual's Win32 event monitor normally reduces each `KEY_EVENT_RECORD` to its
Unicode character before parsing, which loses the Shift state on Return in
classic Console Host. MIRA preserves Textual's existing parsing for every
other record, but encodes raw Shift+Return as Textual's enhanced
`shift+enter` sequence before that state is discarded. This boundary could
normalize other raw modifier combinations later, but MIRA currently limits it
to Shift+Enter.

The prompt owns only Enter submission and Shift+Enter newline insertion. The
application's priority Ctrl+C binding first copies Textual's screen-level
rendered selection, then falls back to the focused widget's internal selection,
and quietly consumes the shortcut if neither exists. Windows clipboard writes
use `CF_UNICODETEXT` directly while keeping Textual's in-process clipboard in
sync, so copying does not depend on terminal selection mode or OSC 52 support.
Non-Windows launches retain Textual's default driver and clipboard behavior.

TUI-only commands that need live app state stay in `ui/app.py`; for example,
`/settings` persists workspace settings before rebuilding agents, while
`/reload` reloads `.env`, current settings, and project resources before
rebuilding agents without restarting the session. Read-only process and agent
inspection is split across `/runtime`, `/tools`, `/memories`, `/skills`, and
`/subagents`; each command renders one focused section without rebuilding agents
or making a model/network request. `/help` keeps every command in one table but
groups related commands under visually distinct soft-blue section headers.
`/session` stays separate because it
summarizes durable conversation state, including active goals and plans.
`/new-chat` and the sidebar
`+ New` action create and switch to a fresh saved session without deleting the
current session. `/compact` is also TUI-only because it needs the active agent,
thread, checkpoint, and session store; it runs outside a normal model turn and
does not create synthetic assistant or tool messages.
The subagents bottom panel is live TUI state only. It opens for running
subagents and renders task, status, and elapsed time as fixed single-line
columns; task text yields width first and truncates with `...` when needed.
While work is active, `[-]`/`[+]` collapses the panel to an animated summary and
the close control is hidden. New subagent activity reopens the panel. Once all
rows are terminal, `x` becomes available; completed state collapses before the
next prompt and is reset by later subagent activity.
While the panel owns live subagent progress, the chat log suppresses separate
task delegation and subagent bubbles so the running turn has one live progress
surface. The status line may briefly report delegation setup, but the task rows
belong in the panel.
Eval-created subagents are grouped in that panel by internal `eval_id`, but the
UI labels them as `Group 1`, `Group 2`, and so on.

**Where to check:** `ui/app.py`, `ui/windows_input.py`,
`ui/windows_driver.py`, `ui/windows_clipboard.py`, `ui/widgets/`,
`ui/renderer.py`, `runtime/*_events.py`, `tests/test_textual_app.py`.

**Update this when:** Rendering responsibility moves, a new UI mode appears,
keyboard or clipboard ownership changes, or tool/subagent events are projected
differently.

## Sessions And Compaction

**Decision:** MIRA stores durable session JSON for replayable UI history, while
DeepAgents handles runtime context counting and compaction.

**Why:** Session files should be stable user-facing history after restart.
Starting a new chat is therefore non-destructive: MIRA creates another session
record and makes it active instead of clearing the previous one.
Runtime compaction is agent-execution behavior and belongs to DeepAgents. MIRA
installs a named `MiraSummarizationMiddleware` subclass built from DeepAgents'
summarization defaults, then observes that middleware's `_count_tokens` result
so the UI can show context pressure. MIRA does not run a parallel dashboard
counter or compute provider prompt tokens. Automatic and agent-selected
eligibility remain DeepAgents decisions. The explicit TUI `/compact` command is
the narrow exception: it reuses the attached summarization middleware to apply
the normal retention policy immediately, then writes the same
`_summarization_event` consumed by subsequent DeepAgents model calls. Provider
`In` and `Out` usage are cumulative per-call totals, not current context
occupancy. ChatAnyLLM reports usage but omits the matching `model_provider`
response metadata required by DeepAgents' reported-token validation. MIRA's
model-response normalization fills only that missing integration identity and
leaves DeepAgents' eligibility thresholds unchanged.
DeepAgents marks summary-model invocations with `lc_source="summarization"`.
MIRA observes that invocation metadata before LangGraph publishes each message
stream and drains marked streams without rendering or recording their internal
reasoning or summary text. Compaction classification never depends on model
wording, prompt fragments, or task-local flags; unmarked reasoning and replies
remain visible even when they discuss compaction or summarization.
Regular subagent completions remain durable `subagent` transcript events so
past sessions can replay them and resume context can refer to their outputs.
The live panel's open/collapsed/closed state and row layout are intentionally
not persisted or replayed. Reopening a session projects those durable subagent
events back into chat transcript blocks rather than reconstructing the old live
panel. Eval-created subagent rows are not stored separately; their durable
history is the surrounding eval tool call/result plus the assistant's summary.

**Where to check:** `agent/compaction.py`, `agent/middleware.py`, `session/store.py`,
`session/context.py`, `session/recorder.py`, `session/dashboard.py`,
`runtime/context_usage.py`, `runtime/message_metadata.py`.

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
