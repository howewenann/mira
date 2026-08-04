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
The always-on Plan/Goal response-status protocol has one workspace System Setting,
`system.planning_response_status.max_retries` (default `2`, range `1`-`20`). It
counts recovery calls after the first rejected response. Changing it follows
the normal settings application path and rebuilds both agents; it is a bound,
not a feature toggle.
Main and rubric LLM profiles are environment configuration. Both accept strict
JSON-object `MODEL_KWARGS`, which `ChatAnyLLM.model_kwargs` passes to AnyLLM as
completion parameters after MIRA rejects runtime-owned call fields. Explicit
temperature, maximum-token, and top-p variables take precedence over duplicate
JSON keys. With no rubric overrides, `RubricMiddleware` receives the exact main
model object as before. A rubric profile on the same provider inherits blank
main fields and shallow-merges JSON kwargs; changing the rubric provider starts
a clean profile, requires its own model, and inherits no credentials, endpoint,
sampling configuration, or arbitrary kwargs.
A missing `.mira/settings.yml` starts from complete defaults. An existing file
must equal the complete normalized settings shape, so partial, malformed,
unknown, or invalid settings are rejected instead of silently defaulted.

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
`agent/factory.py`, `agent/middleware/`, `ui/app.py`.

**Update this when:** `execute` is exposed by a different backend, default
approval behavior changes, or shell environment inheritance changes.

## Project Resources

**Decision:** MIRA loads bundled defaults first, then project `.mira/` resources.
Bundled defaults stay minimal: default memories plus MIRA's built-in project
tools. Project resources replace defaults by memory filename, skill name,
subagent name, or tool name.

**Why:** Defaults make MIRA useful immediately without prescribing a skill or
subagent style. Project resources let a workspace customize behavior without
editing package files. Name-based replacement makes overrides explicit and easy
to inspect.

**How it works at a high level:**

- Defaults live under `agent/default_resources/` and are mounted read-only at
  `/mira-defaults/...`; only default memories and built-in tools are shipped
  there by default.
- Project resources live under the workspace's `.mira/` folder and are mounted
  at `/.mira/...`.
- `build_resources()` loads memories, skills, subagents, and tools, then passes
  the final lists to `create_deep_agent(...)`.
- Metadata keeps `source` and `replaces` fields so `/tools`, `/memories`,
  `/skills`, `/subagents`, and `/settings` can show what happened. Tool
  projections also retain their MIRA-or-Project runtime boundary and the
  concrete Python environment used for execution.

**Overwrite rules:**

- Memories load from `*.md` and replace by filename. For example, a project
  `.mira/memories/AGENTS.md` replaces the bundled `AGENTS.md`, and a project
  `.mira/memories/software-development.md` replaces the bundled software
  development guide; extra Markdown files are added as additional memories.
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
- Default tool-file failures remain fatal because they indicate a broken MIRA
  installation. Project tool files are attempted independently and failures
  are retained as structured `ToolLoadFailure` values. Only tools from files
  that finish loading enter merge, settings, approval, schema, or agent paths;
  a failed file never creates a disabled placeholder. `/reload` retries all
  files, and one-shot mode prints one grouped warning while continuing with the
  successful subset. The TUI shows one grouped startup warning and keeps
  unresolved failures behind its persistent Issues entry point. Explicit
  `/reload` repeats a warning while failures remain. Recovery and
  Issues-driven installation/reload do not toast.
- Standard LangChain `@tool` runs in MIRA's interpreter. The dependency-free
  `mira_tool_api.project_tool` decorator is an explicit alternative: it leaves
  the original callable unchanged and adds versioned metadata. The loader
  derives its LangChain schema from the original signature and registers a
  `StructuredTool` proxy under the public name. Imported marked functions are
  ignored, and project-only imports must stay inside the marked function
  because discovery still imports the containing file in MIRA.
- A project proxy launches one standard-library child runner per call using the
  existing Execute Environment selection. It exchanges JSON through temporary
  request/response files, uses the workspace as cwd and import root, and loads
  the exact standalone `mira_tool_api.py` bridge file as `mira_tool_api` before
  importing the source. This exposes neither MIRA's site-packages nor a second
  environment setting. JSON-compatible results are preserved; other values
  fall back to `repr`, and child exceptions become normal project-runtime tool
  errors. Before crossing the child boundary, conventional structured path
  arguments (`path`, `*_path`, `paths`, and `*_paths`) reuse the project
  backend's virtual-path resolver so `/file` refers to the workspace rather
  than the host filesystem root.
- A normal `@tool` that delays a missing import until its function body loads
  successfully and remains outside startup repair; that invocation follows the
  ordinary tool-error path.
- Disabled project tools stay in metadata for the settings UI but are not
  exposed to the agent.

**Where to check:** `mira_tool_api.py`, `agent/resources/tools.py`,
`agent/resources/tool_failures.py`, `agent/resources/project_tools.py`,
`agent/resources/project_tool_runner.py`, `agent/default_resources/`,
`tests/test_resources.py`, `tests/test_project_tools.py`.

**Update this when:** Resource locations, overwrite rules, display metadata, or
supported export shapes change.

## Tools And HITL

**Decision:** Dangerous built-in tools require approval by default. Built-in
dangerous tools and project tools can be enabled or disabled through settings;
disabled project tools remain visible in metadata even when they are not
exposed to the agent. Newly discovered project tools are enabled by default but
require approval unless explicitly configured as always allowed. QuickJS
programmatic tool calling is limited to
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
`agent/middleware/`, `agent/resources/__init__.py`, `ui/interrupts.py`,
`runtime/runner.py`.

**Update this when:** Approval defaults, interrupt payload handling, or
settings-panel tool behavior changes.

## Planning Mode

**Decision:** `/plan` enters one persistent read-only Plan conversation;
`/plan <prompt>` enters that same mode and submits the suffix through the normal
Plan message path. Imperative language never enables execution. The Plan prompt
uses four non-persisted outcomes: `DISPLAY_RETAINED` immediately calls the
applicable `show_plan` or `show_goal` control, `DISCUSSION` returns ordinary
read-only prose, `NEEDS_DECISION` calls `ask_user`, and `PLAN_READY` calls
`prepare_plan`. Display requests do not research, reproduce retained work in
prose, prepare a replacement, or enter finalisation first. There
is no classifier model, structured response format, prose regex, or semantic
grader.

Both Plan and Act compose one shared question policy: discover facts from
available context first and use `ask_user` whenever user input is required.
Plan mode additionally hides write, delete, execute, evaluation, and delegation
tools and denies filesystem writes as a backstop.

Formal construction is always criteria-first:

```text
prepare_plan
-> SuccessCriteriaService
-> forced finalize_plan
-> Plan bubble
```

The staging middleware exposes `prepare_plan` during conversation and only
`finalize_plan` during finalisation, with required tool choice. The generated
Success Criteria are binding context. The model supplies Title, Key Changes,
Test Plan, and Assumptions around the staged Objective and Context and
Constraints; no Summary field exists. This path and its model inputs are
identical whether automatic rubric evaluation is enabled or disabled.

The user's request is authoritative for meaning, while `prepare_plan` may
rewrite the visible Objective for clarity. That rewrite is wording-only: it
must not add, remove, or materially change the intended outcome, scope,
deliverables, or constraints. The preparation handoff is trusted without a
second semantic grader or model call; the original request remains in the Plan
conversation as binding context.

Plan research also uses a deterministic terminal response-status contract. The
generic `CorrectionMiddleware` appends the Plan-owned rule's contract
transiently to every model request, including requests after tool results,
because a reminder stored only on the initial user message becomes stale as the
tool loop grows. Research keeps optional tool choice. A response with any tool
call bypasses correction. At a natural no-tool stop, the final non-empty textual
line must be exactly one stage-valid `RESPONSE_STATUS` line; only
`RESPONSE_STATUS: COMPLETE` may terminate. `NEEDS_RESEARCH`,
`NEEDS_USER_INPUT`, and `READY_TO_PREPARE_PLAN` retain the rejected assistant
response, append a named correction prompt, and use LangGraph's native jump to
the model node without replaying completed tools. Missing, duplicated,
malformed, non-terminal, empty, reasoning-only, or Goal-only statuses take the
same bounded recovery path with general feedback. Finalisation is isolated from
this contract and remains a forced single `finalize_plan` call.

The retry cap counts retries after the initial rejected candidate. Rejected
prose and its correction prompt remain paired in checkpoint and durable visible
history so neither the user nor the model sees a hanging correction. A generic
`Response check` bubble shows the workflow, failed check, exact retry prompt,
and retry count. Its rule-owned display name keeps the middleware free of
Plan/Goal vocabulary.
Exhaustion retains the last candidate, shows the failed check and exhausted
limit, then appends an explicit incomplete response. The exact status line is
ordinary assistant text: it remains visible and byte-identical in model state,
checkpoints, sessions, reload context, one-shot output, TUI output, and traces.
There is no separate accepted-status event or bubble. This reliability check is
always active and does not call or mutate user rubrics. It deliberately trusts
an exact `COMPLETE` classification: a dishonest model can still label unfinished
prose as complete. Provider-specific channels, English false-progress
heuristics, and always-on semantic grading were rejected because they are less
portable or would add a second model judgment.

When a Plan is current, `current_plan` stores its stable id, Plan fields,
separate Success Criteria, status, rubric policy and cap, latest overall rubric
result, completion source, attempts, and timestamps while `current_goal` is
null. Supported
statuses are `proposed`, `active`, `paused`, `max_iterations_reached`, and
`completed`. Transcript Plan bubbles remain immutable history; resume context
labels only the populated current artifact as authoritative.
Session normalization requires the exact current `current_plan` fields and
types. A populated malformed or retired Plan artifact rejects the session;
retired Summary, generic-stage, and pre-staging transcript event shapes are
not projected as Plan events.

The Plan bubble uses Plan colours for Plan content, rubric colours for Success
Criteria, and muted text for automatic-evaluation policy and status. Its actions
are Implement, Revise, and Close. Implement starts or restarts the exact Plan in
Act mode. Revise stays in the persistent Plan conversation, creates a complete
replacement, and calls `SuccessCriteriaService.revise()` for both rubric
settings; approach-only feedback preserves criteria. Close hides controls
without changing or deleting `current_plan`.

`/plan-show` and the read-only `show_plan` control tool call the same renderer
and make no Plan-generating model call. `/plan-clear` removes only
`current_plan`. `/plan-resume` accepts every incomplete state, including a
never-run `proposed` Plan, switches to Act, and continues immediately; completed
Plans must be deliberately reopened and implemented again.

Rubric-disabled successful attempts complete as `agent-declared`. Rubric-enabled
attempts pass the exact Success Criteria to DeepAgents: `satisfied` completes as
`rubric-verified`, `needs_revision` continues upstream iteration,
`max_iterations_reached` remains resumable, and runtime, grader, or cancellation
failures pause the Plan. Detailed evaluations remain separate rubric events.

**Why:** This preserves safe conversational planning while making the final
artifact exact, durable, criteria-led, and independently recallable after
compaction or reload. Rubrics evaluate execution rather than changing what Plan
MIRA constructs.

**Where to check:** `agent/planning/policy.py`,
`agent/planning/criteria.py`, `agent/middleware/`, `session/plans.py`,
`runtime/runner.py`, `ui/app.py`, `ui/repl.py`, and
`ui/widgets/chat_log.py`.

**Update this when:** Plan tools, construction stages, `current_plan` schema,
review actions, recall commands, or execution completion rules change.

## Durable Criteria-Only Goals

**Decision:** Plan and Goal are alternative forms of formal work. A Plan stores
an Objective, prescribed approach, and Success Criteria. A Goal stores only an
Objective and Success Criteria, leaving the approach to the Act agent. Sessions
have explicit `current_plan` and `current_goal` fields and normalization rejects
a record when both are populated; timestamps never pick a winner.

`/goal <prompt>` uses a dedicated read-only planning thread without changing
the current Plan/Act mode: optional investigation and `ask_user`, then
`prepare_goal` -> `SuccessCriteriaService` -> forced `finalize_goal` ->
`GoalBubble`.

Goal research exposes discovery tools, `ask_user`, `prepare_goal`, `show_plan`,
and `show_goal`. Finalization exposes only `finalize_goal` with required tool
choice. The raw `/goal` request remains authoritative for meaning in transient
staging. `prepare_goal` may provide a concise, user-facing Objective rewrite,
but may not add, remove, or materially change the request's intended outcome,
scope, deliverables, or constraints; a blank Objective falls back to the raw
request. Both texts are passed as binding context to criteria generation and
finalization, while only the polished Objective is persisted in the existing
Goal artifact. Bounded evidence may clarify but not enlarge either. Failed or
cancelled generation leaves current formal work unchanged. Goal construction
never calls `finalize_plan` and never produces a hidden implementation Plan.

Goal research uses the same transient natural-stop protocol and bounded retry
state as Plan research, but its only preparation marker and tool are
`RESPONSE_STATUS: READY_TO_PREPARE_GOAL` and `prepare_goal`. Plan preparation is
not exposed and a Plan-only status is malformed. Conversely, Goal preparation
is absent from Plan research. Goal finalisation remains outside validation and
retains the single required `finalize_goal` call.

`SuccessCriteriaService` receives the raw authoritative request alongside the
polished Objective for initial Goal construction, plus optional research
context. It receives the effective Objective, previous criteria, feedback, and
optional research context during revision, never a previous Plan. There is no
semantic grading call between preparation and finalization; a material
Objective change during revision is permitted only when explicit feedback
changes the desired outcome.

The durable `GoalArtifact` stores id, title, objective, Success Criteria,
status, snapshotted rubric policy and cap, latest overall rubric result,
completion source, attempts, and timestamps. Its statuses are `proposed`,
`active`, `paused`, `max_iterations_reached`, and `completed`. Dedicated `goal`
events preserve exact artifacts. Session normalization requires the exact
current `current_goal` fields and types; a populated malformed or retired Goal
artifact rejects the session. Retired proposal events are not projected as
Goal events.

`GoalBubble` shows Objective and Success Criteria with Implement, Revise, and
Close actions. A newly finalized `proposed` Goal remains active through the
post-command refresh so all three review actions stay available. Implement
starts or restarts one explicit Act attempt. Revise uses the read-only Goal
pipeline and `SuccessCriteriaService.revise()` to create a complete replacement.
Close hides controls without changing `current_goal`.
`/goal-show` and `show_goal` share the exact renderer, `/goal-resume` accepts
incomplete states, and `/goal-clear` removes only the current Goal.
Explicit Goal recall always reopens the retained artifact for review, including
`paused`, `max_iterations_reached`, and `completed` Goals. Ordinary model-turn
cleanup must not resolve that newly rendered bubble. Only the dedicated Goal
execution path resolves its execution bubble after completion, pausing,
cancellation, failure, or rubric exhaustion.

MIRA replaces current formal work only after the new Plan or Goal is presented
successfully. Completed work may be replaced automatically. Incomplete work
requires structured confirmation named for the current artifact, for every
Plan-to-Plan, Plan-to-Goal, Goal-to-Plan, and Goal-to-Goal combination. The
dialog explains which new kind will replace which current kind. Acceptance is
transient and does not clear the old artifact early. Successful replacement
marks the old transcript event `superseded`, clears the opposite current field,
and stores the new artifact.

Goal construction is identical with rubrics on or off. A non-rubric successful
attempt completes as `agent-declared`. A rubric-enabled attempt passes exact
Success Criteria to DeepAgents: `needs_revision` continues, `satisfied`
completes as `rubric-verified`, and `max_iterations_reached` remains resumable.
Runtime, grader, or cancellation failures pause the Goal.

**Streaming and persistence:** One custom-event dispatcher independently
routes QuickJS Eval subagent events and DeepAgents rubric start/end events.
Rubric passes are displayed one-based. While DeepAgents grades, the TUI shows an
animated spinner plus grader identity and monotonic elapsed time; an interactive
one-shot terminal refreshes its elapsed line once per second, while redirected
output and traces record only start and completion. The completed event includes
the grader identity, duration, summary, every native model-generated criterion
name, passed/failed marks, failure gaps, and terminal verdict. MIRA does not
rename criteria to match Goal Success Criteria or introduce a second grading
call. MIRA reads completed checkpoint `_rubric_status` because the final
streamed event can still say `needs_revision` when the cap was reached; newer
terminal statuses are accepted directly. Starts and animation ticks are
transient. Completed evaluations, including identity and duration, are durable
rubric events, never tools. Rubric colors are centralized as `#C58FD6` for
headers/borders and `#F1DCF5` for body text and are isolated to rubric UI.

**Why:** Outcome-focused work should not force users to approve an approach.
Keeping Goals criteria-only makes execution flexible while retaining the same
durability, review, recall, replacement, and evaluation guarantees as Plans.

**Where to check:** `agent/planning/criteria.py`, `agent/factory.py`,
`agent/middleware/`, `agent/default_resources/tools/prepare_goal.py`,
`agent/default_resources/tools/finalize_goal.py`, `runtime/runner.py`,
`session/goals.py`, `session/context.py`, `ui/app.py`, `ui/repl.py`, and
`ui/widgets/chat_log.py`.

**Update this when:** Goal construction, persistence, replacement, review,
recall, migration, or execution completion rules change.

## Model Correction And Control-Tool Recovery

**Decision:** MIRA treats deterministic response correction and semantic Rubric
grading as separate harness capabilities. `CorrectionMiddleware` owns only the
generic no-tool natural-stop lifecycle: transient reminders, rule selection,
bounded counters, correction events, model feedback, retry jumps, acceptance,
and exhaustion. Workflow rules under `agent/planning/` define Plan/Goal response
statuses, checks, prompts, and terminal failure text. Rubrics remain optional semantic
review by a grader model and share no implementation with deterministic
correction.

Planning stages have two matching enforcement layers. Request-time filtering
shows only the formal controls valid for the current stage. A native
`wrap_tool_call` guard rejects remembered or hallucinated wrong-stage formal
controls before their handlers can call `interrupt()`. Finalization is stricter:
`plan_finalize` permits only `finalize_plan`, and `goal_finalize` permits only
`finalize_goal`. That execution boundary covers registered controls, ordinary
tools, cross-workflow controls, and stale or unregistered names. The standard
agent edge returns an error `ToolMessage` with the original call id and exact
repair guidance, so the model may select the finalizer. These calls never
consume natural-stop correction retries. A valid finalizer still reaches
ToolNode, whose schema validation can return its own repairable error before
the finalizer's interrupt body runs.

**Why:** A weak model may call a registered control remembered from transcript
history even when its schema is hidden from the current request. Tool visibility
is guidance, not an execution boundary. LangGraph deliberately bubbles
`GraphInterrupt` out of ToolNode and checkpoints the suspended tool. An error
raised later by MIRA's UI control surface is outside ToolNode, so re-raising it
leaves no error `ToolMessage` for the model and the invocation remains pending
at the interrupt. Model-correctable stage and argument failures must therefore
use native tool errors before interruption. Unexpected criteria-service,
persistence, or renderer failures remain fatal because another model attempt
cannot repair them.

The middleware package mirrors these ownership boundaries:
`builder.py` returns the ordered `AgentMiddlewareBundle`; `correction.py`
implements generic correction; `planning_stage_enforcement.py` enforces formal
stages; and `execute_tool_description_rewrite.py`,
`model_tool_visibility.py`, and `model_response_normalization.py` each name the
behavior they own. Correction events are durable transcript events and are
projected as correction context, not user context, when a saved session is
resumed. Accepted response statuses need no special projection because they are
preserved as ordinary assistant prose.

The authoritative retained artifacts are the session's `current_plan` and
`current_goal`; LangGraph checkpoints instead retain model messages, planning
stage, and interrupt/resume state. Natural-language recall enters the graph so
the model can choose `show_plan` or `show_goal`, after which the control surface
reads the artifact directly from the session. `/plan-show` and `/goal-show`
bypass model interpretation and read the same fields.

**Where to check:** `agent/middleware/`, `agent/planning/response_status.py`,
`runtime/correction_events.py`, `runtime/runner.py`, `session/context.py`.

**Update this when:** A new correction rule is added, correction visibility or
persistence changes, a control tool gains a stage precondition, or recoverable
and fatal control-surface errors are reclassified.

## Textual TUI And One-Shot Output

**Decision:** The Textual TUI is the primary interactive experience. One-shot
terminal output uses a separate renderer. One-shot mode accepts literal prompt
text through `--prompt/-p` or any readable, non-empty UTF-8 text file through
`--file/-f`. Optional invocation-only rubric text comes from `--rubric` or
`--rubric-file`; it enables the existing rubric middleware in memory for that
one-shot run, uses the saved iteration cap, and does not persist a settings
change or formal-artifact state.

**Why:** The TUI can preserve chat order, tool calls, tool results, subagent
progress, settings, and session history in one place. The one-shot renderer
stays simpler for scripts and quick prompts.

All tools share one underlying lifecycle for call identity, visible start,
completion or error, persistence, and replay. This includes the control
tools `ask_user`, `prepare_goal`, `prepare_plan`, `finalize_goal`, `finalize_plan`,
`show_goal`, and `show_plan`: their dedicated surfaces remain, but
no longer suppress the ordinary call/result block. The stable call id associates each surface outcome
with its original call, so the completed result updates that block in place and
two calls with identical output remain distinct.

Completed tool results update their original tool blocks before the overall turn
ends when the provider exposes a live terminal event. MIRA also consumes root
v3 `values` snapshots during every invocation and resume. The first snapshot is
the invocation baseline; later snapshots are compared by call id, with counted
name/output identities for idless error messages. A newly appended native
`ToolMessage(status="error")` is authoritative: MIRA recovers its matching AI
tool call from graph state when necessary, then sends both through the same
live recording and rendering path. This covers middleware-short-circuited and
unregistered calls that never create a tool-call completion handle. Repeated
full-state snapshots do not duplicate either event.

Successful Plan/Goal and `ask_user` controls retain dedicated interrupt-driven
surfaces. Pinned LangGraph exposes successful `interrupt()` control flow through
the raw `ToolCallStream.error` field as an `Interrupt(...)`, which cannot be
distinguished there from failure. MIRA therefore suppresses raw control-stream
errors, `Command` values, interrupt payloads, and empty interrupted completions;
only the native error `ToolMessage` drives an ordinary visible error. Argument
validation and middleware rejection produce that native message. Unexpected
service, persistence, and UI failures remain fatal.

Final graph-state recovery remains the fallback for providers that omit a live
terminal projection. Before each top-level turn MIRA
snapshots the checkpoint's existing lifecycle, then subtracts that occurrence-
aware baseline from the final graph state. Only lifecycle entries introduced by
the current turn are eligible for recovery, so complete checkpoint history
cannot recreate already-saved tool bubbles. Stable call ids are preferred;
idless entries use normalized call/result fingerprints and counts. If an
established checkpoint cannot be read, recovery is skipped rather than risking
historical duplication. Reopened sessions replay saved events in their order.
Native tool failures use the same path with explicit error status: terminal
watchers get a bounded chance to publish an already-observed error before an
overall stream failure propagates, and saved sessions replay the result with an
inline error label. Provider drafts without their final call id are promoted
FIFO by tool name when the stable id arrives, while distinct stable ids remain
separate retry attempts.

Harness response checks use the system/status blue palette in both Textual and
terminal traces. Warnings use a distinct orange palette rather than the user's
gold, while errors remain red and formal artifacts retain their own colors.

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

Windows also replaces Textual's fractional scrollbar-edge glyphs with solid
cells. Textual still owns scrollbar sizing, colors, mouse actions, and scroll
state; MIRA changes only the renderer's sub-cell glyph table. This avoids tofu
or mojibake in older console/font combinations without changing non-Windows
rendering.

TUI-only commands that need live app state stay in `ui/app.py`; for example,
`/settings` persists workspace settings before rebuilding agents, while
`/reload` reloads `.env`, current settings, and project resources before
rebuilding agents without restarting the session. Read-only process and agent
inspection is split across `/runtime`, `/tools`, `/memories`, `/skills`, and
`/subagents`; each command renders one focused section without rebuilding agents
or making a model/network request. `/help` keeps every command in one table but
groups related commands under visually distinct soft-blue section headers.
`/session` stays separate because it
summarizes durable conversation state, including the current Goal or Plan.
`/new-chat` and the sidebar
`+ New` action create and switch to a fresh saved session without deleting the
current session. `/compact` is also TUI-only because it needs the active agent,
thread, checkpoint, and session store; it runs outside a normal model turn and
does not create synthetic assistant or tool messages.
Unavailable project tool files use one narrow TUI-only issue flow rather than a
general notification system. `App.notify()` announces a new failure fingerprint
once per session, a compact `Issues N` button and `/issues` open one scrollable
`ModalScreen`, and a Textual thread worker runs one explicit
`sys.executable -m pip install ...` argument list. The input and both bottom-row
buttons are disabled while pip runs. Success reuses `_reload_runtime()` and
refreshes the open modal; failure retains captured output and re-enables retry.
Closing never changes settings, session history, or unresolved failures.
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

**Where to check:** `ui/app.py`, `ui/widgets/tool_issues.py`, `ui/windows_input.py`,
`ui/windows_driver.py`, `ui/windows_clipboard.py`, `ui/widgets/`,
`ui/renderer.py`, `runtime/*_events.py`, `tests/test_textual_app.py`.

**Update this when:** Rendering responsibility moves, a new UI mode appears,
keyboard or clipboard ownership changes, or tool/subagent events are projected
differently.

## DeepAgents Runtime Ownership

**Decision:** MIRA pins DeepAgents 0.7.3 and `langchain-quickjs` 0.3.5. MIRA
owns a small general-purpose action prompt and its existing planning prompt;
DeepAgents owns the filesystem, delegation, streaming, and middleware
execution. Project and bundled memory files remain opaque Markdown resources,
resolved by the same deterministic resolver and passed identically to action
and planning agents. Existing bundled-resource slots retain their precedence;
new project files retain the resolver's deterministic order.

Planning todos are optional and disabled by default because the upstream
DeepAgents [evaluation](https://github.com/langchain-ai/deepagents/pull/4929)
found no statistically significant accuracy improvement.
Enabling the workspace setting adds one `TodoListMiddleware` to action and
planning agents and to compiled dynamic subagents. `/tools` is derived from the
actual middleware stack, so toggling or reloading cannot retain a stale or
duplicate `write_todos` entry.

DeepAgents 0.7 defines `write_file` as create-or-complete-replacement and
`edit_file` as targeted replacement. MIRA keeps both behind the normal HITL
policy, labels their different consequences in the approval surface, and
exposes recursive `delete` only where the selected backend implements it.
Delete is action-only and destructive; like other consequential tools, it uses
the workspace's configurable HITL policy. Filesystem backends use explicit
virtual paths rooted in the selected workspace.

Rubric terminal events are accepted directly from DeepAgents, including
`max_iterations_reached`. MIRA preserves the explanation and available grader
diagnostics instead of synthesizing another iteration. QuickJS runs with
explicit memory, timeout, thread-persistence, and read-only PTC limits; it does
not receive ambient filesystem, network, process, or clock access.

**Where to check:** `agent/factory.py`, `agent/middleware/`,
`agent/subagent_compilation.py`, `agent/tools/specs.py`, `runtime/runner.py`,
`runtime/rubric_events.py`, `config/settings.py`.

**Update this when:** DeepAgents or QuickJS versions change, prompt ownership
moves, todo defaults change, or filesystem/rubric semantics change.

## Sessions And Compaction

**Decision:** MIRA stores durable session JSON for replayable UI history, while
DeepAgents handles runtime context counting and compaction.

**Why:** Session files should be stable user-facing history after restart.
Starting a new chat is therefore non-destructive: MIRA creates another session
record and makes it active instead of clearing the previous one.
Session records require the current fields; missing or conflicting formal
artifacts are rejected rather than repaired. Transient fields are not saved.
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
MIRA reads compaction summary prose only from the canonical
`_summarization_event.summary_message`; retired raw `summary` and
`summary_text` aliases are ignored.
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

**Where to check:** `agent/compaction.py`, `agent/middleware/`, `session/store.py`,
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
